import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import gymnasium as gym
import tyro
from tqdm import tqdm
import zarr  # Import the zarr library

from adapt_drones.cfgs.config import Config
import adapt_drones.utils.rotation as rotation
from adapt_drones.networks.agents import RMA_DATT
from adapt_drones.networks.adapt_net import AdaptationNetwork

# Set the backend for headless rendering
os.environ["MUJOCO_GL"] = "egl"

@dataclass
class Args:
    """Script arguments for data collection."""
    env_id: str = "traj_v3"
    wind_bool: bool = True
    seed: int = -1
    total_timesteps: int = 100000
    # Updated output path to reflect the Zarr format
    output_path: str = "data/drone_val_dataset.zarr"


def collect_data():
    """
    Main function to collect drone dynamics data using an expert policy.
    Saves the data in the PushT Zarr format.

    The saved data structure will be:
    - `data/state`: Corresponds to the model's observation input.
        Each row is [pos(t), vel(t), quat(t), action(t-1)].
    - `data/action`: Corresponds to the model's prediction target.
        Each row is the residual dynamics H(t).
    - `meta/episode_ends`: An array of indices marking the end of each episode.
    """
    args = tyro.cli(Args)

    # If seed is negative, generate a random one
    args.seed = random.randint(0, 2**32 - 1) if args.seed < 0 else args.seed

    print("Loading config to access scaling laws...")
    cfg_for_laws = Config(
        env_id=args.env_id, seed=args.seed,
        agent="RMA_DATT", scale=True, wind_bool=args.wind_bool
    )

    print("--- Data Collection for DroneDiffusion (PushT Zarr Format) ---")
    print(f"Aligning data to predict H(t) from state(t) and action(t-1)")
    print(f"Using expert policy: 'snowy-lake-170'")

    cfg_for_env = Config(
        env_id=args.env_id, seed=args.seed,
        agent="RMA_DATT", scale=False, wind_bool=args.wind_bool
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(cfg_for_env.env_id, cfg=cfg_for_env)

    # Load agent and adaptation networks
    base_path = "/home/shyam/adapt-drones/runs/adapt-ICRA/traj_v3-RMA_DATT/snowy-lake-170"
    model_path = os.path.join(base_path, "best_model.pt")
    adapt_path = os.path.join(base_path, "adapt_network.pt")
    if not os.path.exists(model_path) or not os.path.exists(adapt_path):
        raise FileNotFoundError(f"Could not find model files at: {base_path}")

    agent = RMA_DATT(
        priv_info_shape=env.unwrapped.priv_info_shape,
        state_shape=env.unwrapped.state_obs_shape,
        traj_shape=env.unwrapped.reference_traj_shape,
        action_shape=env.action_space.shape[0],
    ).to(device)
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = env.unwrapped.state_obs_shape + env.action_space.shape[0]
    time_horizon = cfg_for_env.network.adapt_time_horizon
    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg_for_env.network.env_encoder_output

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))
    adapt_net.eval()

    # Lists to store concatenated data from all episodes
    obs_list = []
    action_list = []
    episode_ends = []
    
    # Get dims for pre-allocation check
    action_dim = env.action_space.shape[0]
    # obs_dim = pos(3) + vel(3) + quat(4) + prev_action(action_dim)
    obs_dim = 3 + 3 + 4 + action_dim
    # target_dim = residual_dynamics(3)
    target_dim = 3


    obs_dict, info = env.reset(seed=args.seed)

    state_action_buffer = torch.zeros((env.unwrapped.state_obs_shape + env.action_space.shape[0], time_horizon), device=device)
    prev_action = torch.zeros(env.action_space.shape[0], device=device)

    for step in tqdm(range(args.total_timesteps), desc="Collecting Timesteps"):
        # === Time `t` starts here ===

        # Capture state at time t.
        pos_t = env.unwrapped.position.copy()
        vel_t = env.unwrapped.velocity.copy()
        quat_t = env.unwrapped.quat.copy()
        action_t_minus_1_np = prev_action.cpu().numpy()

        with torch.no_grad():
            # Decide on action(t) based on state(t)
            state_obs = torch.tensor(obs_dict["state"], dtype=torch.float32, device=device)
            traj_obs = torch.tensor(obs_dict["trajectory"], dtype=torch.float32, device=device)
            state_action = torch.cat((state_obs, prev_action), dim=-1) # prev_action is u(t-1)
            state_action_buffer = torch.cat(
                (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
            )
            env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))
            dummy_priv_info = torch.zeros(env.unwrapped.priv_info_shape, device=device)
            full_obs_tensor = torch.cat((dummy_priv_info, state_obs, traj_obs), dim=-1)

            # This is action(t), which will be applied to the environment
            action = agent(full_obs_tensor.unsqueeze(0), predicited_enc=env_encoder)
            action_np = action.squeeze(0).cpu().numpy()

        # Apply action(t). The environment moves from state(t) to state(t+1).
        next_obs_dict, reward, terminated, truncated, info = env.step(action_np)

        # === Calculate the LABEL for time `t` using info from t and t+1 ===
        # Environment is now at time t+1. We can calculate the true dynamics over the [t, t+1] interval.
        vel_t_plus_1 = env.unwrapped.velocity.copy()
        accel_t = (vel_t_plus_1 - vel_t) / env.unwrapped.ctrl_timestep # This is the true a(t)

        total_thrust_magnitude_t = env.unwrapped.last_force_torque_action[0] # Commanded thrust from action(t)
        rot_mat_t = rotation.q_to_rot_mat(quat_t) # Orientation at time t
        world_frame_thrust_t = rot_mat_t @ np.array([0, 0, total_thrust_magnitude_t])

        # This is the residual dynamics at time t, H(t). This is your model's TARGET.
        residual_dynamics_t = world_frame_thrust_t - env.unwrapped.avg_mass * accel_t

        # === Construct the data pair and append to lists ===
        # Observation for the model: [state(t), action(t-1)]
        obs_t = np.concatenate([pos_t, vel_t, quat_t, action_t_minus_1_np])
        
        # Action for the model (the target to predict): H(t)
        action_t = residual_dynamics_t

        # Append to the main lists
        obs_list.append(obs_t)
        action_list.append(action_t)

        # Prepare for the next loop
        obs_dict = next_obs_dict
        prev_action = action.squeeze(0)

        if terminated or truncated:
            # Mark the end of an episode
            episode_ends.append(len(obs_list))

            # Reset environment and buffers
            obs_dict, info = env.reset(seed=args.seed + step + 1)
            state_action_buffer.zero_()
            prev_action.zero_()

            print(f"[data_collector_policy] Env sampled arm length: {env.unwrapped.arm_length:.3f} m, "
                  f"Env sampled mass: {env.unwrapped.model.body_mass[env.unwrapped.drone_id]:.3f} kg, "
                  f"Env average mass: {env.unwrapped.avg_mass:.3f} kg")
    env.close()

    print("\nData collection finished. Converting and saving to Zarr format...")

    # Convert lists to numpy arrays
    obs_array = np.array(obs_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)
    episode_ends_array = np.array(episode_ends, dtype=np.int64)
    
    # Sanity check dimensions
    assert obs_array.shape[1] == obs_dim
    assert action_array.shape[1] == target_dim

    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    # Write data to a Zarr file
    # The keys 'state', 'action', and 'episode_ends' must match what PushTStateDataset expects.
    # Note: 'state' in the zarr file corresponds to 'obs' in the dataset class.
    root = zarr.open(args.output_path, 'w')
    
    # Create groups for data and metadata
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Use compression for efficiency
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    
    # Store the main observation and action data
    data_group.create_dataset('state', data=obs_array, chunks=(1024, -1), compressor=compressor)
    data_group.create_dataset('action', data=action_array, chunks=(1024, -1), compressor=compressor)
    
    # Store the episode boundaries
    meta_group.create_dataset('episode_ends', data=episode_ends_array, compressor=compressor)

    # Store other useful metadata as attributes of the 'meta' group
    meta_group.attrs['nominal_mass'] = env.unwrapped.avg_mass
    meta_group.attrs['arm_length'] = env.unwrapped.arm_length
    
    print(f"Saved Zarr dataset to {args.output_path}")
    print(f"Dataset contains {len(episode_ends)} episodes and {len(obs_array)} total timesteps.")


if __name__ == "__main__":
    collect_data()