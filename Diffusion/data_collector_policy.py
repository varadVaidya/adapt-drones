import os
import time
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import tyro
from tqdm import tqdm

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
    output_path: str = "data/slow_tcn_eval_diff_dataset.npz"
    save_csv: bool = False

def collect_data():
    """
    Main function to collect drone dynamics data using an expert policy.
    Aligns data to predict H(t) from state(t) and action(t-1).
    """
    args = tyro.cli(Args)

    # If seed is negative, generate a random one
    args.seed = random.randint(0, 2**32 - 1) if args.seed < 0 else args.seed

    print("Loading config to access scaling laws...")
    cfg_for_laws = Config(
        env_id=args.env_id, seed=args.seed,
        agent="RMA_DATT", scale=True, wind_bool=args.wind_bool
    )

    # # Define arm length mean and std, and sample per episode
    # arm_length_mean = args.arm_length  # Get mean arm length from args
    # arm_length_std = (0.15 - 0.10) / 6  # 3-sigma covers [0.10, 0.15]
        # L = np.random.normal(arm_length_mean, arm_length_std)
        # L = np.clip(L, 0.10, 0.15)  # Ensure within bounds

    # avg_mass = np.polyval(cfg_for_laws.scale.avg_mass_fit, L)
    # std_mass = 3 * np.polyval(cfg_for_laws.scale.std_mass_fit, L)
    # std_mass = 0.0 if std_mass < 0.0 else std_mass

    # nominal_mass = avg_mass
    # mass_range = (max(0, avg_mass - std_mass), avg_mass + std_mass)

    print("--- Data Collection for DroneDiffusion ---")
    print(f"Aligning data to predict H(t) from state(t) and action(t-1)")
    print(f"Using expert policy: 'snowy-lake-170'")
    # print(f"Sampled arm length: {:.3f} m (mean: {arm_length_mean:.3f}, std: {arm_length_std:.3f})")
    # print(f"Calculated average mass: {avg_mass:.3f} kg, std: {std_mass:.3f} kg")
    # print(f"Sampling random mass per episode from range: ({mass_range[0]:.3f}, {mass_range[1]:.3f}) kg")
    # print(f"Using FIXED nominal mass (m_hat) for H calculation: {nominal_mass:.3f} kg")

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

    all_data = []

    obs_dict, info = env.reset(seed=args.seed)

    state_action_buffer = torch.zeros((env.unwrapped.state_obs_shape + env.action_space.shape[0], time_horizon), device=device)
    # prev_action will store action u(t-1)
    prev_action = torch.zeros(env.action_space.shape[0], device=device)

    for step in tqdm(range(args.total_timesteps), desc="Collecting Timesteps"):
        # === Time `t` starts here ===

        # Capture state at time t. These are your model's INPUTS.
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

        # === Log the fully aligned data row ===
        # We have state(t), action(t-1), and H(t) all available now.
        all_data.append({
            # State at time t (inputs for your model)
            'position': pos_t,
            'velocity': vel_t,
            'orientation': quat_t,
            # Action from time t-1 (input for your model)
            'action_t_minus_1': action_t_minus_1_np,
            # Residual dynamics at time t (target/label for your model)
            'residual_dynamics': residual_dynamics_t,
            # Also save these for analysis/debugging
            'acceleration': accel_t,
            'world_frame_thrust': world_frame_thrust_t,
        })

        # Prepare for the next loop
        obs_dict = next_obs_dict
        # Update prev_action to be action(t) for the next iteration (where it will be t-1)
        prev_action = action.squeeze(0)

        if terminated or truncated:
            # Insert a NaN separator row to distinguish between trajectories
            all_data.append({key: np.full_like(val, np.nan) for key, val in all_data[-1].items()})
            obs_dict, info = env.reset(seed=args.seed + step + 1)
            state_action_buffer.zero_()
            prev_action.zero_()

            # After env.reset(), print the sampled arm length and mass from the environment
            print(f"[data_collector_policy] Env sampled arm length: {env.unwrapped.arm_length:.3f} m, "
                  f"Env sampled mass: {env.unwrapped.model.body_mass[env.unwrapped.drone_id]:.3f} kg, "
                  f"Env average mass: {env.unwrapped.avg_mass:.3f} kg")
    env.close()

    print("\nData collection finished. Converting and saving...")
    
    # Rebuild the dictionary keys from the first valid entry
    dataset_dict = {key: np.array([d[key] for d in all_data], dtype=np.float32) for key in all_data[0]}

    if args.save_csv:
        flat_data = {}
        action_dim = env.action_space.shape[0]
        # Add orientation (4) and action (N) to suffix mapping
        column_suffixes = {
            3: ['_x', '_y', '_z'],
            4: ['_w', '_x', '_y', '_z'],
            action_dim: [f'_{i}' for i in range(action_dim)]
        }
        for key, data_array in dataset_dict.items():
            shape_val = data_array.shape[1] if data_array.ndim == 2 else -1
            if shape_val in column_suffixes:
                suffixes = column_suffixes[shape_val]
                # Special naming for action to make CSV header cleaner
                col_name = 'action' if 'action' in key else key
                for i, suffix in enumerate(suffixes):
                    flat_data[f"{col_name}{suffix}"] = data_array[:, i]
            else:
                flat_data[key] = data_array
        
        df = pd.DataFrame(flat_data)
        csv_path = args.output_path.replace('.npz', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

    dataset_dict['nominal_mass'] = env.unwrapped.avg_mass
    dataset_dict['arm_length'] = env.unwrapped.arm_length  # Save the actual sampled arm length

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    np.savez(args.output_path, **dataset_dict)
    print(f"Saved NPZ to {args.output_path}")

if __name__ == "__main__":
    collect_data()