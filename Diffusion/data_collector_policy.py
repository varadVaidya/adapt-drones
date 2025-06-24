import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import tyro
from tqdm import tqdm
import random

from adapt_drones.cfgs.config import Config
import adapt_drones.utils.rotation as rotation
from adapt_drones.networks.agents import RMA_DATT
from adapt_drones.networks.adapt_net import AdaptationNetwork

os.environ["MUJOCO_GL"] = "egl"

@dataclass
class Args:
    """Script arguments for data collection."""
    env_id: str = "traj_v3"
    wind_bool: bool = True
    seed: int = -1
    arm_length: float = 0.125
    total_timesteps: int = 5000
    output_path: str = "data/snowy-lake-170_dataset_randomized.npz"
    save_csv: bool = False  

def collect_data():
    args = tyro.cli(Args)

    args.seed = random.randint(0, 2**32 -1) if args.seed < 0 else args.seed
    
    print("Loading config to access scaling laws...")
    cfg_for_laws = Config(
        env_id=args.env_id, seed=args.seed,
        agent="RMA_DATT", scale=True, wind_bool=args.wind_bool
    )
    
    L = args.arm_length
    avg_mass = np.polyval(cfg_for_laws.scale.avg_mass_fit, L)
    std_mass = 3 * np.polyval(cfg_for_laws.scale.std_mass_fit, L)
    std_mass = 0.0 if std_mass < 0.0 else std_mass
    
    nominal_mass = avg_mass
    mass_range = (max(0, avg_mass - std_mass), avg_mass + std_mass)

    print(f"--- Data Collection for DroneDiffusion ---")
    print(f"Using expert policy: 'snowy-lake-170'")
    print(f"Using manually specified arm length: {L:.3f} m")
    print(f"Calculated average mass: {avg_mass:.3f} kg, std: {std_mass:.3f} kg")
    print(f"Sampling random mass per episode from range: ({mass_range[0]:.3f}, {mass_range[1]:.3f}) kg")
    print(f"Using FIXED nominal mass (m_hat) for H calculation: {nominal_mass:.3f} kg")

    cfg_for_env = Config(
        env_id=args.env_id, seed=args.seed,
        agent="RMA_DATT", scale=False, wind_bool=args.wind_bool
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(cfg_for_env.env_id, cfg=cfg_for_env)

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

    # def _set_drone_properties(env, mass, arm_length, inertia):
    #     env.unwrapped.model.body_mass[env.unwrapped.drone_id] = mass
    #     env.unwrapped.arm_length = arm_length
    #     env.unwrapped.model.body_inertia[env.unwrapped.drone_id] = inertia

    # def randomize_drone_properties(env, arm_length, mass_range, cfg_scale):
    #     current_mass = np.random.uniform(mass_range[0], mass_range[1])
    #     ixx = np.polyval(cfg_scale.avg_ixx_fit, arm_length)
    #     iyy = np.polyval(cfg_scale.avg_iyy_fit, arm_length)
    #     izz = np.polyval(cfg_scale.avg_izz_fit, arm_length)
    #     inertia = [ixx, iyy, izz]
    #     _set_drone_properties(env, current_mass, arm_length, inertia)
    #     # Optionally store the current mass on the unwrapped env if you need to access it elsewhere
    #     env.unwrapped.current_mass = current_mass

    all_data = []
    
    obs_dict, info = env.reset(seed=args.seed)
    # randomize_drone_properties(env, L, mass_range, cfg_for_laws.scale)
    
    state_action_buffer = torch.zeros((env.unwrapped.state_obs_shape + env.action_space.shape[0], time_horizon), device=device)
    prev_action = torch.zeros(env.action_space.shape[0], device=device)

    for step in tqdm(range(args.total_timesteps), desc="Collecting Timesteps"):
        # Capture state at time t, BEFORE stepping the environment
        pos_t = env.unwrapped.position.copy()
        vel_t = env.unwrapped.velocity.copy()
        quat_t = env.unwrapped.quat.copy()

        with torch.no_grad():
            state_obs = torch.tensor(obs_dict["state"], dtype=torch.float32, device=device)
            traj_obs = torch.tensor(obs_dict["trajectory"], dtype=torch.float32, device=device)
            state_action = torch.cat((state_obs, prev_action), dim=-1)
            state_action_buffer = torch.cat(
                (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
            )
            env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))
            dummy_priv_info = torch.zeros(env.unwrapped.priv_info_shape, device=device)
            full_obs_tensor = torch.cat((dummy_priv_info, state_obs, traj_obs), dim=-1)
            action = agent(full_obs_tensor.unsqueeze(0), predicited_enc=env_encoder)
            action_np = action.squeeze(0).cpu().numpy()
            
        # Step the environment to get state at t+1
        next_obs_dict, reward, terminated, truncated, info = env.step(action_np)
        
        # Calculate dynamics based on the t -> t+1 transition 
        vel_t_plus_1 = env.unwrapped.velocity.copy()
        accel_t = (vel_t_plus_1 - vel_t) / env.unwrapped.dt
        
        # Use orientation from time t for correct thrust calculation
        rot_mat_t = rotation.q_to_rot_mat(quat_t)
        total_thrust_magnitude = np.sum(env.unwrapped.data.actuator_force)
        print(f"Total thrust magnitude at step {step}: {total_thrust_magnitude:.3f} N")
        world_frame_thrust_t = rot_mat_t @ np.array([0, 0, total_thrust_magnitude])
        
        residual_dynamics_t = world_frame_thrust_t - nominal_mass * accel_t
        
        # Log data from time t
        all_data.append({
            'position': pos_t, 'velocity': vel_t, 'acceleration': accel_t,
            'world_frame_thrust': world_frame_thrust_t, 'residual_dynamics': residual_dynamics_t
        })
        
        obs_dict = next_obs_dict
        prev_action = action.squeeze(0)

        if terminated or truncated:
            # Insert NaN separator row
            all_data.append({
                'position': [np.nan, np.nan, np.nan],
                'velocity': [np.nan, np.nan, np.nan],
                'acceleration': [np.nan, np.nan, np.nan],
                'world_frame_thrust': [np.nan, np.nan, np.nan],
                'residual_dynamics': [np.nan, np.nan, np.nan]
            })

            obs_dict, info = env.reset(seed=args.seed + step + 1)
            # randomize_drone_properties(env, L, mass_range, cfg_for_laws.scale)
            
            state_action_buffer.zero_()
            prev_action.zero_()

    env.close()
    
    print("\nData collection finished. Converting and saving...")
    
    dataset_dict = {key: np.array([d[key] for d in all_data], dtype=np.float32) for key in all_data[0]}

    if args.save_csv:
        flat_data = {}
        column_suffixes = {3: ['_x', '_y', '_z']}
        for key, data_array in dataset_dict.items():
            if data_array.ndim == 2 and data_array.shape[1] in column_suffixes:
                suffixes = column_suffixes[data_array.shape[1]]
                for i, suffix in enumerate(suffixes):
                    flat_data[f"{key}{suffix}"] = data_array[:, i]
            else:
                flat_data[key] = data_array
        
        df = pd.DataFrame(flat_data)
        csv_path = args.output_path.replace('.npz', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

    dataset_dict['nominal_mass'] = nominal_mass
    dataset_dict['arm_length'] = args.arm_length
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
        
    np.savez(args.output_path, **dataset_dict)
    print(f"Saved NPZ to {args.output_path}")

if __name__ == "__main__":
    collect_data()
