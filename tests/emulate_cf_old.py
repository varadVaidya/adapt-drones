import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym


from adapt_drones.cfgs.config import *
from adapt_drones.networks.agents import *
from adapt_drones.networks.adapt_net import AdaptationNetwork
from adapt_drones.utils.emulate import *
from adapt_drones.utils.dynamics import CustomDynamics


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "earthy-snowball-77"
    seed: int = 15092024
    agent: str = "RMA_DATT"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = False


if __name__ == "__main__":

    args = Args()
    cfg = Config(
        env_id=args.env_id,
        seed=args.seed,
        eval=True,
        run_name=args.run_name,
        agent=args.agent,
        scale=args.scale,
        wind_bool=args.wind_bool,
    )

    print("=================================")
    print("CF Emulation")

    # set the dynamics of crazyflie to both the emulation dynamics
    # and env dynamics

    dynamics = CustomDynamics(
        mass=0.792,
        arm_length=0.16,
        ixx=0.0047,
        iyy=0.005,
        izz=0.0074,
        km_kf=0.006,
    )
    cf_state = State()
    cf_quad = Quadrotor(state=cf_state)

    best_model = True
    idx = 0
    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

    # load the model
    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    print("Adapt Path:", adapt_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg, record=True)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    priv_info_shape = env.unwrapped.priv_info_shape
    print("Private Info Shape:", priv_info_shape)
    state_shape = env.unwrapped.state_obs_shape
    print("State Shape:", state_shape)
    traj_shape = env.unwrapped.reference_traj_shape
    print("Trajectory Shape:", traj_shape)
    action_shape = env.action_space.shape[0]
    print("Action Shape:", action_shape)

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
    print("State Action Shape:", state_action_shape)
    time_horizon = cfg.network.adapt_time_horizon
    print("Time Horizon:", time_horizon)

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output
    print("Adapt Input:", adapt_input)
    print("Adapt Output:", adapt_output)

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    obs, _ = env.reset(seed=cfg.seed, options=asdict(dynamics))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    cf_state.pos = env.unwrapped.position
    cf_state.vel = env.unwrapped.velocity
    cf_state.quat = env.unwrapped.quat
    cf_state.omega = env.unwrapped.angular_velocity

    position, velocity = [], []
    cf_position, cf_velocity = [], []

    for i in range(len(t)):
        # print(cf_state)
        # print(
        #     env.unwrapped.position,
        #     env.unwrapped.velocity,
        #     env.unwrapped.quat,
        #     env.unwrapped.angular_velocity,
        # )
        print("Time:", i / 100)
        print("Position:", env.unwrapped.position, cf_state.pos)
        print("Velocity:", env.unwrapped.velocity, cf_state.vel)
        print("Quat:", env.unwrapped.quat, cf_state.quat)
        print("Angular Velocity:", env.unwrapped.angular_velocity, cf_state.omega)
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]
        action = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )[0]
        obs = env.step(action.cpu().numpy()[0])[0]
        # print("NN Action:", env.unwrapped.last_force_torque_action)
        cf_quad.step(
            env.unwrapped.last_force_torque_action,
            env.unwrapped.mj_timestep,
        )

        position.append(env.unwrapped.position)
        velocity.append(env.unwrapped.velocity)
        cf_position.append(cf_quad.state.pos)
        cf_velocity.append(cf_quad.state.vel)
        input()

    position = np.array(position)
    velocity = np.array(velocity)
    cf_position = np.array(cf_position)
    cf_velocity = np.array(cf_velocity)

    assert len(t) == len(position), "Length of t and position are not same"
    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))
    print("Mean Error NN:", mean_error)
    print("RMS Error NN:", rms_error)

    pos_error = ref_positon - cf_position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))
    print("Mean Error CF:", mean_error)
    print("RMS Error CF:", rms_error)
