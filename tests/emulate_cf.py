import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Union
import pkg_resources

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt


from adapt_drones.cfgs.config import *
from adapt_drones.networks.agents import *
from adapt_drones.networks.adapt_net import AdaptationNetwork
from adapt_drones.utils.emulate_new import *
from adapt_drones.utils.mpc_utils import quaternion_to_euler
from adapt_drones.utils.dynamics import CustomDynamics
from adapt_drones.utils.ploting import data_plot, TextonPlot


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "desert-star-119"
    seed: int = 15092024
    agent: str = "RMA_DATT"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = False


def compute_obs(cf_quad: Quadrotor):
    # calculate the state observation
    trajectory_window = cf_quad.get_trajectory_window()
    # print(trajectory_window.shape)
    assert trajectory_window.shape == (cf_quad.trajectory_window_length, 6)

    window_position = trajectory_window[:, 0:3] - cf_quad.state.pos
    window_velocity = trajectory_window[:, 3:6] - cf_quad.state.vel

    target_pos = trajectory_window[0, 0:3]
    target_vel = trajectory_window[0, 3:6]

    delta_pos = target_pos - cf_quad.state.pos
    delta_vel = target_vel - cf_quad.state.vel

    delta_ori = sub_quat(cf_quad.state.quat, np.array([1, 0, 0, 0]))
    delta_angular_vel = -cf_quad.state.omega

    state_obs = np.concatenate((delta_pos, delta_ori, delta_vel, delta_angular_vel))

    window_trajectory = np.hstack([window_position, window_velocity]).flatten()

    priv_info = np.concatenate(
        (
            [cf_quad.mass],  # mass
            cf_quad.J,  # inertia
            [2.75],  # thrust to weight ratio
            [cf_quad.prop_const],  # propeller constant
            [cf_quad.arm_length],  # arm length
            np.array([0, 0, 0]),  # wind
        )
    )

    return np.concatenate((priv_info, state_obs, window_trajectory)).astype(np.float32)


def reset(cf_state: State, cf_quad: Quadrotor):
    pass


def eval_trajectory(
    path: str,
    idx: Union[int, None],
    emulate_freq: int,
    trajectory_window_length: int,
    trajectory: Union[np.ndarray, None] = None,
):
    if trajectory is None:
        eval_trajs = np.load(path)

        eval_traj = eval_trajs[idx]
        rows_not_nan = sum(~np.isnan(eval_traj[:, 1]))

        eval_traj = eval_traj[:rows_not_nan]

        ref_position = eval_traj[:, 1:4]
        ref_velocity = eval_traj[:, 8:11]

        duration = eval_traj.shape[0] - (trajectory_window_length + 1)
        t = np.linspace(0, duration / emulate_freq, duration)
        # ref_position = eval_traj[:duration, 1:4]
        # ref_velocity = eval_traj[:duration, 8:11]

        return t, np.concatenate((ref_position, ref_velocity), axis=1)

    else:
        raise NotImplementedError


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

    cf_state = State()
    cf_quad = Quadrotor(state=cf_state)

    best_model = True
    idx = 4

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)

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

    # setup helping variables.
    # hardcode the values for now. ! TODO: maybe change before final submission
    priv_info_shape = 10
    state_shape = 12
    traj_shape = 600
    action_shape = 4
    cf_quad.trajectory_window_length = cfg.environment.trajectory_window
    trajectory_path = pkg_resources.resource_filename(
        "adapt_drones", "assets/slow_pi_tcn_eval_mpc.npy"
    )
    t, ref_trajectory = eval_trajectory(
        trajectory_path,
        idx=idx,
        emulate_freq=100,
        trajectory_window_length=cf_quad.trajectory_window_length,
    )
    cf_quad.reference_trajectory = ref_trajectory

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
    time_horizon = 50
    adapt_input = time_horizon * state_action_shape
    adapt_output = 8

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    # compute_obs(cf_quad)
    # cf_quad.step(np.zeros(4), 0.01)
    # print(cf_quad.state)

    cf_quad.state.pos = ref_trajectory[0, :3]
    cf_quad.state.vel = ref_trajectory[0, 3:6]

    print("Length of t:", len(t))

    cf_position = []
    cf_velocity = []
    cf_quaternion = []
    cf_ang_vel = []
    cf_action = []

    action = np.zeros(4)

    for i in range(len(t)):
        obs = compute_obs(cf_quad)
        state_action = np.concatenate(
            (obs[priv_info_shape : priv_info_shape + state_shape], action)
        )

        state_action = (
            torch.tensor(state_action, dtype=torch.float32).unsqueeze(-1).to(device)
        )

        state_action_buffer = torch.cat(
            (state_action, state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
            predicited_enc=env_encoder,
        )
        action = action.cpu().detach().numpy()[0]
        cf_action.append(action)
        cf_quad.step(action, 0.01)
        # print(cf_quad.state.pos)
        cf_position.append(cf_quad.state.pos.copy())
        cf_velocity.append(cf_quad.state.vel.copy())
        cf_quaternion.append(cf_quad.state.quat.copy())
        cf_ang_vel.append(cf_quad.state.omega.copy())
        # print(cf_quad.state)
        # print(cf_quad.step_counter)

    cf_position = np.array(cf_position)
    cf_velocity = np.array(cf_velocity)
    cf_quaternion = np.array(cf_quaternion)
    cf_ang_vel = np.array(cf_ang_vel)
    cf_action = np.array(cf_action)
    
    for i in range(len(cf_action)):
        cf_action[i] = cf_quad.pre_process_action(cf_action[i])

    ref_position = ref_trajectory[: len(t), :3]
    ref_velocity = ref_trajectory[: len(t), 3:6]

    plot_text = TextonPlot(
        seed=args.seed,
        mass=cf_quad.mass,
        inertia=cf_quad.J,
        wind=np.array([0, 0, 0]),
        com=np.array([0, 0, 0]),
        prop_const=cf_quad.prop_const,
        arm_length=cf_quad.arm_length,
        thrust2weight="2.75",
        mean_error="",
        rms_error="",
    )

    data_plot(
        t,
        cf_position,
        ref_position,
        cf_velocity,
        ref_velocity,
        cf_quaternion,
        cf_ang_vel,
        cf_action,
        plot_text,
        "cf_emulate",
        "new_rk",
    )

    # cf_rpy = [rpy for rpy in map(quaternion_to_euler, cf_quaternion)]
    # cf_rpy = np.array(cf_rpy)

    # pos_error = ref_trajectory[: len(t), :3] - cf_position
    # vel_error = ref_trajectory[: len(t), 3:6] - cf_velocity

    # mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    # print("Mean Position Error:", mean_error)
    # mean_vel_error = np.mean(np.linalg.norm(vel_error, axis=1))
    # print("Mean Velocity Error:", mean_vel_error)

    # print("Max Roll Angle:", np.rad2deg(np.max(cf_rpy[:, 0])))
    # print("Max Pitch Angle:", np.rad2deg(np.max(cf_rpy[:, 1])))

    # plt.figure()
    # plt.plot(t, cf_position[:, 0], "r", label="x")
    # plt.plot(t, cf_position[:, 1], "g", label="y")
    # plt.plot(t, cf_position[:, 2], "b", label="z")

    # plt.plot(t, ref_trajectory[: len(t), 0], "r--", alpha=0.75, label="x_ref")
    # plt.plot(t, ref_trajectory[: len(t), 1], "g--", alpha=0.75, label="y_ref")
    # plt.plot(t, ref_trajectory[: len(t), 2], "b--", alpha=0.75, label="z_ref")

    # plt.grid()
    # plt.title("Position")
    # plt.legend()
    # plt.show()

    # question: what all functions do we need to implement?
    # 1. compute_obs
    # 2. compute_action or some form of post processing of the normal action
    # 3. trajectory generation, or loading of some sorts.
    # 4. reset?? not sure if we need this.
