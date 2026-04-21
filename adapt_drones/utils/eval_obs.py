import os
import random
from dataclasses import asdict
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym

from adapt_drones.cfgs.config import *
from adapt_drones.networks.agents import *
from adapt_drones.utils.ploting import data_plot, TextonPlot
from adapt_drones.networks.adapt_net import AdaptationNetwork


def phase1_obs_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
):
    print("=================================")
    print("Phase 1 Obstacle Evaluation")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(datadump_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    print("Model Path:", model_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg, record=True)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = RMA_DATT_Safe(
        priv_info_shape=env.unwrapped.priv_info_shape,
        state_shape=env.unwrapped.state_obs_shape,
        traj_shape=env.unwrapped.reference_traj_shape,
        action_shape=env.action_space.shape,
    ).to(device)
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    obs, _ = env.reset(seed=cfg.seed, options=options)

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]
    prop_const = env.unwrapped.prop_const
    arm_length = env.unwrapped.arm_length
    thrust2weight = env.unwrapped.thrust2weight

    text_plot = TextonPlot(
        seed=f"Seed: {cfg.seed}",
        mass=f"Mass: {mass:.3f}",
        inertia=f"Inertia: {inertia}",
        wind=f"Wind: {wind}",
        com=f"Com:{com}",
        prop_const=f"Prop Constant:{prop_const}",
        arm_length=f"Arm Length:{arm_length}",
        thrust2weight=f"TWR:{thrust2weight}",
        mean_error="",
        rms_error="",
    )

    print("\n".join("{}".format(v) for k, v in asdict(text_plot).items()))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)
    print("Trajectory Length:", len(t))

    position, quaternion, lin_velocity, ang_velocity = [], [], [], []
    action_numpy = []
    collision_count = 0
    min_obstacle_clearance = np.inf

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]
        action = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )[0]
        obs, _, _, _, info = env.step(action.cpu().numpy()[0])

        if info.get("obstacle_collision", False):
            collision_count += 1
        min_dist = info.get("min_obstacle_distance", np.inf)
        min_obstacle_clearance = min(min_obstacle_clearance, min_dist)

        position.append(env.unwrapped.position)
        quaternion.append(env.unwrapped.quat)
        lin_velocity.append(env.unwrapped.velocity)
        ang_velocity.append(env.unwrapped.angular_velocity)
        action_numpy.append(action.cpu().numpy()[0])

    position = np.array(position)
    quaternion = np.array(quaternion)
    lin_velocity = np.array(lin_velocity)
    ang_velocity = np.array(ang_velocity)
    action_numpy = np.array(action_numpy)

    for i in range(len(action_numpy)):
        action_numpy[i] = env.unwrapped.preprocess_action(action_numpy[i])

    pos_error = ref_positon[: len(position)] - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    print(f"\n--- Safety Metrics (Phase 1) ---")
    print(f"  Collision steps: {collision_count}/{len(t)}")
    print(f"  Min obstacle clearance: {min_obstacle_clearance:.4f}m")
    print(f"  Mean position error: {mean_error:.4f}m")
    print(f"  RMS position error: {rms_error:.4f}m")

    datadump = np.hstack(
        (t[: len(position)].reshape(-1, 1), position, lin_velocity, ref_positon[: len(position)], ref_velocity[: len(position)])
    )
    headers = ["p", "v", "pd", "vd"]
    axes = ["x", "y", "z"]
    headers = [f"{a}_{b}" for a in headers for b in axes]
    headers = ["t"] + headers
    np.savetxt(
        datadump_folder + f"phase1_obs-{cfg.seed}.csv",
        datadump,
        delimiter=",",
        header=",".join(headers),
    )

    data_plot(
        t[: len(position)],
        position=position,
        goal_position=ref_positon[: len(position)],
        velocity=lin_velocity,
        goal_velocity=ref_velocity[: len(position)],
        quaternion=quaternion,
        angular_velocity=ang_velocity,
        action=action_numpy,
        plot_text=text_plot,
        save_prefix="phase_1_obs",
        save_path=results_folder,
    )
    env.unwrapped.vidwrite(results_folder + "phase_1_obs.mp4")
    env.unwrapped.renderer.close()


def RMA_DATT_obs_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
):
    print("=================================")
    print("Adaptation Obstacle Evaluation")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(datadump_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

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
    state_shape = env.unwrapped.state_obs_shape
    traj_shape = env.unwrapped.reference_traj_shape
    action_shape = env.action_space.shape[0]

    agent = RMA_DATT_Safe(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    # Adaptation network uses core 12D state only
    core_state_shape = 12
    state_action_shape = core_state_shape + action_shape
    time_horizon = cfg.network.adapt_time_horizon

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    obs, _ = env.reset(seed=cfg.seed, options=options)

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]
    prop_const = env.unwrapped.prop_const
    arm_length = env.unwrapped.arm_length
    thrust2weight = env.unwrapped.thrust2weight

    text_plot = TextonPlot(
        seed=f"Seed: {cfg.seed}",
        mass=f"Mass: {mass:.3f}",
        inertia=f"Inertia: {inertia}",
        wind=f"Wind: {wind}",
        com=f"Com:{com}",
        prop_const=f"Prop Constant:{prop_const}",
        arm_length=f"Arm Length:{arm_length}",
        thrust2weight=f"TWR:{thrust2weight}",
        mean_error="",
        rms_error="",
    )

    print("\n".join("{}".format(v) for k, v in asdict(text_plot).items()))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    position, quaternion, lin_velocity, ang_velocity = [], [], [], []
    action_numpy = []
    collision_count = 0
    min_obstacle_clearance = np.inf

    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]

        # Only use core 12D state for adaptation buffer
        core_state_obs = state_obs[:core_state_shape]
        state_action = torch.cat((core_state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent.get_action_and_value(
            obs.unsqueeze(0), predicited_enc=env_encoder
        )[0]

        obs_np, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs_np, dtype=torch.float32).to(device)

        if info.get("obstacle_collision", False):
            collision_count += 1
        min_dist = info.get("min_obstacle_distance", np.inf)
        min_obstacle_clearance = min(min_obstacle_clearance, min_dist)

        position.append(env.unwrapped.position)
        quaternion.append(env.unwrapped.quat)
        lin_velocity.append(env.unwrapped.velocity)
        ang_velocity.append(env.unwrapped.angular_velocity)
        action_numpy.append(action.cpu().numpy()[0])

    position = np.array(position)
    quaternion = np.array(quaternion)
    lin_velocity = np.array(lin_velocity)
    ang_velocity = np.array(ang_velocity)
    action_numpy = np.array(action_numpy)

    for i in range(len(action_numpy)):
        action_numpy[i] = env.unwrapped.preprocess_action(action_numpy[i])

    pos_error = ref_positon[: len(position)] - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    print(f"\n--- Safety Metrics (Adaptation) ---")
    print(f"  Collision steps: {collision_count}/{len(t)}")
    print(f"  Min obstacle clearance: {min_obstacle_clearance:.4f}m")
    print(f"  Mean position error: {mean_error:.4f}m")
    print(f"  RMS position error: {rms_error:.4f}m")

    datadump = np.hstack(
        (t[: len(position)].reshape(-1, 1), position, lin_velocity, ref_positon[: len(position)], ref_velocity[: len(position)])
    )
    headers = ["p", "v", "pd", "vd"]
    axes = ["x", "y", "z"]
    headers = [f"{a}_{b}" for a in headers for b in axes]
    headers = ["t"] + headers
    np.savetxt(
        datadump_folder + f"adapt_obs-{cfg.seed}.csv",
        datadump,
        delimiter=",",
        header=",".join(headers),
    )

    data_plot(
        t[: len(position)],
        position=position,
        goal_position=ref_positon[: len(position)],
        velocity=lin_velocity,
        goal_velocity=ref_velocity[: len(position)],
        quaternion=quaternion,
        angular_velocity=ang_velocity,
        action=action_numpy,
        plot_text=text_plot,
        save_prefix="adapt_obs",
        save_path=results_folder,
    )

    env.unwrapped.vidwrite(results_folder + "adapt_obs.mp4")
    env.unwrapped.renderer.close()
