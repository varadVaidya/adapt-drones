import os
import random
from dataclasses import asdict
from typing import Union
import time

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from adapt_drones.cfgs.config import *
from adapt_drones.networks.agents import *
from adapt_drones.utils.ploting import data_plot, TextonPlot
from adapt_drones.networks.adapt_net import AdaptationNetwork


def phase1_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
):
    """Phase 1 evaluation script

    Args:
        cfg (Config): main config file
        idx: int: The index of the trajectory to be evaluated. If None, a trajectory is
        sampled randomly from the evaluation dataset.
        best_model (bool, optional): Use the best saved mode. Defaults to True.
        Uses the final model if False.
    """
    print("=================================")
    print("Phase 1 Evaluation")

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
    results_folder = run_folder + "results/"
    archive_folder = run_folder + "archive/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(datadump_folder, exist_ok=True)
    os.makedirs(archive_folder, exist_ok=True)
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

    if cfg.agent == "AC":
        agent = SimpleActorCritic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
        ).to(device)
        agent.load_state_dict(torch.load(model_path, weights_only=True))

    elif cfg.agent == "RMA_DATT":
        agent = RMA_DATT(
            priv_info_shape=env.unwrapped.priv_info_shape,
            state_shape=env.unwrapped.state_obs_shape,
            traj_shape=env.unwrapped.reference_traj_shape,
            action_shape=env.action_space.shape,
        ).to(device)
        agent.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        raise ValueError("Invalid agent type")

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

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]
        action = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )[0]
        obs = env.step(action.cpu().numpy()[0])[0]

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

    datadump = np.hstack(
        (t.reshape(-1, 1), position, lin_velocity, ref_positon, ref_velocity)
    )
    headers = ["p", "v", "pd", "vd"]
    axes = ["x", "y", "z"]
    headers = [f"{a}_{b}" for a in headers for b in axes]
    headers = ["t"] + headers
    np.savetxt(
        datadump_folder + f"phase1-{cfg.seed}.csv",
        datadump,
        delimiter=",",
        header=",".join(headers),
    )

    data_plot(
        t,
        position=position,
        goal_position=ref_positon,
        velocity=lin_velocity,
        goal_velocity=ref_velocity,
        quaternion=quaternion,
        angular_velocity=ang_velocity,
        action=action_numpy,
        plot_text=text_plot,
        save_prefix="phase_1",
        save_path=results_folder,
    )
    env.unwrapped.vidwrite(results_folder + "phase_1.mp4")
    env.unwrapped.renderer.close()


def RMA_DATT_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
):
    print("=================================")
    print("Adaptation Evaluation")

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
    results_folder = run_folder + "results/"
    archive_folder = run_folder + "archive/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(datadump_folder, exist_ok=True)
    os.makedirs(archive_folder, exist_ok=True)
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

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
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
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]

        state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent.get_action_and_value(
            obs.unsqueeze(0), predicited_enc=env_encoder
        )[0]

        obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

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

    datadump = np.hstack(
        (t.reshape(-1, 1), position, lin_velocity, ref_positon, ref_velocity)
    )
    headers = ["p", "v", "pd", "vd"]
    axes = ["x", "y", "z"]
    headers = [f"{a}_{b}" for a in headers for b in axes]
    headers = ["t"] + headers
    np.savetxt(
        datadump_folder + f"adapt-{cfg.seed}.csv",
        datadump,
        delimiter=",",
        header=",".join(headers),
    )
    data_plot(
        t,
        position=position,
        goal_position=ref_positon,
        velocity=lin_velocity,
        goal_velocity=ref_velocity,
        quaternion=quaternion,
        angular_velocity=ang_velocity,
        action=action_numpy,
        plot_text=text_plot,
        save_prefix="adapt",
        save_path=results_folder,
    )

    env.unwrapped.vidwrite(results_folder + "adapt.mp4")
    env.unwrapped.renderer.close()


def paper_phase_1_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):
    # print("=================================")
    # print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results-icra/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(results_folder, exist_ok=True)

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    # print("Model Path:", model_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg)
    env = gym.wrappers.FlattenObservation(env)

    if cfg.agent == "AC":
        agent = SimpleActorCritic(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
        ).to(device)
        agent.load_state_dict(torch.load(model_path, weights_only=True))

    elif cfg.agent == "RMA_DATT":
        agent = RMA_DATT(
            priv_info_shape=env.unwrapped.priv_info_shape,
            state_shape=env.unwrapped.state_obs_shape,
            traj_shape=env.unwrapped.reference_traj_shape,
            action_shape=env.action_space.shape,
        ).to(device)
        agent.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        raise ValueError("Invalid agent type")

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    # print("Model Path:", model_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

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

    # print("\n".join("{}".format(v) for k, v in asdict(text_plot).items()))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    position, velocity = [], []

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]
        action = agent.get_action_and_value(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )[0]
        obs = env.step(action.cpu().numpy()[0])[0]

        position.append(env.unwrapped.position)
        velocity.append(env.unwrapped.velocity)

    position = np.array(position)
    velocity = np.array(velocity)

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    if not return_traj_len:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2]
    else:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2], len(t)


def paper_RMA_DATT_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):
    # print("=================================")
    # print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

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
    # print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    # print("Adapt Path:", adapt_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    del cfg.environment.trajectory_dataset

    priv_info_shape = env.unwrapped.priv_info_shape
    state_shape = env.unwrapped.state_obs_shape
    traj_shape = env.unwrapped.reference_traj_shape
    action_shape = env.action_space.shape[0]

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
    time_horizon = cfg.network.adapt_time_horizon

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    obs, _ = env.reset(seed=cfg.seed, options=options)
    # del cfg.environment.trajectory_dataset

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    position, velocity = [], []
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]

        state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent(obs.unsqueeze(0), predicited_enc=env_encoder)

        obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        position.append(env.unwrapped.position)
        velocity.append(env.unwrapped.velocity)

    position = np.array(position)
    velocity = np.array(velocity)

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    if not return_traj_len:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2]
    else:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2], len(t)


def timed_RMA_DATT_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):
    # print("=================================")
    # print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

    torch.set_float32_matmul_precision("high")

    run_folder = (
        "runs/"
        + cfg.experiment.wandb_project_name
        + "/"
        + cfg.grp_name
        + "/"
        + cfg.run_name
        + "/"
    )
    results_folder = run_folder + "results-icra/"
    datadump_folder = results_folder + "datadump/"

    os.makedirs(results_folder, exist_ok=True)

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    # print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    # print("Adapt Path:", adapt_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    priv_info_shape = env.unwrapped.priv_info_shape
    state_shape = env.unwrapped.state_obs_shape
    traj_shape = env.unwrapped.reference_traj_shape
    action_shape = env.action_space.shape[0]

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
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

    # print("\n".join("{}".format(v) for k, v in asdict(text_plot).items()))

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    position, velocity = [], []
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    mean_compute_time = 0.0
    start_time = time.time()
    with torch.inference_mode():
        for i in range(len(t)):
            env.unwrapped.target_position = ref_positon[i]
            env.unwrapped.target_velocity = ref_velocity[i]

            env_obs = obs[: env.unwrapped.priv_info_shape]
            state_obs = obs[
                env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
                + env.unwrapped.state_obs_shape
            ]
            traj_obs = obs[
                env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :
            ]

            state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
            state_action_buffer = torch.cat(
                (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()),
                dim=-1,
            )

            env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

            start_time = time.time()
            action = agent(obs.unsqueeze(0), predicited_enc=env_encoder)

            mean_compute_time += time.time() - start_time

            obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
            obs = torch.tensor(obs, dtype=torch.float32).to(device)

            position.append(env.unwrapped.position)
            velocity.append(env.unwrapped.velocity)

    position = np.array(position)
    velocity = np.array(velocity)

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    mean_compute_time = mean_compute_time / len(t) * 1000

    return mean_error, rms_error, mean_compute_time

    # if not return_traj_len:
    #     return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2],
    # else:
    #     return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2], len(t)


def paper_custom_traj_eval(
    cfg: Config,
    reference_traj: np.ndarray,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):

    # print("=================================")
    # print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

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
    # print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    # print("Adapt Path:", adapt_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    priv_info_shape = env.unwrapped.priv_info_shape
    state_shape = env.unwrapped.state_obs_shape
    traj_shape = env.unwrapped.reference_traj_shape
    action_shape = env.action_space.shape[0]

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
    time_horizon = cfg.network.adapt_time_horizon

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    obs, _ = env.reset(seed=cfg.seed, options=options)
    env.unwrapped.thrust2weight = (
        2.5  # set the thrust to weight ratio to 2.75 for max speed evaluation
    )

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]
    prop_const = env.unwrapped.prop_const
    arm_length = env.unwrapped.arm_length
    thrust2weight = env.unwrapped.thrust2weight

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(
        trajectory=reference_traj
    )

    position, quaternion, lin_velocity, ang_velocity = [], [], [], []
    action_numpy = []
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]

        state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent(obs.unsqueeze(0), predicited_enc=env_encoder)

        obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

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

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    if not return_traj_len:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2]
    else:
        return mean_error, rms_error, mass, inertia[0], inertia[1], inertia[2], len(t)


def custom_traj_eval(
    cfg: Config,
    reference_traj: np.ndarray,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):

    print("=================================")
    print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

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

    model_path = (
        run_folder + "best_model.pt" if best_model else run_folder + "final_model.pt"
    )
    # print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    # print("Adapt Path:", adapt_path)

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

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
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

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(
        trajectory=reference_traj
    )

    position, quaternion, lin_velocity, ang_velocity = [], [], [], []
    action_numpy = []
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]

        state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent(obs.unsqueeze(0), predicited_enc=env_encoder)

        obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

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

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    data_plot(
        t,
        position=position,
        goal_position=ref_positon,
        velocity=lin_velocity,
        goal_velocity=ref_velocity,
        quaternion=quaternion,
        angular_velocity=ang_velocity,
        action=action_numpy,
        plot_text=text_plot,
        save_prefix="custom",
        save_path=results_folder,
    )

    env.unwrapped.vidwrite(results_folder + "custom.mp4")
    env.unwrapped.renderer.close()


def paper_fig_RMA_DATT_eval(
    cfg: Config,
    idx: [int, None] = None,
    best_model: bool = True,
    options: Union[None, dict] = None,
    return_traj_len: bool = False,
):
    # print("=================================")
    # print("Adapt Evaluation")

    # seeding again for sanity
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    rng = np.random.default_rng(seed=cfg.seed)

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
    # print("Model Path:", model_path)
    adapt_path = run_folder + "adapt_network.pt"
    # print("Adapt Path:", adapt_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )

    env = gym.make(cfg.env_id, cfg=cfg)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # del cfg.environment.trajectory_dataset

    priv_info_shape = env.unwrapped.priv_info_shape
    state_shape = env.unwrapped.state_obs_shape
    traj_shape = env.unwrapped.reference_traj_shape
    action_shape = env.action_space.shape[0]

    mass = env.unwrapped.model.body_mass[env.unwrapped.drone_id]
    inertia = env.unwrapped.model.body_inertia[env.unwrapped.drone_id]
    wind = env.unwrapped.model.opt.wind
    com = env.unwrapped.model.body_ipos[env.unwrapped.drone_id]

    agent = RMA_DATT(
        priv_info_shape=priv_info_shape,
        state_shape=state_shape,
        traj_shape=traj_shape,
        action_shape=action_shape,
    ).to(device)

    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()

    state_action_shape = state_shape + action_shape
    time_horizon = cfg.network.adapt_time_horizon

    adapt_input = time_horizon * state_action_shape
    adapt_output = cfg.network.env_encoder_output

    adapt_net = AdaptationNetwork(adapt_input, adapt_output).to(device)
    adapt_net.load_state_dict(torch.load(adapt_path, weights_only=True))

    state_action_buffer = torch.zeros(state_action_shape, time_horizon).to(device)

    obs, _ = env.reset(seed=cfg.seed, options=options)
    # del cfg.environment.trajectory_dataset

    t, ref_positon, ref_velocity = env.unwrapped.eval_trajectory(idx=idx)

    position, velocity = [], []
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.zeros(env.action_space.shape[0]).to(device)

    for i in range(len(t)):
        env.unwrapped.target_position = ref_positon[i]
        env.unwrapped.target_velocity = ref_velocity[i]

        env_obs = obs[: env.unwrapped.priv_info_shape]
        state_obs = obs[
            env.unwrapped.priv_info_shape : env.unwrapped.priv_info_shape
            + env.unwrapped.state_obs_shape
        ]
        traj_obs = obs[env.unwrapped.priv_info_shape + env.unwrapped.state_obs_shape :]

        state_action = torch.cat((state_obs, action.squeeze(0)), dim=-1)
        state_action_buffer = torch.cat(
            (state_action.unsqueeze(-1), state_action_buffer[:, :-1].clone()), dim=-1
        )
        env_encoder = adapt_net(state_action_buffer.flatten().unsqueeze(0))

        action = agent(obs.unsqueeze(0), predicited_enc=env_encoder)

        obs, rew, truncated, terminated, info = env.step(action.cpu().numpy()[0])
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        position.append(env.unwrapped.position)
        velocity.append(env.unwrapped.velocity)

    position = np.array(position)
    velocity = np.array(velocity)

    pos_error = ref_positon - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))

    # return position, and reference position as a 3D array
    # stack t to be same shape as position and ref_position
    t = t.reshape(-1, 1)
    t = np.repeat(t, position.shape[1], axis=1)
    stacked_position = np.array([t, position, ref_positon])
    print(stacked_position.shape)
    
    # pos_error = ref_positon - position
    # mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    
    return stacked_position
