""" Tracks a specified trajectory on the simplified simulator using the data-augmented MPC."""

import time
import numpy as np
import pkg_resources
from tqdm import tqdm
from typing import Union
import random
from dataclasses import dataclass

from adapt_drones.utils.mpc_utils import (
    separate_variables,
    get_reference_chunk,
    get_reference_trajectory,
)
from adapt_drones.controller.mpc.quad_3d_mpc import Quad3DMPC
from adapt_drones.controller.mpc.quad_3d import Quadrotor3D
from adapt_drones.utils.dynamics import CustomDynamics, ScaledDynamics
from adapt_drones.cfgs.config import *


@dataclass
class Args:
    env_id: str
    run_name: str
    seed: int = 4551
    agent: str = "RMA_DATT"
    scale: bool = True
    wind_bool: bool = True


def prepare_quadrotor_mpc(
    rng,
    simulation_dt=1e-2,
    n_mpc_node=20,
    q_diagonal=None,
    r_diagonal=None,
    q_mask=None,
    quad_name=None,
    t_horizon=1,
    noisy=False,
    acados_path_postfix: Union[str, None] = None,
    cfg=None,
):
    # Default Q and R matrix for LQR cost
    if q_diagonal is None:
        q_diagonal = np.array([5, 5, 5, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
    if r_diagonal is None:
        r_diagonal = np.array([0.5, 0.5, 0.5, 0.5])
    if q_mask is None:
        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

    scale = 0.1

    scaled_ground = ScaledDynamics(seed=rng, arm_length=scale, cfg=cfg, do_random=False)
    scaled_changed = ScaledDynamics(seed=rng, arm_length=scale, cfg=cfg, do_random=True)

    ground_dynamics = CustomDynamics(
        arm_length=scaled_ground.length_scale(),
        mass=scaled_ground.mass_scale(),
        ixx=scaled_ground.ixx_yy_scale(),
        iyy=scaled_ground.ixx_yy_scale(),
        izz=scaled_ground.izz_scale(),
        km_kf=scaled_ground.torque_to_thrust(),
    )

    changed_dynamics = CustomDynamics(
        arm_length=scaled_changed.length_scale(),
        mass=scaled_changed.mass_scale(),
        ixx=scaled_changed.ixx_yy_scale(),
        iyy=scaled_changed.ixx_yy_scale(),
        izz=scaled_changed.izz_scale(),
        km_kf=scaled_changed.torque_to_thrust(),
    )

    # changed_dynamics = ground_dynamics

    # get the difference in dynamics
    dynamics_diff = {
        "arm_length": changed_dynamics.arm_length - ground_dynamics.arm_length,
        "mass": changed_dynamics.mass - ground_dynamics.mass,
        "ixx": changed_dynamics.ixx - ground_dynamics.ixx,
        "iyy": changed_dynamics.iyy - ground_dynamics.iyy,
        "izz": changed_dynamics.izz - ground_dynamics.izz,
        "km_kf": changed_dynamics.km_kf - ground_dynamics.km_kf,
    }

    for key, value in dynamics_diff.items():
        print(f"{key}: {value}")

    my_quad = Quadrotor3D(
        noisy=noisy,
        rng=rng,
        changed_dynamics=changed_dynamics,
        ground_dynamics=ground_dynamics,
    )

    if quad_name is None:
        quad_name = "my_quad"

    optimisation_dt = t_horizon / n_mpc_node

    quad_mpc = Quad3DMPC(
        quad=my_quad,
        t_horizon=t_horizon,
        n_nodes=n_mpc_node,
        q_cost=q_diagonal,
        r_cost=r_diagonal,
        optimization_dt=optimisation_dt,
        simulation_dt=simulation_dt,
        model_name=quad_name,
        q_mask=q_mask,
        acados_path_postfix=acados_path_postfix,
    )

    return quad_mpc


def main(noisy=False):

    env_run = ["traj_v3", "true-durian-33", True]
    args = Args(env_id=env_run[0], run_name=env_run[1], wind_bool=env_run[2])

    cfg = Config(
        env_id=args.env_id,
        seed=args.seed,
        eval=True,
        run_name=args.run_name,
        agent=args.agent,
        scale=args.scale,
        wind_bool=args.wind_bool,
    )
    seed = 4553232
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    rng = np.random.default_rng(seed=seed)

    quad_mpc = prepare_quadrotor_mpc(
        noisy=noisy,
        acados_path_postfix="test_postfix",
        rng=rng,
        cfg=cfg,
        n_mpc_node=10,
        t_horizon=1,
    )

    my_quad = quad_mpc.quad
    n_mpc_node = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 2
    control_period = t_horizon / (n_mpc_node * reference_over_sampling)

    print("Time Horizon: ", t_horizon)
    print("Simulation dt: ", simulation_dt)
    print("Control Period: ", control_period)

    # load reference trajectory
    traj_path = pkg_resources.resource_filename(
        "adapt_drones", "assets/slow_pi_tcn_eval_mpc.npy"
    )
    trajector_dataset = np.load(traj_path)
    traj_idx = 1

    reference_trajectory, reference_input, reference_timestamp = (
        get_reference_trajectory(trajector_dataset, traj_idx, control_period)
    )

    reference_input[:, 0] = quad_mpc.quad.mass * 9.81 / quad_mpc.quad.max_thrust

    delta_pos = rng.uniform(-0.1, 0.1, 3)
    delta_vel = rng.uniform(-0.1, 0.1, 3)
    delta_ori = rng.uniform(-0.1, 0.1, 4)
    delta_ori /= np.linalg.norm(delta_ori)
    delta_rate = rng.uniform(-0.05, 0.05, 3)

    delta_init_pos = np.concatenate([delta_pos, delta_ori, delta_vel, delta_rate])
    quad_current_state = reference_trajectory[0, :] + delta_init_pos
    # quad_current_state[3:7] /= np.linalg.norm(quad_current_state[3:7])

    my_quad.set_state(quad_current_state)

    ref_u = np.zeros(4)
    quad_trajectory = np.zeros((len(reference_timestamp), len(quad_current_state)))
    u_optimised_seq = np.zeros((len(reference_timestamp), 4))

    current_idx = 0
    mean_opt_time = 0.0

    total_sim_time = 0.0

    print("Running Simulation")
    for current_idx in tqdm(range(reference_trajectory.shape[0])):
        quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
        quad_trajectory[current_idx, :] = np.expand_dims(quad_current_state, axis=0)

        # get reference trajectory chunk
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            reference_trajectory,
            reference_input,
            current_idx,
            n_mpc_node,
            reference_over_sampling,
        )

        # Set the reference for the OCP
        model_ind = quad_mpc.set_reference(
            x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk
        )

        t_opt_init = time.time()
        w_opt, x_pred = quad_mpc.optimize(use_model=model_ind, return_x=True)

        mean_opt_time += time.time() - t_opt_init

        ref_u = np.squeeze(np.array(w_opt[:4]))
        u_optimised_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

        simulation_time = 0.0
        while simulation_time < control_period:
            simulation_time += simulation_dt
            total_sim_time += simulation_dt
            quad_mpc.simulate(ref_u)

    u_optimised_seq[current_idx, :] = np.reshape(ref_u, (1, -1))

    quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
    quad_trajectory[-1, :] = np.expand_dims(quad_current_state, axis=0)
    u_optimised_seq[-1, :] = np.reshape(ref_u, (1, -1))

    # Average optimisation time
    mean_opt_time = mean_opt_time / current_idx * 1000
    position_error = np.linalg.norm(
        quad_trajectory[:, :3] - reference_trajectory[:, :3], axis=1
    )

    # position_error = position_error[250:]
    mean_error = np.mean(position_error)
    rms_error = np.sqrt(np.mean(position_error**2))

    print("\n::::::::::::: SIMULATION RESULTS :::::::::::::\n")
    print("Mean optimization time: %.3f ms" % mean_opt_time)
    print("Tracking RMSE: %.7f m\n" % rms_error)
    print("mean_error: ", mean_error)

    quad_mpc.clear()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        quad_trajectory[:, 0],
        quad_trajectory[:, 1],
        quad_trajectory[:, 2],
        label="quad",
    )

    ax.plot(
        reference_trajectory[:, 0],
        reference_trajectory[:, 1],
        reference_trajectory[:, 2],
        "--",
        label="ref",
    )

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(quad_trajectory[:, 0], label="x")
    axs[0].plot(reference_trajectory[:, 0], "--", label="ref")
    axs[0].set_title("X")
    axs[0].legend()

    axs[1].plot(quad_trajectory[:, 1], label="y")
    axs[1].plot(reference_trajectory[:, 1], "--", label="ref")
    axs[1].set_title("Y")

    axs[2].plot(quad_trajectory[:, 2], label="z")
    axs[2].plot(reference_trajectory[:, 2], "--", label="ref")
    axs[2].set_title("Z")

    fig = plt.figure()
    plt.plot(u_optimised_seq[:, 0], label="u1")
    plt.plot(u_optimised_seq[:, 1], label="u2")
    plt.plot(u_optimised_seq[:, 2], label="u3")
    plt.plot(u_optimised_seq[:, 3], label="u4")
    plt.show()


if __name__ == "__main__":
    noisy = True
    main(noisy=noisy)
