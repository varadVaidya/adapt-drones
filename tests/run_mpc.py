""" Tracks a specified trajectory on the simplified simulator using the data-augmented MPC."""

import time
import numpy as np
import pkg_resources
from tqdm import tqdm
from typing import Union
import random

from adapt_drones.utils.mpc_utils import (
    separate_variables,
    get_reference_chunk,
    get_reference_trajectory,
)
from adapt_drones.controller.mpc.quad_3d_mpc import Quad3DMPC
from adapt_drones.controller.mpc.quad_3d import Quadrotor3D


def prepare_quadrotor_mpc(
    rng,
    simulation_dt=1e-2,
    n_mpc_node=10,
    q_diagonal=None,
    r_diagonal=None,
    q_mask=None,
    quad_name=None,
    t_horizon=1.0,
    noisy=False,
    acados_path_postfix: Union[str, None] = None,
):
    # Default Q and R matrix for LQR cost
    if q_diagonal is None:
        q_diagonal = np.array(
            [10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
    if r_diagonal is None:
        r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])
    if q_mask is None:
        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

    my_quad = Quadrotor3D(noisy=noisy, rng=rng)

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
    seed = -1
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    rng = np.random.default_rng(seed=seed)

    quad_mpc = prepare_quadrotor_mpc(
        noisy=noisy,
        acados_path_postfix="test_postfix",
        rng=rng,
        n_mpc_node=20,
        t_horizon=1.0,
    )

    my_quad = quad_mpc.quad
    n_mpc_node = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    print("Time Horizon: ", t_horizon)
    simulation_dt = quad_mpc.simulation_dt
    print("Simulation dt: ", simulation_dt)
    reference_over_sampling = 1
    control_period = t_horizon / (n_mpc_node * reference_over_sampling)
    # control_period = simulation_dt
    print("Control Period: ", control_period)

    # load reference trajectory
    traj_path = pkg_resources.resource_filename(
        "adapt_drones", "assets/slow_pi_tcn_eval_mpc.npy"
    )
    trajector_dataset = np.load(traj_path)
    traj_idx = 0

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
    quad_current_state[3:7] /= np.linalg.norm(quad_current_state[3:7])

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
    mean_error = np.mean(position_error)
    rms_error = np.sqrt(np.mean(position_error**2))

    print("\n::::::::::::: SIMULATION RESULTS :::::::::::::\n")
    print("Mean optimization time: %.3f ms" % mean_opt_time)
    print("Tracking RMSE: %.7f m\n" % rms_error)
    print("mean_error: ", mean_error)


if __name__ == "__main__":
    noisy = True
    main(noisy=noisy)
