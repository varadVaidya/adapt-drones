""" Evaluated Trajectory Scale with perfect dynamics. This will serve as the ideal baseline for the MPC controller.
The dynamics scale is set to the xadapt's scale. """

import os
import numpy as np
import time
import pkg_resources
import tqdm
import concurrent.futures

from adapt_drones.utils.mpc_utils import (
    separate_variables,
    get_reference_chunk,
    get_reference_trajectory,
)
from adapt_drones.controller.mpc.quad_3d_mpc import Quad3DMPC
from adapt_drones.controller.mpc.quad_3d import Quadrotor3D
from adapt_drones.utils.dynamics import CustomDynamics
from experiments.xadapt.scale_functions import *

import mujoco
from adapt_drones.utils.rotation import euler2mat
import pandas as pd


def prepare_quadrotor_mpc(
    ground_dynamics: CustomDynamics,
    changed_dynamics: CustomDynamics,
    simulation_dt=1e-2,
    n_mpc_node=10,
    q_diagonal=None,
    r_diagonal=None,
    q_mask=None,
    quad_name=None,
    t_horizon=1.0,
    noisy=False,
):
    # Default Q and R matrix for LQR cost
    if q_diagonal is None:
        q_diagonal = np.array(
            [2.5, 2.5, 2.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
    if r_diagonal is None:
        r_diagonal = np.array([0.1, 0.1, 0.1, 0.1])
    if q_mask is None:
        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

    my_quad = Quadrotor3D(
        ground_dynamics=ground_dynamics, changed_dynamics=changed_dynamics, noisy=noisy
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
    )

    return quad_mpc


def mpc_traj_seed_scale(idx, seed, scale, give_MPC_truth=False):
    """
    Runs and evaluates MPC for a given trajectory, seed and scale.
    Returns idx, seed, scale, mean error, rms error.
    """
    rng = np.random.default_rng(seed=seed)

    xadapt_ground = Dynamics(seed=rng, c=scale, do_random=False)
    xadapt_changed = Dynamics(seed=rng, c=scale, do_random=True)

    ground_dynamics = CustomDynamics(
        arm_length=xadapt_ground.length_scale(),
        mass=xadapt_ground.mass_scale(),
        ixx=xadapt_ground.ixx_yy_scale(),
        iyy=xadapt_ground.ixx_yy_scale(),
        izz=xadapt_ground.izz_scale(),
        km_kf=xadapt_ground.torque_to_thrust(),
    )

    changed_dynamics = CustomDynamics(
        arm_length=xadapt_changed.length_scale(),
        mass=xadapt_changed.mass_scale(),
        ixx=xadapt_changed.ixx_yy_scale(),
        iyy=xadapt_changed.ixx_yy_scale(),
        izz=xadapt_changed.izz_scale(),
        km_kf=xadapt_changed.torque_to_thrust(),
    )

    quad_mpc = prepare_quadrotor_mpc(
        ground_dynamics=ground_dynamics,
        changed_dynamics=changed_dynamics if not give_MPC_truth else ground_dynamics,
        noisy=True,
    )

    my_quad = quad_mpc.quad
    # print(my_quad.mass, my_quad.mass_actual)
    n_mpc_node = quad_mpc.n_nodes
    t_horizon = quad_mpc.t_horizon
    simulation_dt = quad_mpc.simulation_dt
    reference_over_sampling = 5
    control_period = t_horizon / (n_mpc_node * reference_over_sampling)

    # load reference trajectory
    traj_path = pkg_resources.resource_filename(
        "adapt_drones", "assets/slow_pi_tcn_eval_mpc.npy"
    )
    trajector_dataset = np.load(traj_path)
    reference_trajectory, reference_input, reference_timestamp = (
        get_reference_trajectory(trajector_dataset, idx, control_period)
    )

    reference_input[:, 0] = quad_mpc.quad.mass * 9.81 / quad_mpc.quad.max_thrust
    delta_pos = rng.uniform(-0.1, 0.1, 3)

    # whole mess oof euler angles. I HATE EULER ANGLES
    delta_roll_pitch = rng.uniform(-0.15, 0.15, 2)
    delta_quat = np.zeros(4)
    euler = np.array([delta_roll_pitch[0], delta_roll_pitch[1], 0])
    mat = euler2mat(euler).reshape(9)
    mujoco.mju_mat2Quat(delta_quat, mat)

    delta_vel = rng.uniform(-0.125, 0.125, 3)
    delta_rate = rng.uniform(-0.05, 0.05, 3)

    delta_init_pos = np.concatenate([delta_pos, delta_quat, delta_vel, delta_rate])

    quad_current_state = (reference_trajectory[0, :] + delta_init_pos).tolist()
    quad_current_state[3:7] = delta_quat / np.linalg.norm(delta_quat)

    my_quad.set_state(quad_current_state)

    ref_u = np.zeros(4)
    quad_trajectory = np.zeros((len(reference_timestamp), len(quad_current_state)))
    u_optimised_seq = np.zeros((len(reference_timestamp), 4))

    current_idx = 0
    mean_opt_time = 0.0

    total_sim_time = 0.0

    for current_idx in range(reference_trajectory.shape[0]):
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

    return idx, seed, scale, mean_error, rms_error


if __name__ == "__main__":
    c = np.linspace(0, 1, 15)
    seeds = np.arange(4551, 4551 + 16)
    idx = np.arange(0, 13)

    print(c)
    print(seeds)
    print(idx)

    num_lengths = len(c)
    num_seeds = len(seeds)
    num_idx = len(idx)

    give_MPC_truth = [False, True]

    print(mpc_traj_seed_scale(0, 4551, 1, False))

    for i, MPC_truth in enumerate(give_MPC_truth):

        traj_mpc_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
        traj_mpc_eval[:, :, :, :] = np.nan

        prefix = "ground_dynamics_" if MPC_truth else "changed_dynamics_"

        map_iterable = [
            (int(i), int(seed), float(_c), give_MPC_truth[0])
            for i in idx
            for seed in seeds
            for _c in c
        ]

        print(len(map_iterable))
        print("=====================================================================")
        print(f"Running MPC with {'ground' if MPC_truth else 'changed'} dynamics")

        # with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:

        #     results = list(
        #         tqdm.tqdm(
        #             executor.map(
        #                 mpc_traj_seed_scale,
        #                 *zip(*map_iterable),
        #             ),
        #             total=len(map_iterable),
        #         )
        #     )

        # do plain old fashioned for loop
        results = []
        for i, seed, _c, MPC_truth in tqdm.tqdm(map_iterable, desc="Running MPC"):
            results.append(mpc_traj_seed_scale(i, seed, _c, MPC_truth))

        for result in results:
            _idx, _seed, _c, _mean_error, _rms_error = result
            idx_idx = np.where(idx == _idx)[0][0]
            seed_idx = np.where(seeds == _seed)[0][0]
            c_idx = np.where(c == _c)[0][0]

            traj_mpc_eval[idx_idx, seed_idx, c_idx] = [
                _idx,
                _seed,
                _c,
                _mean_error,
                _rms_error,
            ]

        print(traj_mpc_eval.shape)

        # check for nans
        print(np.argwhere(np.isnan(traj_mpc_eval)))

        run_folder = "experiments/mpc/results/"

        os.makedirs(run_folder, exist_ok=True)

        np.save(run_folder + prefix + "traj_mpc_eval.npy", traj_mpc_eval)

        # remove the seeds which performed worst across all scales, in
        # sense of mean error

        idx_sort_eval = np.argsort(traj_mpc_eval[:, :, :, 3], axis=2)
        sorted_traj_eval = traj_mpc_eval[:, idx_sort_eval[0], :, :]

        # remove top 3 and bottom 3 seeds
        # traj_mpc_eval = sorted_traj_eval[:, 3:-3, :, :]

        # compile the data into a csv file contain errors and std dev
        data_compile = np.zeros((traj_mpc_eval.shape[0] * traj_mpc_eval.shape[2], 6))

        # ^ 0: idx, 1: c,
        # ^ 3:  avg mean error, 4: std dev mean error,
        # ^ 4: avg rms error, 5: std dev rms error

        data_compile[:, 0] = np.repeat(
            np.arange(traj_mpc_eval.shape[0]), traj_mpc_eval.shape[2]
        )  # tile idx

        data_compile[:, 1] = np.tile(
            traj_mpc_eval[0, 0, :, 2], traj_mpc_eval.shape[0]
        )  # tile c

        data_compile[:, 2] = np.mean(traj_mpc_eval[:, :, :, 3], axis=1).reshape(
            traj_mpc_eval.shape[0] * traj_mpc_eval.shape[2]
        )  # avg mean error

        data_compile[:, 3] = np.std(traj_mpc_eval[:, :, :, 3], axis=1).reshape(
            traj_mpc_eval.shape[0] * traj_mpc_eval.shape[2]
        )  # std dev mean error

        data_compile[:, 4] = np.mean(traj_mpc_eval[:, :, :, 4], axis=1).reshape(
            traj_mpc_eval.shape[0] * traj_mpc_eval.shape[2]
        )  # avg rms error

        data_compile[:, 5] = np.std(traj_mpc_eval[:, :, :, 4], axis=1).reshape(
            traj_mpc_eval.shape[0] * traj_mpc_eval.shape[2]
        )  # std dev rms error

        print(data_compile.shape)

        print("Saving data_compile to: ", run_folder + prefix + "mpc_traj_excel.csv")

        header = [
            "idx",
            "c",
            "mean_error",
            "std_dev_mean_error",
            "rms_error",
            "std_dev_rms_error",
        ]

        np.savetxt(
            run_folder + prefix + "mpc_traj_excel.csv",
            data_compile,
            delimiter=",",
            header=",".join(header),
            comments="",
        )

        def get_error_info(df, idx_value, c_value):
            # Filter the row based on idx and c values
            row = df.loc[(df["idx"] == idx_value) & (df["c"] == c_value)]

            # If the row exists, round and concatenate mean_error and std_dev_mean_error
            if not row.empty:
                mean_error = round(row["mean_error"].values[0], 4)
                std_dev_mean_error = round(row["std_dev_mean_error"].values[0], 4)
                return f"{mean_error} \pm {std_dev_mean_error}"
            else:
                return "No match found"

        df = pd.read_csv(run_folder + prefix + "mpc_traj_excel.csv")

        tex_table = []
        for idx_value in np.unique(df["idx"]):
            idx_row = []
            for c_value in np.unique(df["c"]):
                idx_row.append(get_error_info(df, idx_value, c_value))
            tex_table.append(idx_row)

        # convert tex_table to pandas dataframe
        tex_table = np.array(tex_table)
        tex_table = pd.DataFrame(
            tex_table,
            columns=np.round(
                Dynamics(seed=0, c=np.unique(df["c"]), do_random=False).length_scale(),
                3,
            ),
            index=np.unique(df["idx"]),
        )
        # save the table to a csv file
        tex_table.to_csv(run_folder + prefix + "mpc_table.csv")
