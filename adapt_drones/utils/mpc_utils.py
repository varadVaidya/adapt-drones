import os
import math
import random

import pyquaternion
import numpy as np
import casadi as cs
from scipy.interpolate import interp1d

from adapt_drones.utils.rotation import object_velocity, transform_spatial

# from adapt_drones.controller.mpc.quad_3d import Quadrotor3D


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array(
            [
                [0, -v[0], -v[1], -v[2]],
                [v[0], 0, v[2], -v[1]],
                [v[1], -v[2], 0, v[0]],
                [v[2], v[1], -v[0], 0],
            ]
        )

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0),
    )


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)


def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.

    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw


    return np.array([t0, t1, t2, t3])


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ),
            cs.horzcat(
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ),
            cs.horzcat(
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ),
        )

    return rot_mat


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q**2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q


def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print("Error while removing directory: {0}".format(directory))


def discretize_dynamics_and_cost(
    t_horizon, n_points, m_steps_per_point, x, u, dynamics_f, cost_f, ind
):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), angle (4D), velocity (3D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :param ind: Only used for trajectory tracking. Index of cost function to use.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and
    """

    if isinstance(cost_f, list):
        # Select the list of cost functions
        cost_f = cost_f[ind * m_steps_per_point : (ind + 1) * m_steps_per_point]
    else:
        cost_f = [cost_f]

    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    x0 = x
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(x=x, u=u)["x_dot"]
        k2 = dynamics_f(x=x + dt / 2 * k1, u=u)["x_dot"]
        k3 = dynamics_f(x=x + dt / 2 * k2, u=u)["x_dot"]
        k4 = dynamics_f(x=x + dt * k3, u=u)["x_dot"]
        x_out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x = x_out

        if cost_f and cost_f[j] is not None:
            q = q + cost_f[j](x=x, u=u)["q"]

    return cs.Function("F", [x0, u], [x, q], ["x0", "p"], ["xf", "qf"])


def simulate_plant(
    quad,
    wopt,
    simulation_dt,
    simulate_func,
    t_horizon=None,
    dt_vec=None,
    progress_bar=False,
):
    # Default_parameters
    if t_horizon is None and dt_vec is None:
        raise ValueError("At least the t_horizon or dt should be provided")

    if t_horizon is None:
        t_horizon = np.sum(dt_vec)

    state_safe = quad.get_state(quaternion=True, stacked=True)

    # Compute true simulated trajectory
    total_sim_time = t_horizon
    sim_traj = []

    if dt_vec is None:
        change_control_input = np.arange(0, t_horizon, t_horizon / (w_opt.shape[0]))[1:]
        first_dt = t_horizon / w_opt.shape[0]
        sim_traj.append(quad.get_state(quaternion=True, stacked=True))
    else:
        change_control_input = np.cumsum(dt_vec)
        first_dt = dt_vec[0]

    t_start_ep = 1e-6
    int_range = (
        tqdm(np.arange(t_start_ep, total_sim_time, simulation_dt))
        if progress_bar
        else np.arange(t_start_ep, total_sim_time, simulation_dt)
    )

    current_ind = 0
    past_ind = 0
    for t_elapsed in int_range:
        ref_u = w_opt[current_ind, :].T
        simulate_func(ref_u)
        if t_elapsed + simulation_dt >= first_dt:
            current_ind = (
                np.argwhere(change_control_input <= t_elapsed + simulation_dt)[-1, 0]
                + 1
            )
            if past_ind != current_ind:
                sim_traj.append(quad.get_state(quaternion=True, stacked=True))
                past_ind = current_ind

    if dt_vec is None:
        sim_traj.append(quad.get_state(quaternion=True, stacked=True))
    sim_traj = np.squeeze(sim_traj)

    quad.set_state(state_safe)

    return sim_traj


def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.

    :param traj: N x 13 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    return [p_traj, a_traj, v_traj, r_traj]


def get_reference_chunk(
    reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling
):
    """
    Extracts the reference states and controls for the current MPC optimization given the over-sampled counterparts.

    :param reference_traj: The reference trajectory, which has been finely over-sampled by a factor of
    reference_over_sampling. It should be a vector of shape (Nx13), where N is the length of the trajectory in samples.
    :param reference_u: The reference controls, following the same requirements as reference_traj. Should be a vector
    of shape (Nx4).
    :param current_idx: Current index of the trajectory tracking. Should be an integer number between 0 and N-1.
    :param n_mpc_nodes: Number of MPC nodes considered in the optimization.
    :param reference_over_sampling: The over-sampling factor of the reference trajectories. Should be a positive
    integer.
    :return: Returns the chunks of reference selected for the current MPC iteration. Two numpy arrays will be returned:
        - An ((N+1)x13) array, corresponding to the reference trajectory. The first row is the state of current_idx.
        - An (Nx4) array, corresponding to the reference controls.
    """

    # Dense references
    ref_traj_chunk = reference_traj[
        current_idx : current_idx + (n_mpc_nodes + 1) * reference_over_sampling, :
    ]
    ref_u_chunk = reference_u[
        current_idx : current_idx + n_mpc_nodes * reference_over_sampling, :
    ]

    # Indices for down-sampling the reference to number of MPC nodes
    downsample_ref_ind = np.arange(
        0,
        min(reference_over_sampling * (n_mpc_nodes + 1), ref_traj_chunk.shape[0]),
        reference_over_sampling,
        dtype=int,
    )

    ref_traj_chunk = ref_traj_chunk[downsample_ref_ind, :]
    ref_u_chunk = ref_u_chunk[
        downsample_ref_ind[: max(len(downsample_ref_ind) - 1, 1)], :
    ]

    return ref_traj_chunk, ref_u_chunk


def get_reference_trajectory(eval_npz, idx, control_dt):
    """
    Gives the position and velocity reference as accurate.
    Orientation and Angular velocity are set to zero.
    Control reference is also set to 0.
    """

    eval_traj = eval_npz[idx]
    rows_not_nan = sum(~np.isnan(eval_traj[:, 1]))
    eval_traj = eval_traj[:rows_not_nan]

    trajectory_dt = eval_traj[1, 0] - eval_traj[0, 0]
    T = eval_traj[-1, 0]

    resampled_dt = control_dt
    resampled_t = np.arange(0, T, resampled_dt)

    interp = interp1d(eval_traj[:, 0], eval_traj[:, 1:], axis=0)
    resampled_traj = interp(resampled_t)
    resampled_traj = np.hstack((resampled_t.reshape(-1, 1), resampled_traj))

    assert resampled_traj.shape[1] == eval_traj.shape[1]
    # print("Resampled trajectory shape:", resampled_traj.shape)

    # Position, Quaternion, Velocity, Angular Velocity
    reference_trajectory = np.zeros((resampled_traj.shape[0], 13))
    reference_trajectory[:] = np.nan

    reference_inputs = np.zeros((resampled_traj.shape[0], 4))

    # position
    reference_trajectory[:, 0:3] = resampled_traj[:, 1:4]
    reference_trajectory[:, 3:7] = np.array([1, 0, 0, 0])
    reference_trajectory[:, 7:10] = resampled_traj[:, 8:11]
    reference_trajectory[:, 10:] = np.array([0, 0, 0])

    reference_timestamps = resampled_t

    return reference_trajectory, reference_inputs, reference_timestamps


def fluid_force_model(mass, inertia, wind, pos, rot, vel, rho, beta):

    factor = 3 / (2 * mass)

    r_x = np.sqrt(factor * (inertia[1] + inertia[2] - inertia[0]))
    r_y = np.sqrt(factor * (inertia[2] + inertia[0] - inertia[1]))
    r_z = np.sqrt(factor * (inertia[0] + inertia[1] - inertia[2]))

    # check if the radii are real numbers
    assert np.all(np.isreal([r_x, r_y, r_z]))
    r = np.array([r_x, r_y, r_z])

    local_vel = object_velocity(pos, rot, vel)

    wind6 = np.zeros(6)
    wind6[3:] = wind
    local_wind = transform_spatial(wind6, 0, pos, pos, rot)
    # print(local_wind)

    local_vel[3:] -= local_wind[3:]

    wind_force = np.zeros(3)
    wind_torque = np.zeros(3)

    # viscous force

    r_eq = np.sum(r) / 3
    wind_force += -6 * beta * np.pi * r_eq * local_vel[3:]
    wind_torque += -8 * beta * np.pi * r_eq**3 * local_vel[:3]

    # drag force
    prod_r = np.prod(r)
    wind_force += -2 * rho * (prod_r / r) * np.abs(local_vel[3:]) * local_vel[3:]
    sum_r_4 = np.sum(r**4)
    wind_torque += (
        (-1 / 2) * rho * r * (sum_r_4 - r**4) * np.abs(local_vel[:3]) * local_vel[:3]
    )

    force_world = np.dot(rot, wind_force)
    torque_world = np.dot(rot, wind_torque)

    force_torque_world = np.concatenate([force_world, torque_world])

    return force_torque_world
