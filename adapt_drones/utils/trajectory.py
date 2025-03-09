import numpy as np
import random

from adapt_drones.utils.mpc_utils import q_dot_q, quaternion_inverse


def lemniscate_trajectory(
    discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max
):
    """

    :param quad:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = (
        transition_t_vec[-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = (
        down_coasting_t_vec[-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            ramp_t_vec,
            coasting_t_vec,
            transition_t_vec,
            down_coasting_t_vec,
            ramp_up_t_vec,
        )
    )
    alpha_vec = np.concatenate(
        (
            ramp_up_alpha,
            coasting_alpha,
            transition_alpha,
            down_coasting_alpha,
            ramp_up_alpha_end,
        )
    )

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = (
        radius * (np.sin(angle_vec) * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    )
    pos_traj_z = -z_dim * np.cos(4.0 * angle_vec)[np.newaxis, np.newaxis, :] + z

    vel_traj_x = -radius * (w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = (
        radius
        * (w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2)[
            np.newaxis, np.newaxis, :
        ]
    )
    vel_traj_z = (
        4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)[np.newaxis, np.newaxis, :]
    )

    x_ref = pos_traj_x.reshape(-1)
    y_ref = pos_traj_y.reshape(-1)
    z_ref = pos_traj_z.reshape(-1)

    vx_ref = vel_traj_x.reshape(-1)
    vy_ref = vel_traj_y.reshape(-1)
    vz_ref = vel_traj_z.reshape(-1)

    position_ref = np.vstack((x_ref, y_ref, z_ref)).T
    velocity_ref = np.vstack((vx_ref, vy_ref, vz_ref)).T
    acceleration_ref = np.gradient(velocity_ref, axis=0) / discretization_dt
    jerk_ref = np.gradient(acceleration_ref, axis=0) / discretization_dt

    traj_derivatives = np.stack(
        (position_ref, velocity_ref, acceleration_ref, jerk_ref), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, len(t_ref))), t_ref
    )


def random_trajectory(seed, total_time=16, dt=0.01):

    rng = np.random.default_rng(seed=seed)
    num_waypoints = 5
    t_waypoints = np.linspace(0, total_time, num_waypoints)

    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    waypoints_xy = rng.uniform(-1.5, 1.5, (num_waypoints, 2))
    waypoints_z = rng.uniform(0.5, 1.0, num_waypoints)

    waypoints_xy[-1] = waypoints_xy[0]
    waypoints_z[-1] = waypoints_z[0]

    waypoints = np.hstack((waypoints_xy, waypoints_z.reshape(-1, 1)))

    velocity_constraints = np.zeros((num_waypoints, 3))
    velocity_constraints[1:-1, :2] = rng.uniform(-1.0, 1.0, (num_waypoints - 2, 2))
    velocity_constraints[1:-1, 2] = rng.uniform(-0.75, 0.75, (num_waypoints - 2))

    acceleration_constraints = np.zeros((num_waypoints, 3))
    # acceleration_constraints[1:-1] = rng.uniform(0.0, 0.1, (num_waypoints - 2, 3))

    boundary_conditions = np.zeros((num_splines * 6, 3))
    coeffMatrix = np.zeros((num_splines * 6, num_splines * 6))

    for i in range(num_splines):
        T = spline_durations[i]
        idx = i * 6

        boundary_conditions[idx : idx + 6] = np.array(
            [
                waypoints[i],
                velocity_constraints[i],
                acceleration_constraints[i],
                waypoints[i + 1],
                velocity_constraints[i + 1],
                acceleration_constraints[i + 1],
            ]
        )
        coeffMatrix[idx : idx + 6, idx : idx + 6] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

    xTrajCoeff, yTrajCoeff, zTrajCoeff = np.linalg.solve(
        coeffMatrix, boundary_conditions
    ).T

    xVelCoeff, yVelCoeff, zVelCoeff = [], [], []

    for i in range(num_splines):
        idx = i * 6
        xVelCoeff.append(np.polyder(xTrajCoeff[idx : idx + 6]))
        yVelCoeff.append(np.polyder(yTrajCoeff[idx : idx + 6]))
        zVelCoeff.append(np.polyder(zTrajCoeff[idx : idx + 6]))

    xVelCoeff = np.array(xVelCoeff)
    yVelCoeff = np.array(yVelCoeff)
    zVelCoeff = np.array(zVelCoeff)

    t = np.linspace(0, total_time, int(total_time / dt))
    reference_trajectory = np.zeros((len(t), 6))

    for i in range(num_splines):
        T = spline_durations[i]
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]

        reference_trajectory[t_idx, 0:3] = np.array(
            [
                np.polyval(xTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(yTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(zTrajCoeff[i * 6 : i * 6 + 6], t_rel),
            ]
        ).T

        reference_trajectory[t_idx, 3:6] = np.array(
            [
                np.polyval(xVelCoeff[i], t_rel),
                np.polyval(yVelCoeff[i], t_rel),
                np.polyval(zVelCoeff[i], t_rel),
            ]
        ).T

    position = reference_trajectory[:, :3]
    position[:, 2] += 0.15
    velocity = reference_trajectory[:, 3:]
    speed_velocity = np.linalg.norm(velocity, axis=1)

    print(f"Max speed: {np.max(speed_velocity)}")

    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def random_straight_trajectory(seed, total_time=30, dt=0.01):

    rng = np.random.default_rng(seed=seed)

    num_points = 10
    t_waypoints = np.linspace(0, total_time, num_points)
    t = np.linspace(0, total_time, int(total_time / dt))

    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    waypoints_xy = rng.uniform(-1.75, 1.75, (num_points, 2))
    waypoints_z = rng.uniform(0.5, 1.5, num_points)
    waypoints = np.hstack((waypoints_xy, waypoints_z.reshape(-1, 1)))
    print(waypoints)

    # velocity vector points in the direction of the next waypoint
    # vel magnitude at each waypoint is determined by the distance to the next waypoint
    # and is constant for each spline

    spline_vel = np.diff(waypoints, axis=0) / spline_durations.reshape(-1, 1)

    position = np.zeros((len(t), 3))
    velocity = np.zeros((len(t), 3))
    for i in range(num_splines):
        # linear interpolate between waypoints in each spline to create position trajectory for
        # each spline, then stack them together
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]
        position[t_idx] = np.array(
            [
                np.interp(
                    t_rel,
                    [0, spline_durations[i]],
                    [waypoints[i, 0], waypoints[i + 1, 0]],
                ),
                np.interp(
                    t_rel,
                    [0, spline_durations[i]],
                    [waypoints[i, 1], waypoints[i + 1, 1]],
                ),
                np.interp(
                    t_rel,
                    [0, spline_durations[i]],
                    [waypoints[i, 2], waypoints[i + 1, 2]],
                ),
            ]
        ).T
        velocity[t_idx] = spline_vel[i]

    acceleration = np.zeros((len(t), 3))
    jerk = np.zeros((len(t), 3))

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def random_looped_trajectory(seed, total_time=16, dt=0.01):

    rng = np.random.default_rng(seed=seed)
    num_waypoints_per_loop = 5
    num_loops = 3
    num_waypoints = num_waypoints_per_loop * num_loops + 2
    # plus 2 for the start and end points
    t_waypoints = np.linspace(0, total_time, num_waypoints)
    spline_durations = np.diff(t_waypoints)
    num_splines = len(spline_durations)

    _start_point_xy = rng.uniform(-1.75, 1.75, 2)
    _start_point_z = rng.uniform(0.5, 1.25)
    _start_point = np.hstack((_start_point_xy, _start_point_z))

    _end_point = _start_point.copy()

    _loop_waypoints_xy = rng.uniform(-1.75, 1.75, (num_waypoints_per_loop, 2))
    _loop_waypoints_z = rng.uniform(0.5, 1.0, num_waypoints_per_loop)
    _loop_waypoints = np.hstack((_loop_waypoints_xy, _loop_waypoints_z.reshape(-1, 1)))

    _all_loop_waypoints = np.tile(_loop_waypoints, (num_loops, 1))
    # print(_all_loop_waypoints.shape)

    waypoints = np.vstack((_start_point, _all_loop_waypoints, _end_point))
    # print(waypoints.shape)

    _loop_velocity_constraints = np.zeros((num_waypoints_per_loop, 3))
    _loop_velocity_constraints[:, :2] = rng.uniform(
        -1.25, 1.25, (num_waypoints_per_loop, 2)
    )
    _loop_velocity_constraints[:, 2] = rng.uniform(-0.75, 0.75, num_waypoints_per_loop)
    _start_velocity_constraints = np.zeros(3)
    _end_velocity_constraints = np.zeros(3)

    velocity_constraints = np.vstack(
        (
            _start_velocity_constraints,
            np.tile(_loop_velocity_constraints, (num_loops, 1)),
            _end_velocity_constraints,
        )
    )

    acceleration_constraints = np.zeros_like(velocity_constraints)

    boundary_conditions = np.zeros((num_splines * 6, 3))
    coeffMatrix = np.zeros((num_splines * 6, num_splines * 6))

    for i in range(num_splines):
        T = spline_durations[i]
        idx = i * 6

        boundary_conditions[idx : idx + 6] = np.array(
            [
                waypoints[i],
                velocity_constraints[i],
                acceleration_constraints[i],
                waypoints[i + 1],
                velocity_constraints[i + 1],
                acceleration_constraints[i + 1],
            ]
        )
        coeffMatrix[idx : idx + 6, idx : idx + 6] = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ]
        )

    xTrajCoeff, yTrajCoeff, zTrajCoeff = np.linalg.solve(
        coeffMatrix, boundary_conditions
    ).T

    xVelCoeff, yVelCoeff, zVelCoeff = [], [], []

    for i in range(num_splines):
        idx = i * 6
        xVelCoeff.append(np.polyder(xTrajCoeff[idx : idx + 6]))
        yVelCoeff.append(np.polyder(yTrajCoeff[idx : idx + 6]))
        zVelCoeff.append(np.polyder(zTrajCoeff[idx : idx + 6]))

    xVelCoeff = np.array(xVelCoeff)
    yVelCoeff = np.array(yVelCoeff)
    zVelCoeff = np.array(zVelCoeff)

    t = np.linspace(0, total_time, int(total_time / dt))
    reference_trajectory = np.zeros((len(t), 6))

    for i in range(num_splines):
        T = spline_durations[i]
        t_idx = np.logical_and(t >= t_waypoints[i], t <= t_waypoints[i + 1])
        t_rel = t[t_idx] - t_waypoints[i]

        reference_trajectory[t_idx, 0:3] = np.array(
            [
                np.polyval(xTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(yTrajCoeff[i * 6 : i * 6 + 6], t_rel),
                np.polyval(zTrajCoeff[i * 6 : i * 6 + 6], t_rel),
            ]
        ).T

        reference_trajectory[t_idx, 3:6] = np.array(
            [
                np.polyval(xVelCoeff[i], t_rel),
                np.polyval(yVelCoeff[i], t_rel),
                np.polyval(zVelCoeff[i], t_rel),
            ]
        ).T

    position = reference_trajectory[:, :3]
    position[:, 2] += 0.15
    velocity = reference_trajectory[:, 3:]
    speed_velocity = np.linalg.norm(velocity, axis=1)

    print(f"Max speed: {np.max(speed_velocity)}")

    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    # return t, reference_trajectory[:, :3], reference_trajectory[:, 3:]

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def find_eulerian_circuit(graph, start):
    """
    Compute an Eulerian circuit in an undirected graph using Hierholzer's algorithm.

    Parameters
    ----------
    graph : dict
        Dictionary mapping each vertex to a list of neighboring vertices.
    start : hashable
        The starting vertex.

    Returns
    -------
    circuit : list
        A list of vertices representing the Eulerian circuit.
    """
    # Make a copy so we can modify the graph.
    graph_copy = {u: list(neighbors) for u, neighbors in graph.items()}
    circuit = []
    stack = [start]

    while stack:
        v = stack[-1]
        if graph_copy[v]:
            # Choose an arbitrary neighbor, remove the edge, and traverse.
            w = graph_copy[v].pop()
            graph_copy[w].remove(v)
            stack.append(w)
        else:
            circuit.append(stack.pop())

    return circuit


def octahedron_trajectory(scale=0.5, total_time=30, dt=0.01, hover_time=2.5):
    """
    Generates a continuous 3D trajectory that traces an Eulerian circuit
    along the edges of a regular octahedron, using a fixed time step dt.

    The six vertices of the octahedron (scaled by `scale`) are at:
      0: ( 1,  0,  0)
      1: (-1,  0,  0)
      2: ( 0,  1,  0)
      3: ( 0, -1,  0)
      4: ( 0,  0,  1)
      5: ( 0,  0, -1)

    An Eulerian circuit is computed (since every vertex has even degree), and then the
    path is reparameterized with constant speed. Positions are computed at time intervals
    of dt (here, 0.01 s).

    Parameters
    ----------
    scale : float
        Scale factor for the size of the octahedron.
    total_time : float
        Total flight time (seconds).
    dt : float
        Fixed time step (seconds).

    Returns
    -------
    t_total : 1D numpy array
        Array of time stamps from 0 to total_time in steps of dt.
    x, y, z : 1D numpy arrays
        Coordinates of the trajectory.
    waypoints : 2D numpy array
        The key vertices (in the order visited in the Eulerian circuit).
    """
    # Define the six vertices of a regular octahedron.
    vertices = {
        0: np.array([1, 0, 0]),
        1: np.array([-1, 0, 0]),
        2: np.array([0, 1, 0]),
        3: np.array([0, -1, 0]),
        4: np.array([0, 0, 1]),
        5: np.array([0, 0, -1]),
    }
    # Scale the vertices.
    for k in vertices:
        vertices[k] = vertices[k] * scale

    # Define the connectivity of the octahedron.
    graph = {
        0: [2, 3, 4, 5],
        1: [2, 3, 4, 5],
        2: [0, 1, 4, 5],
        3: [0, 1, 4, 5],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3],
    }

    # Compute an Eulerian circuit starting from vertex 0.
    circuit = find_eulerian_circuit(graph, start=0)
    # Convert vertex indices to coordinates.
    waypoints = np.array([vertices[v] for v in circuit])

    # Compute segment lengths and the cumulative arc-length along the path.
    segment_lengths = []
    cum_length = [0]  # cumulative distance starts at 0
    for i in range(len(waypoints) - 1):
        seg_len = np.linalg.norm(waypoints[i + 1] - waypoints[i])
        segment_lengths.append(seg_len)
        cum_length.append(cum_length[-1] + seg_len)
    cum_length = np.array(cum_length)
    total_length = cum_length[-1]

    # Determine constant speed needed to traverse the full length in total_time.
    speed = total_length / total_time

    # Create the time array with fixed dt.
    t_total = np.arange(0, total_time + dt, dt)

    # For each time, determine the arc-length traveled.
    s_values = speed * t_total

    # For each s, find the corresponding segment and interpolate the position.
    traj_points = []
    seg_index = 0
    for s in s_values:
        # If s equals total_length (or very close), use the last waypoint.
        if s >= total_length:
            traj_points.append(waypoints[-1])
            continue
        # Find the segment in which s falls.
        while seg_index < len(cum_length) - 1 and s > cum_length[seg_index + 1]:
            seg_index += 1
        # Interpolate between waypoints[seg_index] and waypoints[seg_index+1].
        s0 = cum_length[seg_index]
        s1 = cum_length[seg_index + 1]
        f = (s - s0) / (s1 - s0)
        point = waypoints[seg_index] + f * (
            waypoints[seg_index + 1] - waypoints[seg_index]
        )
        traj_points.append(point)

        # Now add the hover phase at the final endpoint.
    n_extra = int(hover_time / dt)
    # Create extra time stamps starting from the end of the trajectory.
    t_hover = t_total[-1] + np.arange(dt, hover_time + dt, dt)
    # For these extra time steps, the position remains constant at the final point.
    hover_points = np.tile(waypoints[-1], (len(t_hover), 1))

    # Concatenate the trajectory with the hover phase.
    t_total = np.concatenate([t_total, t_hover])
    traj_points = np.concatenate([traj_points, hover_points], axis=0)

    x = traj_points[:, 0]
    y = traj_points[:, 1]
    z = traj_points[:, 2]

    position = np.vstack((x, y, z)).T
    z += 1.0
    
    print("Zminmax",z.max(), z.min())

    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    vz = np.gradient(z) / dt

    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    az = np.zeros_like(vz)

    position = np.vstack((x, y, z)).T
    velocity = np.vstack((vx, vy, vz)).T
    acceleration = np.vstack((ax, ay, az)).T
    jerk = np.zeros_like(acceleration)

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, len(t_total))), t_total
    )


def satellite_orbit(radius=1.0, loops=3, total_time=10.0, dt=0.01):
    """
    Generates a 3D parametric 'satellite orbit' trajectory with a smooth speed
    profile: speed ramps up from 0, reaches a max, then ramps down to 0.

    Parameters
    ----------
    radius : float
        Radius of the orbit.
    loops : int
        How many times the drone will loop around in the path.
    total_time : float
        Total duration (in seconds) for completing the entire trajectory.
    n_points : int
        Number of points to sample along the trajectory.

    Returns
    -------
    t : 1D numpy array
        Time array, from 0 to total_time.
    x, y, z : 1D numpy arrays
        Coordinates of the trajectory in 3D.
    """
    # Create a time array from 0 to total_time
    n_points = int(total_time / dt)
    t = np.linspace(0, total_time, n_points)

    # Dimensionless progress parameter s(t) in [0,1],
    # with a smooth 'cosine ramp' from 0 to 1
    # s(0) = 0 and s(T) = 1
    s = 0.5 * (1 - np.cos(np.pi * t / total_time))

    # Orbit angles:
    #   theta(t) = loops * 2π * s(t)
    #   phi(t)   = 2π * s(t)
    theta = loops * 2 * np.pi * s
    phi = 2 * np.pi * s

    # Parametric equations for the rotated circle:
    x = -radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta)
    z = radius * np.cos(theta) * np.cos(phi)

    z += 1.0

    position = np.vstack((x, y, z)).T
    velocity = np.gradient(position, axis=0) / dt
    acceleration = np.gradient(velocity, axis=0) / dt
    jerk = np.gradient(acceleration, axis=0) / dt

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(traj_derivatives, np.zeros((2, len(t))), t)


def loop_trajectory(
    discretization_dt,
    radius,
    z,
    lin_acc,
    clockwise,
    yawing,
    v_max,
):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = (
        transition_t_vec[-1]
        + np.arange(0, coasting_duration, discretization_dt)
        + discretization_dt
    )
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = (
        down_coasting_t_vec[-1]
        + np.arange(0, ramp_up_t, discretization_dt)
        + discretization_dt
    )
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate(
        (
            ramp_t_vec,
            coasting_t_vec,
            transition_t_vec,
            down_coasting_t_vec,
            ramp_up_t_vec,
        )
    )
    alpha_vec = np.concatenate(
        (
            ramp_up_alpha,
            coasting_alpha,
            transition_alpha,
            down_coasting_alpha,
            ramp_up_alpha_end,
        )
    )

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = (
        alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    )
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = (
        -alpha_acc
        * np.pi
        / (2 * ramp_up_t)
        * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    )
    alpha_dt = np.concatenate(
        (
            ramp_up_alpha_dt,
            coasting_alpha_dt,
            transition_alpha_dt,
            coasting_alpha_dt,
            ramp_up_alpha_dt,
        )
    )

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = -(radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]

    xref = pos_traj_x.reshape(-1)
    yref = pos_traj_y.reshape(-1)
    zref = pos_traj_z.reshape(-1)

    vxref = vel_traj_x.reshape(-1)
    vyref = vel_traj_y.reshape(-1)
    vzref = np.zeros_like(vxref)

    position_ref = np.vstack((xref, yref, zref)).T
    velocity_ref = np.vstack((vxref, vyref, vzref)).T
    acceleration_ref = np.gradient(velocity_ref, axis=0) / discretization_dt
    jerk_ref = np.gradient(acceleration_ref, axis=0) / discretization_dt

    traj_derivatives = np.stack(
        (position_ref, velocity_ref, acceleration_ref, jerk_ref), axis=0
    ).transpose(0, 2, 1)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, len(t_ref))), t_ref
    )


def quintic_ramp(t, T):
    """
    Quintic polynomial ramp function and its derivatives.

    Args:
        t (numpy array): Time array.
        T (float): Total duration of the ramp.

    Returns:
        tuple: Scaling function s(t), its first derivative ds/dt,
               second derivative d^2s/dt^2, and third derivative d^3s/dt^3.
    """
    s = np.piecewise(
        t,
        [t < T, t >= T],
        [
            lambda t: 10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5,
            1.0,
        ],
    )
    ds = np.piecewise(
        t,
        [t < T, t >= T],
        [
            lambda t: (30 / T) * (t / T) ** 2
            - (60 / T) * (t / T) ** 3
            + (30 / T) * (t / T) ** 4,
            0.0,
        ],
    )
    dds = np.piecewise(
        t,
        [t < T, t >= T],
        [
            lambda t: (60 / T**2) * (t / T)
            - (180 / T**2) * (t / T) ** 2
            + (120 / T**2) * (t / T) ** 3,
            0.0,
        ],
    )
    ddds = np.piecewise(
        t,
        [t < T, t >= T],
        [
            lambda t: (60 / T**3)
            - (360 / T**3) * (t / T)
            + (360 / T**3) * (t / T) ** 2,
            0.0,
        ],
    )
    return s, ds, dds, ddds


def lissajous_3d_with_loops(
    A=1.0,
    B=1.0,
    C=1.0,
    a_w=1,
    b_w=2,
    delta=np.pi / 2,
    time_per_loop=5.0,
    total_time=20.0,
    dt=0.01,
):
    """
    Generate a 3D Lissajous curve with specified time per loop and total time.

    Args:
        a (float): Amplitude along the x-axis.
        b (float): Amplitude along the y-axis.
        c (float): Amplitude along the z-axis.
        delta (float): Phase difference for the y-axis.
        time_per_loop (float): Time taken to complete one loop.
        total_time (float): Total time for the trajectory (allows for multiple loops).
        num_points (int): Total number of points in the trajectory.

    Returns:
        tuple: Position, velocity, acceleration, and jerk arrays (Nx3 each).
    """
    num_loops = total_time / time_per_loop  # Number of loops
    omega = 2 * np.pi / time_per_loop  # Angular frequency for one loop
    num_points = int(total_time / dt)  # Total number of points in the trajectory
    ramp_duration = total_time / 10

    c_w = np.lcm(a_w, b_w)  # LCM of a_w and b_w

    print("Freqency ration of x-y axis:", a_w / b_w)
    print("Freqency ration of x-z axis:", a_w / c_w)
    print("Freqency ration of y-z axis:", b_w / c_w)

    # Time array
    t = np.linspace(0, total_time, num_points)
    # Compute ramping factors
    s_ramp_up, ds_ramp_up, dds_ramp_up, ddds_ramp_up = quintic_ramp(t, ramp_duration)
    s_ramp_down, ds_ramp_down, dds_ramp_down, ddds_ramp_down = quintic_ramp(
        total_time - t, ramp_duration
    )

    s = np.maximum(s_ramp_up, s_ramp_down)
    ds = np.maximum(ds_ramp_up, ds_ramp_down)
    dds = np.maximum(dds_ramp_up, dds_ramp_down)
    ddds = np.maximum(ddds_ramp_up, ddds_ramp_down)
    # Position
    x = A * np.sin(a_w * omega * t + delta)
    y = B * np.sin(b_w * omega * t)
    z = C * np.sin(c_w * omega * t)

    # Velocity
    vx = A * (a_w * omega) * np.cos(a_w * omega * t + delta)
    vy = B * (b_w * omega) * np.cos(b_w * omega * t)
    vz = C * (c_w * omega) * np.cos(c_w * omega * t)

    # Acceleration
    ax = -A * (a_w * omega) ** 2 * np.sin(a_w * omega * t + delta)
    ay = -B * (b_w * omega) ** 2 * np.sin(b_w * omega * t)
    az = -C * (c_w * omega) ** 2 * np.sin(c_w * omega * t)

    # Jerk
    jx = -A * (a_w * omega) ** 3 * np.cos(a_w * omega * t + delta)
    jy = -B * (b_w * omega) ** 3 * np.cos(b_w * omega * t)
    jz = -C * (c_w * omega) ** 3 * np.cos(c_w * omega * t)

    position = np.stack((s * x, s * y, s * z), axis=-1)
    position[:, 2] += (
        0.45 - position[:, 2].min() if 0.45 - position[:, 2].min() > 0 else 0
    )
    velocity = np.stack((ds * x + s * vx, ds * y + s * vy, ds * z + s * vz), axis=-1)
    acceleration = np.stack(
        (
            dds * x + 2 * ds * vx + s * ax,
            dds * y + 2 * ds * vy + s * ay,
            dds * z + 2 * ds * vz + s * az,
        ),
        axis=-1,
    )
    jerk = np.stack((jx, jy, jz), axis=-1)

    # stack the trajectory derivatives in the format of 4x3xN
    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)
    print(traj_derivatives.shape)

    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, num_points)), t
    )


def lissajous_3d_with_smooth_start(
    A=1.5,
    B=1.5,
    C=1.0,
    a_w=3,
    b_w=2,
    delta=3*np.pi/4,
    time_per_loop=5.0,
    total_time=20.0,
    dt=0.01,
    ramp_fraction=0.1,
):
    """
    Generate a 3D Lissajous curve with a smooth start and end.

    Args:
        a (float): Amplitude along the x-axis.
        b (float): Amplitude along the y-axis.
        c (float): Amplitude along the z-axis.
        delta (float): Phase difference for the y-axis.
        time_per_loop (float): Time taken to complete one loop.
        total_time (float): Total time for the trajectory (allows for multiple loops).
        num_points (int): Total number of points in the trajectory.
        ramp_fraction (float): Fraction of the total time used for velocity ramp-up/down.

    Returns:
        tuple: Position, velocity, acceleration, and jerk arrays (Nx3 each).
    """
    num_loops = total_time / time_per_loop  # Number of loops
    omega = 2 * np.pi / time_per_loop  # Angular frequency for one loop
    num_points = int(total_time / dt)  # Total number of points in the trajectory
    c_w = np.lcm(a_w, b_w)  # LCM of a_w and b_w

    # Time array
    t = np.linspace(0, total_time, num_points)

    # Ramp time and indices
    ramp_time = ramp_fraction * total_time
    ramp_steps = int(ramp_time / dt)

    # # Generate the velocity ramp
    ramp_up = np.sin(np.linspace(0, np.pi / 2, ramp_steps)) ** 2
    ramp_down = np.sin(np.linspace(np.pi / 2, np.pi, ramp_steps)) ** 2
    ramp = np.concatenate([ramp_up, np.ones(num_points - 2 * ramp_steps), ramp_down])

    # # Generate higher-order ramp using a quintic polynomial
    # ramp_up = np.linspace(0, 1, ramp_steps)
    # ramp_up = 10 * ramp_up**3 - 15 * ramp_up**4 + 6 * ramp_up**5  # Quintic polynomial
    # ramp_down = np.flip(ramp_up)
    # ramp = np.concatenate([ramp_up, np.ones(num_points - 2 * ramp_steps), ramp_down])

    # Position
    x = A * np.sin(a_w * omega * t + delta)
    y = B * np.sin(b_w * omega * t)
    z = C * np.sin(c_w * omega * t)

    # Adjust position for ramped velocity
    x = np.cumsum(ramp * np.gradient(x, t), axis=0) * dt
    y = np.cumsum(ramp * np.gradient(y, t), axis=0) * dt
    y += 0.5
    z = np.cumsum(ramp * np.gradient(z, t), axis=0) * dt

    # Velocity
    vx = ramp * np.gradient(x, t)
    vy = ramp * np.gradient(y, t)
    vz = ramp * np.gradient(z, t)

    # Acceleration
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    az = np.gradient(vz, t)

    # Jerk
    jx = np.gradient(ax, t)
    jy = np.gradient(ay, t)
    jz = np.gradient(az, t)

    position = np.stack((x, y, z), axis=-1)
    position[:, 2] += (
        0.45 - position[:, 2].min() if 0.45 - position[:, 2].min() > 0 else 0
    )
    velocity = np.stack((vx, vy, vz), axis=-1)
    acceleration = np.stack((ax, ay, az), axis=-1)
    jerk = np.stack((jx, jy, jz), axis=-1)

    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)

    # return t, position, velocity, acceleration, jerk
    return minimum_snap_trajectory_generator(
        traj_derivatives, np.zeros((2, num_points)), t
    )


def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = (
        traj_derivatives[2, :, :].T
        + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    )
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust**2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))

    # new way to compute attitude:
    # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
    e_z = np.array([[0.0, 0.0, 1.0]])
    q_w = 1.0 + np.sum(e_z * z_b, axis=1)
    q_xyz = np.cross(e_z, z_b)
    q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
    q = q / np.sqrt(np.sum(q**2, 1))[:, np.newaxis]

    # Use numerical differentiation of quaternions
    q_dot = np.gradient(q, axis=0) / discretization_dt
    w_int = np.zeros((len_traj, 3))
    for i in range(len_traj):
        w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
    rate[:, 0] = w_int[:, 0]
    rate[:, 1] = w_int[:, 1]
    rate[:, 2] = w_int[:, 2]

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    full_acc = traj_derivatives[2, :, :].T
    reference_traj = np.concatenate((full_pos, full_vel, q, rate), 1)

    return reference_traj, t_ref


if __name__ == "__main__":
    # t, pos, vel = lemniscate_trajectory(
    #     discretization_dt=0.01,
    #     radius=5,
    #     z=1,
    #     lin_acc=0.25,
    #     clockwise=True,
    #     yawing=False,
    #     v_max=5,
    # )
    # seed = 4151110721
    seed = -1
    # seed = 2174653152
    seed = seed if seed > 0 else random.randint(0, 2**32 - 1)
    print(f"Seed: {seed}")
    # t, pos, vel = random_trajectory(seed=seed)
    # t, pos, vel = loop_trajectory(
    #     discretization_dt=0.01,
    #     radius=5,
    #     z=1,
    #     lin_acc=0.20,
    #     clockwise=True,
    #     yawing=False,
    #     v_max=5,
    # )

    # traj, t = random_straight_trajectory(seed=seed, total_time=40)

    traj, t = lissajous_3d_with_smooth_start(
        A=1.5,
        B=1.5,
        C=0.5,
        a_w=2,
        b_w=1,
        delta=3 * np.pi / 4,
        time_per_loop=10.0,
        total_time=40.0,
        dt=0.01,
        ramp_fraction=0.1,
    )
    # valid_traj = False

    # while not valid_traj:
    #     traj, t = random_looped_trajectory(seed=seed, total_time=45)
    #     pos = traj[:, :3]
    #     valid_traj = (pos[:, 2].min() > 0.2) and (pos[:, 2].max() < 1.6)
    #     valid_traj = valid_traj and (pos[:, 0].min() > -2.0) and (pos[:, 0].max() < 2.0)
    #     valid_traj = valid_traj and (pos[:, 1].min() > -2.0) and (pos[:, 1].max() < 2.0)

    #     if not valid_traj:
    #         seed = random.randint(0, 2**32 - 1)
    #         print(f"Seed: {seed}")

    # print("Valid trajectory found at seed: ", seed)
    # print("Trajectory shape: ", traj.shape)
    pos = traj[:, :3]
    print("Zmin: ", pos[:, 2].min())
    print("Zmax: ", pos[:, 2].max())
    vel = traj[:, 3:6]
    acc = np.gradient(vel, axis=0) / 0.01
    quat = traj[:, 9:13]
    omega = traj[:, 13:]
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(t, pos[:, 0], label="x")
    axs[0].plot(t, pos[:, 1], label="y")
    axs[0].plot(t, pos[:, 2], label="z")

    axs[1].plot(t, vel[:, 0], label="vx")
    axs[1].plot(t, vel[:, 1], label="vy")
    axs[1].plot(t, vel[:, 2], label="vz")

    axs[0].legend()
    axs[1].legend()

    axs[2].plot(t, acc[:, 0], label="ax")
    axs[2].plot(t, acc[:, 1], label="ay")
    axs[2].plot(t, acc[:, 2], label="az")
    axs[2].legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    # ax.set_box_aspect([1, 1, 1])
    # ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("trajectory.png")
    plt.show()
