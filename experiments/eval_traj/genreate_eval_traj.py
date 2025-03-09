import os
import numpy as np
import matplotlib.pyplot as plt

from adapt_drones.utils.trajectory import (
    satellite_orbit,
    octahedron_trajectory,
    random_looped_trajectory,
    lissajous_3d_with_smooth_start,
)


if __name__ == "__main__":
    butterfly = lissajous_3d_with_smooth_start(
        A=1.5,
        B=1.5,
        C=0.5,
        a_w=2,
        b_w=1,
        delta=3 * np.pi / 4,
        time_per_loop=20.0,
        total_time=50.0,
        ramp_fraction=0.1,
    )
    random_looped_trajectory = random_looped_trajectory(
        seed=2174653152, total_time=50.0
    )

    satellite_orbit = satellite_orbit(radius=0.6, loops=8, total_time=60)
    octahedron = octahedron_trajectory(
        scale=0.625, total_time=30, dt=0.01, hover_time=2.5
    )

    print("butterfly", butterfly[0].shape)
    print("random_looped_trajectory", random_looped_trajectory[0].shape)
    print("satellite_orbit", satellite_orbit[0].shape)
    print("octahedron", octahedron[0].shape)

    exp_eval_array = np.empty(
        (
            4,
            max(
                butterfly[0].shape[0],
                random_looped_trajectory[0].shape[0],
                satellite_orbit[0].shape[0],
                octahedron[0].shape[0],
            ),
            14,
        )
    )

    exp_eval_array[:] = np.nan

    exp_eval_array[0, :butterfly[0].shape[0], 1:] = butterfly[0]
    exp_eval_array[1, :satellite_orbit[0].shape[0], 1:] = satellite_orbit[0]
    exp_eval_array[2, :random_looped_trajectory[0].shape[0], 1:] = random_looped_trajectory[0]
    exp_eval_array[3, :octahedron[0].shape[0], 1:] = octahedron[0]

    # # fill the first column with the time
    # exp_eval_array[0, :butterfly.shape[0], 0] = np.linspace(0, 50, butterfly.shape[0])
    # exp_eval_array[1, :satellite_orbit.shape[0], 0] = np.linspace(0, 60, satellite_orbit.shape[0])
    # exp_eval_array[2, :random_looped_trajectory.shape[0], 0] = np.linspace(0, 50, random_looped_trajectory.shape[0])
    # exp_eval_array[3, :octahedron.shape[0], 0] = np.linspace(0, 30, octahedron.shape[0])

    exp_eval_array[0, :butterfly[0].shape[0], 0] = butterfly[1]
    exp_eval_array[1, :satellite_orbit[0].shape[0], 0] = satellite_orbit[1]
    exp_eval_array[2, :random_looped_trajectory[0].shape[0], 0] = random_looped_trajectory[1]
    exp_eval_array[3, :octahedron[0].shape[0], 0] = octahedron[1]

    np.save("adapt_drones/assets/exp_eval_array.npy", exp_eval_array)

    # # this was in time[0], position[1:4], velocity[4:7], quaternion[7:11], angular velocity[11:14]
    # # mpc requires time, position, quaternion, velocity, angular velocity
    # # so we need to swap the velocity and quaternion columns

    exp_mpc_eval_array = exp_eval_array.copy()

    for i in range(4):
        time = exp_eval_array[i, :, 0].copy()
        positions = exp_eval_array[i, :, 1:4].copy()
        velocities = exp_eval_array[i, :, 4:7].copy()
        quaternions = exp_eval_array[i, :, 7:11].copy()
        angular_velocities = exp_eval_array[i, :, 11:14].copy()

        exp_mpc_eval_array[i,:,0] = time
        exp_mpc_eval_array[i,:,1:4] = positions
        exp_mpc_eval_array[i,:,4:8] = quaternions
        exp_mpc_eval_array[i,:,8:11] = velocities
        exp_mpc_eval_array[i,:,11:14] = angular_velocities

    np.save("adapt_drones/assets/exp_eval_array_mpc.npy", exp_mpc_eval_array)

    # ## xdadapts also needs acceleration. We can calculate it from the velocities

    exp_xadapt_array = np.empty(
        (
            4,
            max(
                butterfly[0].shape[0],
                random_looped_trajectory[0].shape[0],
                satellite_orbit[0].shape[0],
                octahedron[0].shape[0],
            ),
            17,
        )
    )
    exp_xadapt_array[:] = np.nan

    for i in range(4):

        rows_not_nan = sum(~np.isnan(exp_eval_array[i, :, 0]))
        time = exp_eval_array[i, :rows_not_nan, 0].copy()
        xadapt_positions = exp_eval_array[i, :rows_not_nan, 1:4].copy()
        xadapt_velocities = exp_eval_array[i, :rows_not_nan, 4:7].copy()
        xadapt_quaternions = exp_eval_array[i, :rows_not_nan, 7:11].copy()
        xadapt_angular_velocities = exp_eval_array[i, :rows_not_nan, 11:14].copy()
        # xadapt_acceleration = np.zeros_like(xadapt_velocities)
        xadapt_acceleration = np.gradient(xadapt_velocities, axis=0) / np.gradient(time).reshape(-1, 1)
        # check if any nan values in the acceleration
        print(np.any(np.isnan(xadapt_acceleration)))

        fig, axs = plt.subplots(5, 1)
        axs[0].plot(time, xadapt_positions)
        axs[0].set_title("Position")

        axs[1].plot(time, xadapt_velocities)
        axs[1].set_title("Velocity")

        axs[2].plot(time, xadapt_quaternions)
        axs[2].set_title("Quaternion")

        axs[3].plot(time, xadapt_angular_velocities)
        axs[3].set_title("Angular Velocity")

        axs[4].plot(time, xadapt_acceleration)
        axs[4].set_title("Acceleration")

        plt.show()
        exp_xadapt_array[i, :rows_not_nan, 0] = time
        exp_xadapt_array[i, :rows_not_nan, 1:4] = xadapt_positions
        exp_xadapt_array[i, :rows_not_nan, 4:7] = xadapt_velocities
        exp_xadapt_array[i, :rows_not_nan, 7:10] = xadapt_acceleration
        exp_xadapt_array[i, :rows_not_nan, 10:13] = xadapt_angular_velocities
        exp_xadapt_array[i, :rows_not_nan, 13:17] = xadapt_quaternions

    np.save("adapt_drones/assets/exp_xadapt_array.npy", exp_xadapt_array)
