# import sympy as sp
# import numpy as np
# import matplotlib.pyplot as plt

# t = sp.symbols('t')
# A, B, C = 1.25, 1.25, 0.25
# a, b, delta = 2/3, 3/3, sp.pi/4
# c = sp.lcm(a, b)
# print(c)

# x = A * sp.sin(a * t + delta)
# y = B * sp.sin(b * t)
# z = C * sp.sin(c * t)
# vx = sp.diff(x, t)
# vy = sp.diff(y, t)
# vz = sp.diff(z, t)

# ax = sp.diff(vx, t)
# ay = sp.diff(vy, t)
# az = sp.diff(vz, t)

# print("Velocity (x):", vx)
# print("Velocity (y):", vy)
# print("Velocity (z):", vz)

# print("Acceleration (x):", ax)
# print("Acceleration (y):", ay)
# print("Acceleration (z):", az)

# # T_total = 2 * np.pi / np.gcd(a, b)
# # print(f'Total Period: {T_total}')
# T_total = 20
# t_vals = np.linspace(0, T_total * 2, 1000)

# x_func = sp.lambdify(t, x, "numpy")
# y_func = sp.lambdify(t, y, "numpy")
# z_func = sp.lambdify(t, z, "numpy")

# vx_func = sp.lambdify(t, vx, "numpy")
# vy_func = sp.lambdify(t, vy, "numpy")
# vz_func = sp.lambdify(t, vz, "numpy")

# ax_func = sp.lambdify(t, ax, "numpy")
# ay_func = sp.lambdify(t, ay, "numpy")
# az_func = sp.lambdify(t, az, "numpy")

# x_vals = x_func(t_vals)
# y_vals = y_func(t_vals)
# z_vals = z_func(t_vals) + 1.5

# vx_vals = vx_func(t_vals)
# vy_vals = vy_func(t_vals)
# vz_vals = vz_func(t_vals)

# ax_vals = ax_func(t_vals)
# ay_vals = ay_func(t_vals)
# az_vals = az_func(t_vals)

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_vals, y_vals, z_vals, label='Lissajous Curve')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()

# fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# axs[0, 0].plot(t_vals, vx_vals, label='Velocity (X)', color='r')
# axs[1, 0].plot(t_vals, vy_vals, label='Velocity (Y)', color='g')
# axs[2, 0].plot(t_vals, vz_vals, label='Velocity (Z)', color='b')

# axs[0, 1].plot(t_vals, ax_vals, label='Acceleration (X)', color='r')
# axs[1, 1].plot(t_vals, ay_vals, label='Acceleration (Y)', color='g')
# axs[2, 1].plot(t_vals, az_vals, label='Acceleration (Z)', color='b')

# for i, ax in enumerate(axs[:, 0]):
#     ax.set_title(f'Velocity Plot {["X", "Y", "Z"][i]}')
#     ax.legend()
#     ax.grid(True)

# for i, ax in enumerate(axs[:, 1]):
#     ax.set_title(f'Acceleration Plot {["X", "Y", "Z"][i]}')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    # Position
    x = A * np.sin(a_w * omega * t + delta)
    y = B * np.sin(b_w * omega * t)
    z = C * np.sin(c_w * omega * t)

    z += 0.25 - z.min() if 0.25 - z.min() > 0 else 0

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

    # position = np.stack((s * x, s * y, s * z), axis=-1)
    # velocity = np.stack((ds * x + s * vx, ds * y + s * vy, ds * z + s * vz), axis=-1)
    # acceleration = np.stack(
    #     (
    #         dds * x + 2 * ds * vx + s * ax,
    #         dds * y + 2 * ds * vy + s * ay,
    #         dds * z + 2 * ds * vz + s * az,
    #     ),
    #     axis=-1,
    # )
    position = np.stack((x, y, z), axis=-1)
    velocity = np.stack((vx, vy, vz), axis=-1)
    acceleration = np.stack((ax, ay, az), axis=-1)
    jerk = np.stack((jx, jy, jz), axis=-1)

    # stack the trajectory derivatives in the format of 4x3xN
    traj_derivatives = np.stack(
        (position, velocity, acceleration, jerk), axis=0
    ).transpose(0, 2, 1)
    print(traj_derivatives.shape)

    return position, velocity, acceleration, jerk


# Generate the 3D Lissajous trajectory
lissajous_pos, lissajous_vel, lissajous_acc, lissajous_jerk = lissajous_3d_with_loops(
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


# Visualize the 3D Lissajous curve
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory
ax.plot(
    lissajous_pos[:, 0],
    lissajous_pos[:, 1],
    lissajous_pos[:, 2],
    label="3D Lissajous Curve",
)
ax.set_title("3D Lissajous Curve", fontsize=14)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()


# # plot the velocity, acceleration in two sperate plots
# fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# # Plot the velocity components
# axs[0, 0].plot(lissajous_vel[:, 0], label="Velocity (X)", color="r")
# axs[1, 0].plot(lissajous_vel[:, 1], label="Velocity (Y)", color="g")
# axs[2, 0].plot(lissajous_vel[:, 2], label="Velocity (Z)", color="b")

# # Plot the acceleration components
# axs[0, 1].plot(lissajous_acc[:, 0], label="Acceleration (X)", color="r")
# axs[1, 1].plot(lissajous_acc[:, 1], label="Acceleration (Y)", color="g")
# axs[2, 1].plot(lissajous_acc[:, 2], label="Acceleration (Z)", color="b")

# # Set the titles and legends
# for i, ax in enumerate(axs[:, 0]):
#     ax.set_title(f"Velocity Plot {['X', 'Y', 'Z'][i]}")
#     ax.legend()
#     ax.grid(True)

# for i, ax in enumerate(axs[:, 1]):
#     ax.set_title(f"Acceleration Plot {['X', 'Y', 'Z'][i]}")
#     ax.legend()
#     ax.grid(True)

plt.tight_layout()
plt.show()
