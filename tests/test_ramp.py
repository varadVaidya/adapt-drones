# Re-importing libraries and defining the required functions due to environment reset
import numpy as np
import matplotlib.pyplot as plt


# Function to generate 3D Lissajous with smooth ramp
def lissajous_3d_with_ramp(
    a=1.0,
    b=1.0,
    c=1.0,
    delta=np.pi / 2,
    time_per_loop=5.0,
    total_time=20.0,
    ramp_duration=2.0,
    num_points=1000,
):
    """
    Generate a 3D Lissajous curve with velocity ramp-up and ramp-down phases.

    Args:
        a (float): Amplitude along the x-axis.
        b (float): Amplitude along the y-axis.
        c (float): Amplitude along the z-axis.
        delta (float): Phase difference for the y-axis.
        time_per_loop (float): Time taken to complete one loop.
        total_time (float): Total time for the trajectory (multiple loops).
        ramp_duration (float): Duration of ramp-up and ramp-down phases.
        num_points (int): Total number of points in the trajectory.

    Returns:
        tuple: Position, velocity, acceleration, and jerk arrays (Nx3 each).
    """
    num_loops = total_time / time_per_loop  # Number of loops
    omega = 2 * np.pi / time_per_loop  # Angular frequency for one loop
    dt = total_time / num_points  # Time step

    # Time array
    t = np.linspace(0, total_time, num_points)

    # Smooth scaling function (sinusoidal ramp)
    s = np.piecewise(
        t,
        [t < ramp_duration, t > (total_time - ramp_duration)],
        [
            lambda t: 0.5 * (1 - np.cos(np.pi * t / ramp_duration)),  # Ramp-up
            lambda t: 0.5
            * (1 - np.cos(np.pi * (total_time - t) / ramp_duration)),  # Ramp-down
            1.0,
        ],  # Steady phase
    )
    ds = np.gradient(s, dt)  # First derivative of s(t)
    dds = np.gradient(ds, dt)  # Second derivative of s(t)
    ddds = np.gradient(dds, dt)  # Third derivative of s(t)

    # Position
    x = a * np.sin(omega * t)
    y = b * np.sin(2 * omega * t + delta)
    z = c * np.sin(3 * omega * t)

    # Velocity
    vx = a * omega * np.cos(omega * t)
    vy = b * 2 * omega * np.cos(2 * omega * t + delta)
    vz = c * 3 * omega * np.cos(3 * omega * t)

    # Acceleration
    ax = -a * omega**2 * np.sin(omega * t)
    ay = -b * (2 * omega) ** 2 * np.sin(2 * omega * t + delta)
    az = -c * (3 * omega) ** 2 * np.sin(3 * omega * t)

    # Jerk
    jx = -a * omega**3 * np.cos(omega * t)
    jy = -b * (2 * omega) ** 3 * np.cos(2 * omega * t + delta)
    jz = -c * (3 * omega) ** 3 * np.cos(3 * omega * t)

    # Apply scaling
    position = np.stack((s * x, s * y, s * z), axis=-1)
    velocity = np.stack((ds * x + s * vx, ds * y + s * vy, ds * z + s * vz), axis=-1)
    acceleration = np.stack(
        (
            dds * x + 2 * ds * vx + s * ax,
            dds * y + 2 * ds * vy + s * ay,
            dds * z + 2 * ds * vz + s * az,
        ),
        axis=-1,
    )
    jerk = np.stack(
        (
            ddds * x + 3 * dds * vx + 3 * ds * ax + s * jx,
            ddds * y + 3 * dds * vy + 3 * ds * ay + s * jy,
            ddds * z + 3 * dds * vz + 3 * ds * az + s * jz,
        ),
        axis=-1,
    )

    return position, velocity, acceleration, jerk


# Generate the 3D Lissajous curve with smooth ramp-up and ramp-down
lissajous_pos, lissajous_vel, _, _ = lissajous_3d_with_ramp(
    a=1.0,
    b=1.5,
    c=2.0,
    delta=np.pi / 4,
    time_per_loop=5.0,
    total_time=20.0,
    ramp_duration=3.0,
    num_points=1000,
)

# Visualize the trajectory and velocity
fig = plt.figure(figsize=(14, 6))

# 3D Trajectory plot
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(
    lissajous_pos[:, 0],
    lissajous_pos[:, 1],
    lissajous_pos[:, 2],
    label="Position (3D Lissajous)",
)
ax1.set_title("3D Lissajous Curve with Smooth Ramp", fontsize=14)
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")
ax1.set_zlabel("Z-axis")
ax1.legend()

# Velocity plot
t = np.linspace(0, 20.0, 1000)  # Time array
ax2 = fig.add_subplot(122)
ax2.plot(t, np.linalg.norm(lissajous_vel, axis=1), label="Velocity Magnitude")
ax2.set_title("Velocity Magnitude Over Time", fontsize=14)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity Magnitude (m/s)")
ax2.legend()

plt.tight_layout()
plt.show()
