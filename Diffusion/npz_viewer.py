import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load the .npy files
traj1 = np.load("adapt_drones/assets/slow_pi_tcn_train.npy")
traj2 = np.load("adapt_drones/assets/slow_pi_tcn_eval.npy")  # Change to your second file

# Get the first trajectory from each
first_traj1 = traj1[4]
first_traj2 = traj2[4]

# Assuming each trajectory is (length, 3) for (x, y, z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(first_traj1[:, 0], first_traj1[:, 1], first_traj1[:, 2], label='File 1')
ax.plot(first_traj2[:, 0], first_traj2[:, 1], first_traj2[:, 2], label='File 2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()