import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Load tracked trajectory
data = np.load('data/snowy-lake-170_dataset_aligned.npz')
array = data['position']

# Load reference trajectory (5th trajectory, index 4)
ref_trajs = np.load('adapt_drones/assets/slow_pi_tcn_train.npy')
ref_traj = ref_trajs[4]  # shape: (timesteps, 13)
ref_xyz = ref_traj[:, 1:4]  # columns 1,2,3 are x,y,z

# Find rows where all elements are nan (episode separators)
is_nan_row = np.all(np.isnan(array), axis=1)
episode_indices = np.where(is_nan_row)[0]

# Add start and end indices
indices = np.concatenate(([0], episode_indices, [len(array)]))

# Plot each episode trajectory one by one in 3D, overlaying the reference
for i in range(len(indices) - 1):
    start = indices[i] + 1  # skip the nan row
    end = indices[i + 1]
    traj = array[start:end]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot tracked trajectory
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='Tracked Trajectory')
    # Plot reference trajectory in dotted lines
    ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], 'k--', label='Reference Trajectory (5th)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory of Episode {i+1}')
    ax.legend()
    plt.show()