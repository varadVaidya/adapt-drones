import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("data/snowy-lake-170_dataset_aligned.npz")
position = data["position"]

# Identify episode boundaries (NaN separators)
nan_mask = np.isnan(position).any(axis=1)
split_indices = np.where(nan_mask)[0]

# Split the position data into episodes
episodes = []
start_idx = 0
for end_idx in split_indices:
    episodes.append(position[start_idx:end_idx])
    start_idx = end_idx + 1
# Add last episode if it exists
if start_idx < len(position):
    episodes.append(position[start_idx:])

# Set up plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plot up to 5 episodes
for i, ep in enumerate(episodes[:5]):
    if len(ep) == 0:
        continue
    ax.plot(ep[:, 0], ep[:, 1], ep[:, 2], color=colors[i % len(colors)], label=f'Episode {i+1}')
    # Mark starting point
    ax.scatter(ep[0, 0], ep[0, 1], ep[0, 2], color=colors[i % len(colors)], marker='o', s=50)
    # Mark ending point
    ax.scatter(ep[-1, 0], ep[-1, 1], ep[-1, 2], color=colors[i % len(colors)], marker='X', s=70)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('First 5 Drone Trajectories (Start: dot, End: X)')
ax.legend()

plt.show()