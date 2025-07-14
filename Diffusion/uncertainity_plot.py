import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load dataset
data = np.load("data/slow_tcn_train_diff_dataset.npz")
position = data["position"]
residual_dynamics = data["residual_dynamics"]  # H

def split_episodes(arr):
    isnan = np.isnan(arr).any(axis=1)
    episode_indices = np.where(isnan)[0]
    episodes = []
    start = 0
    for idx in episode_indices:
        if idx > start:
            episodes.append(arr[start:idx])
        start = idx + 1
    if start < len(arr):
        episodes.append(arr[start:])
    return episodes

residual_episodes = split_episodes(residual_dynamics)
position_episodes = split_episodes(position)

# Filter episodes with length > median
lengths = np.array([ep.shape[0] for ep in residual_episodes])
median_len = int(np.median(lengths))
filtered_residual_episodes = [ep for ep in residual_episodes if ep.shape[0] > median_len]
filtered_position_episodes = [ep for ep in position_episodes if ep.shape[0] > median_len]

if len(filtered_residual_episodes) == 0:
    raise ValueError("No episodes longer than the median length.")

# Limit to 500 timesteps or shortest filtered episode
max_timesteps = 500
min_len = min(ep.shape[0] for ep in filtered_residual_episodes)
use_len = min(min_len, max_timesteps)

residual_stack = np.stack([ep[:use_len] for ep in filtered_residual_episodes])  # (num_episodes, use_len, 3)
position_stack = np.stack([ep[:use_len] for ep in filtered_position_episodes])  # (num_episodes, use_len, 3)

mean_H = np.nanmean(residual_stack, axis=0)  # (use_len, 3)
std_H = np.nanstd(residual_stack, axis=0)    # (use_len, 3)
mean_pos = np.nanmean(position_stack, axis=0)  # (use_len, 3)

# 3D plot: position (X, Y, Z) vs uncertainty (norm of std_H)
from mpl_toolkits.mplot3d import Axes3D

uncertainty_norm = np.linalg.norm(std_H, axis=1)  # (use_len,)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2], c=uncertainty_norm, cmap='viridis', s=40)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Position vs Uncertainty (||std_H||) for Episodes > Median Length')
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label('Uncertainty (Norm of std_H)')
plt.tight_layout()
plt.show()

# Find the longest episode
longest_idx = np.argmax([ep.shape[0] for ep in position_episodes])
longest_episode = position_episodes[longest_idx]  # shape: (length, 3)

# Plot the path followed by the longest episode in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(longest_episode[:, 0], longest_episode[:, 1], longest_episode[:, 2], color='blue', label='Longest Episode Path')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Path of Longest Episode')
ax.legend()
plt.tight_layout()
plt.show()

# 3D plots for X, Y, Z uncertainty components separately
labels = ['X', 'Y', 'Z']
cmaps = ['Reds', 'Greens', 'Blues']

for i in range(3):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        mean_pos[:, 0], mean_pos[:, 1], mean_pos[:, 2],
        c=std_H[:, i], cmap=cmaps[i], s=40
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Position vs Uncertainty (std_H {labels[i]}) for Episodes > Median Length')
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label(f'Uncertainty (Std of H {labels[i]})')
    plt.tight_layout()
    plt.show()

# Let's assume you have:
# std_H: (use_len, 3) - actual uncertainty (std of H) at each position
# predicted_std_H: (use_len, 3) - predicted uncertainty at each position (replace with your actual predicted values)

# For demonstration, let's create dummy predicted uncertainties (replace with your model's predictions)
predicted_std_H = std_H + np.random.normal(0, 0.02, std_H.shape)  # Replace with your predicted uncertainties

# Stack actual and predicted uncertainties for t-SNE
actual_flat = std_H.reshape(-1, 3)
pred_flat = predicted_std_H.reshape(-1, 3)
X = np.vstack([actual_flat, pred_flat])
labels = np.array(['Actual'] * len(actual_flat) + ['Predicted'] * len(pred_flat))

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# Plot t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[labels == 'Actual', 0], X_embedded[labels == 'Actual', 1], label='Actual', alpha=0.7)
plt.scatter(X_embedded[labels == 'Predicted', 0], X_embedded[labels == 'Predicted', 1], label='Predicted', alpha=0.7)
plt.title('t-SNE of Actual vs Predicted Uncertainties')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.tight_layout()
plt.show()

# Stack all collected H (residual_dynamics) values from filtered episodes
H_values = np.concatenate([ep[:use_len] for ep in filtered_residual_episodes], axis=0)  # shape: (num_episodes * use_len, 3)

# Run t-SNE on collected H values
tsne = TSNE(n_components=2, random_state=42)
H_embedded = tsne.fit_transform(H_values)

# Plot t-SNE for H values
plt.figure(figsize=(8, 6))
plt.scatter(H_embedded[:, 0], H_embedded[:, 1], alpha=0.7, s=10)
plt.title('t-SNE of Collected Residual Dynamics (H)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.show()