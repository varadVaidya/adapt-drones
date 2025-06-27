import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = np.load("data/slow_tcn_train_diff_dataset.npz")
position = data["position"]
velocity = data["velocity"]
acceleration = data["acceleration"]
residual_dynamics = data["residual_dynamics"]  # H

# Remove NaN separator rows
mask = ~np.isnan(position).any(axis=1)
pos_clean = position[mask]
vel_clean = velocity[mask]
acc_clean = acceleration[mask]
H_clean = residual_dynamics[mask]

# Plot
fig, axs = plt.subplots(4, 1, figsize=(12, 14))

# Position
for i, label in enumerate(['X', 'Y', 'Z']):
    sns.kdeplot(pos_clean[:, i], ax=axs[0], label=label)
axs[0].set_title("Position Distribution")
axs[0].set_xlabel("Position (m)")
axs[0].set_ylabel("Density")
axs[0].legend()

# Velocity
for i, label in enumerate(['X', 'Y', 'Z']):
    sns.kdeplot(vel_clean[:, i], ax=axs[1], label=label)
axs[1].set_title("Velocity Distribution")
axs[1].set_xlabel("Velocity (m/s)")
axs[1].set_ylabel("Density")
axs[1].legend()

# Acceleration
for i, label in enumerate(['X', 'Y', 'Z']):
    sns.kdeplot(acc_clean[:, i], ax=axs[2], label=label)
axs[2].set_title("Acceleration Distribution")
axs[2].set_xlabel("Acceleration (m/s²)")
axs[2].set_ylabel("Density")
axs[2].legend()

# H (residual dynamics)
for i, label in enumerate(['X', 'Y', 'Z']):
    sns.kdeplot(H_clean[:, i], ax=axs[3], label=f"H {label}")
axs[3].set_title("Residual Dynamics (H) Distribution")
axs[3].set_xlabel("Residual (N)")
axs[3].set_ylabel("Density")
axs[3].legend()

plt.tight_layout()
plt.show()