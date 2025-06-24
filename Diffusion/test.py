import numpy as np

# Load your dataset
data = np.load('data/snowy-lake-170_dataset_aligned.npz')

# Pick a valid data row (e.g., not a NaN separator)
i = 100 

# --- Build the Input Vector (X) ---
# 1. Dynamic State at time t
pos_t = data['position'][i]
vel_t = data['velocity'][i]
ori_t = data['orientation'][i]

# 2. Control History from time t-1
act_t_minus_1 = data['action_t_minus_1'][i]

# 3. Static Parameters
arm_len = data['arm_length']
nom_mass = data['nominal_mass']

# Concatenate everything into a single flat vector
# Note: need to make scalars into 1-element arrays to concatenate
X = np.concatenate([
    pos_t,
    vel_t,
    ori_t,
    act_t_minus_1,
    np.array([arm_len]),
    np.array([nom_mass])
])

# --- Get the Output/Target Vector (Y) ---
Y = data['residual_dynamics'][i]


print("--- Single Data Point for ML Model ---")
print(f"Input Vector X (shape: {X.shape}):\n{X}")
print(f"\nTarget Vector Y (shape: {Y.shape}):\n{Y}")

# Don't forget to close the file
data.close()