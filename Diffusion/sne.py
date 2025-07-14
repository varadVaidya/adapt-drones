import torch
import numpy as np
from unet import ConditionalUnet1D
from diffusers import DDPMScheduler
from dataset import create_train_val_dataloaders
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Configs ---
VAL_DATA_PATH = 'data/slow_tcn_train_diff_dataset.npz'
SCALER_DIR = 'scalers/'
OBS_HORIZON, PRED_HORIZON = 4, 8
STATE_DIM, ACTION_DIM, RESIDUAL_DIM = 10, 4, 3
GLOBAL_COND_DIM = OBS_HORIZON * (STATE_DIM + ACTION_DIM)
BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load validation data ---
_, val_dataloader, _ = create_train_val_dataloaders(
    train_dataset_path=VAL_DATA_PATH,
    val_dataset_path=VAL_DATA_PATH,
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    batch_size=BATCH_SIZE,
    scaler_dir=SCALER_DIR
)

# --- Load trained model ---
model = ConditionalUnet1D(input_dim=RESIDUAL_DIM, global_cond_dim=GLOBAL_COND_DIM).to(device)
model.load_state_dict(torch.load("models/diffusion_unet_ema_best.pt", map_location=device))
model.eval()

noise_scheduler = DDPMScheduler(num_train_timesteps=20, beta_schedule='linear', prediction_type='epsilon')

# --- Collect predictions ---
all_pred_H, all_true_H = [], []
episode_count, max_episodes = 0, 500

with torch.no_grad():
    for batch in val_dataloader:
        if episode_count >= max_episodes:
            break
        condition_vector = batch['condition'].to(device)
        target_vector = batch['target'].to(device)
        B = condition_vector.shape[0]

        clean_sequence = target_vector.view(-1, PRED_HORIZON, RESIDUAL_DIM)
        noise = torch.randn_like(clean_sequence)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_sequence = noise_scheduler.add_noise(clean_sequence, noise, timesteps)

        noise_pred = model(noisy_sequence, timesteps, global_cond=condition_vector)
        pred_H = noisy_sequence - noise_pred  # Denoised prediction

        all_pred_H.append(pred_H.cpu().numpy())
        all_true_H.append(clean_sequence.cpu().numpy())
        episode_count += B

# --- Flatten and Truncate ---
all_pred_H = np.concatenate(all_pred_H, axis=0)[:max_episodes * PRED_HORIZON].reshape(-1, RESIDUAL_DIM)
all_true_H = np.concatenate(all_true_H, axis=0)[:max_episodes * PRED_HORIZON].reshape(-1, RESIDUAL_DIM)

# --- Run t-SNE ---
X = np.vstack([all_true_H, all_pred_H])
labels = np.array([0] * len(all_true_H) + [1] * len(all_pred_H))  # 0: Ground Truth, 1: Predicted

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_embedded = tsne.fit_transform(X)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[labels == 0, 0], X_embedded[labels == 0, 1], label='Ground Truth', alpha=0.5, s=10, c='blue')
plt.scatter(X_embedded[labels == 1, 0], X_embedded[labels == 1, 1], label='Predicted', alpha=0.5, s=10, c='orange')
plt.title('t-SNE: Ground Truth vs Predicted Residual Dynamics (500 episodes)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
