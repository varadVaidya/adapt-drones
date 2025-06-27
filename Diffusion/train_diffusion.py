# train.py

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

# --- Local Imports ---
# Assumes dataset.py and unet.py are in the same directory
from dataset import create_train_val_dataloaders
from unet import ConditionalUnet1D

# --- Configuration & Hyperparameters ---
# Data parameters (MODIFIED)
# Provide separate paths for your training and validation data files.
TRAIN_DATA_PATH = 'data/slow_tcn_train_diff_dataset.npz'
VAL_DATA_PATH = 'data/slow_tcn_eval_diff_dataset.npz'

SCALER_DIR = 'scalers/'
OBS_HORIZON = 4
PRED_HORIZON = 8
# VAL_SPLIT_RATIO is no longer needed as data is pre-split.

# Model dimensions
STATE_DIM = 10
ACTION_DIM = 4
RESIDUAL_DIM = 3
GLOBAL_COND_DIM = OBS_HORIZON * (STATE_DIM + ACTION_DIM)

# Training parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 256 # Adjust based on your GPU memory
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-6
NUM_DIFFUSION_TIMESTEPS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Check if data files exist ---
if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VAL_DATA_PATH):
    print("\nERROR: Training or validation data file not found.")
    print(f"Please make sure the following files exist:")
    print(f" - Training data: '{TRAIN_DATA_PATH}'")
    print(f" - Validation data: '{VAL_DATA_PATH}'")
    exit() # Exit the script if data is missing

# --- Data Loading with separate Train/Val files (MODIFIED) ---
train_dataloader, val_dataloader, scalers = create_train_val_dataloaders(
    train_dataset_path=TRAIN_DATA_PATH,  # Use the new argument
    val_dataset_path=VAL_DATA_PATH,      # Use the new argument
    obs_horizon=OBS_HORIZON,
    pred_horizon=PRED_HORIZON,
    batch_size=BATCH_SIZE,
    scaler_dir=SCALER_DIR
    # val_split_ratio is removed
)


# --- Model, Optimizer, and Scheduler Setup (NO CHANGES NEEDED) ---
print("Setting up model and optimizer...")
# The U-Net that predicts noise
noise_pred_net = ConditionalUnet1D(
    input_dim=RESIDUAL_DIM,
    global_cond_dim=GLOBAL_COND_DIM
).to(device)

# The noise scheduler from Hugging Face's `diffusers` library
noise_scheduler = DDPMScheduler(
    num_train_timesteps=NUM_DIFFUSION_TIMESTEPS,
    beta_schedule='linear',
    prediction_type='epsilon'
)

# Exponential Moving Average for improved stability and performance
ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

# Standard AdamW optimizer
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Cosine learning rate scheduler with warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * NUM_EPOCHS
)


# --- Training & Validation Loop (NO CHANGES NEEDED) ---

print("Starting training...")
best_val_loss = float('inf')
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


with tqdm(range(NUM_EPOCHS), desc='Epoch') as tglobal:
    for epoch_idx in tglobal:
        # TRAINING PHASE
        noise_pred_net.train()
        epoch_train_loss = []
        with tqdm(train_dataloader, desc='Train Batch', leave=False) as tepoch:
            for batch in tepoch:
                # 1. Get data and move to device
                condition_vector = batch['condition'].to(device)
                target_vector = batch['target'].to(device)
                B = condition_vector.shape[0]

                # 2. Reshape target for the U-Net
                clean_sequence = target_vector.view(-1, PRED_HORIZON, RESIDUAL_DIM)

                # 3. Diffusion process
                noise = torch.randn(clean_sequence.shape, device=device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                noisy_sequence = noise_scheduler.add_noise(clean_sequence, noise, timesteps)
                
                # 4. Predict noise
                noise_pred = noise_pred_net(noisy_sequence, timesteps, global_cond=condition_vector)
                
                # 5. Calculate loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # 6. Optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(noise_pred_net.parameters())

                epoch_train_loss.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
        
        avg_train_loss = np.mean(epoch_train_loss)

        # VALIDATION PHASE
        noise_pred_net.eval()
        epoch_val_loss = []
        with torch.no_grad():
            with tqdm(val_dataloader, desc='Validation Batch', leave=False) as tval:
                for batch in tval:
                    condition_vector = batch['condition'].to(device)
                    target_vector = batch['target'].to(device)
                    B = condition_vector.shape[0]
                    clean_sequence = target_vector.view(-1, PRED_HORIZON, RESIDUAL_DIM)

                    noise = torch.randn(clean_sequence.shape, device=device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                    noisy_sequence = noise_scheduler.add_noise(clean_sequence, noise, timesteps)
                    
                    noise_pred = noise_pred_net(noisy_sequence, timesteps, global_cond=condition_vector)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    epoch_val_loss.append(loss.item())

        avg_val_loss = np.mean(epoch_val_loss)
        tglobal.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")

        # --- Save the best model based on validation loss ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Copy the EMA parameters to the model before saving
            ema.copy_to(noise_pred_net.parameters())
            torch.save(noise_pred_net.state_dict(), os.path.join(MODELS_DIR, "diffusion_unet_ema_best.pt"))
            print(f"Epoch {epoch_idx+1}: New best model saved with val_loss: {best_val_loss:.4f}")

print("\nTraining complete.")
print(f"Best validation loss achieved: {best_val_loss:.4f}")
print(f"Best model saved to {os.path.join(MODELS_DIR, 'diffusion_unet_ema_best.pt')}")