import os
import random
import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from unified import (
    ConditionalUnet1D,
    PushTStateDataset,
    unnormalize_data,
    get_data_stats
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def evaluate_and_plot():
    # Paths
    dataset_path = "data/drone_val_dataset.zarr"
    train_dataset_path = "data/drone_train_dataset.zarr"
    model_path = "coolest_model.pth"

    # Parameters
    pred_horizon, obs_horizon, action_horizon = 16, 2, 8
    obs_dim, action_dim = 14, 3
    num_diffusion_iters = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check files
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print(f"Error: Required file not found. Check paths for dataset and model.")
        return

    val_zarr = zarr.open(dataset_path, 'r')
    all_obs_data = val_zarr['data']['state'][:]
    all_action_data = val_zarr['data']['action'][:]
    episode_ends = val_zarr['meta']['episode_ends'][:]

    train_zarr = zarr.open(train_dataset_path, 'r')
    stats = {
        'obs': get_data_stats(train_zarr['data']['state'][:]),
        'action': get_data_stats(train_zarr['data']['action'][:])
    }

    # Random episode
    episode_idx = random.randint(0, len(episode_ends) - 1)
    start_idx = 0
    if episode_idx > 0:
        start_idx = episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]
    print(f"Selected random episode #{episode_idx} (from timestep {start_idx} to {end_idx}).")

    episode_obs_gt = all_obs_data[start_idx:end_idx]
    episode_action_gt = all_action_data[start_idx:end_idx]
    action_gt_for_comparison = episode_action_gt[:pred_horizon]

    # Load model
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)
    noise_pred_net.load_state_dict(torch.load(model_path, map_location=device))
    noise_pred_net.eval()
    print("Model loaded successfully.")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Prepare input
    obs_cond_data = episode_obs_gt[:obs_horizon]
    nobs_cond_data = (obs_cond_data - stats['obs']['min']) / (stats['obs']['max'] - stats['obs']['min'])
    nobs_cond_data = nobs_cond_data * 2 - 1
    nobs_cond_tensor = torch.from_numpy(nobs_cond_data).to(torch.float32).unsqueeze(0).to(device)
    global_cond = nobs_cond_tensor.flatten(start_dim=1)
    noisy_actions = torch.randn((1, pred_horizon, action_dim), device=device)
    noise_scheduler.set_timesteps(num_diffusion_iters)

    # Denoising loop
    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc="Generating Prediction"):
            noise_pred = noise_pred_net(noisy_actions, t, global_cond)
            noisy_actions = noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_actions
            ).prev_sample

    # Un-normalize and plot
    naction_gen = noisy_actions.cpu().numpy().squeeze()
    action_gen = unnormalize_data(naction_gen, stats['action'])

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'Model Prediction vs. Ground Truth for Episode #{episode_idx}', fontsize=16)
    labels = ['H Component X (N)', 'H Component Y (N)', 'H Component Z (N)']
    timesteps = np.arange(pred_horizon)
    for i in range(3):
        axs[i].plot(timesteps, action_gt_for_comparison[:, i], label='Ground Truth', color='blue', linestyle='--', marker='.', markersize=8)
        axs[i].plot(timesteps, action_gen[:, i], label='Model Prediction', color='red', marker='x', markersize=6)
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[2].set_xlabel('Time Step in Prediction Horizon')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"evaluation_episode_{episode_idx}.png")
    print(f"Plot saved to evaluation_episode_{episode_idx}.png")
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()