import os
import math
from typing import Union, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt

# =================================================================================
#
#                                   PART 1: DATASET (MODIFIED)
#
# =================================================================================

# Helper functions (create_sample_indices, sample_sequence, etc.) remain the same
def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class PushTStateDataset(torch.utils.data.Dataset):
    """
    Dataset class that loads data from a single Zarr file.
    It requires statistics for normalization to be passed, which is crucial
    for creating a validation set that uses the training set's statistics.
    """
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 stats: Dict[str, Dict[str, np.ndarray]]):

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.stats = stats

        # Read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')
        raw_train_data = {
            'action': dataset_root['data']['action'][:],
            'obs': dataset_root['data']['state'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # Normalize data
        self.normalized_train_data = dict()
        for key, data in raw_train_data.items():
            self.normalized_train_data[key] = normalize_data(data, self.stats[key])

        # Compute start and end of each state-action sequence
        self.indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        nsample = {k: torch.from_numpy(v).to(torch.float32) for k,v in nsample.items()}
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
        return nsample

# =================================================================================
#
#                                   PART 2: NETWORK (Unchanged)
#
# =================================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        print("Number of parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int], global_cond=None):
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


# =================================================================================
#
#                         PART 3: TRAINING & EVALUATION (MODIFIED)
#
# =================================================================================
def main():
    # =========== 1. CONFIGURATION ===========
    # Assume you have two files generated from your data collection script
    train_dataset_path = "data/drone_train_dataset.zarr"
    val_dataset_path = "data/drone_val_dataset.zarr"

    pred_horizon, obs_horizon, action_horizon = 16, 2, 8
    obs_dim, action_dim = 14, 3
    num_epochs, batch_size, learning_rate = 100, 256, 1e-4
    num_diffusion_iters = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =========== 2. DATA PREPARATION ===========
    # Check if dataset files exist
    if not os.path.exists(train_dataset_path) or not os.path.exists(val_dataset_path):
        print(f"Dataset files not found. Make sure '{train_dataset_path}' and '{val_dataset_path}' exist.")
        return

    # First, load the raw training data to compute normalization stats
    train_zarr = zarr.open(train_dataset_path, 'r')
    stats = {
        'obs': get_data_stats(train_zarr['data']['state'][:]),
        'action': get_data_stats(train_zarr['data']['action'][:])
    }

    # Create train and validation datasets
    train_dataset = PushTStateDataset(
        dataset_path=train_dataset_path, pred_horizon=pred_horizon,
        obs_horizon=obs_horizon, action_horizon=action_horizon, stats=stats
    )
    # CRITICAL: Pass the training stats to the validation set
    val_dataset = PushTStateDataset(
        dataset_path=val_dataset_path, pred_horizon=pred_horizon,
        obs_horizon=obs_horizon, action_horizon=action_horizon, stats=stats
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4,
        shuffle=True, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4,
        shuffle=False, pin_memory=True, persistent_workers=True
    )

    # =========== 3. MODEL, SCHEDULER & OPTIMIZER ===========
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon
    ).to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon'
    )
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(), lr=learning_rate, weight_decay=1e-6
    )

    # =========== 4. TRAINING & VALIDATION LOOP ===========
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- Training ---
        noise_pred_net.train()
        train_epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False) as t:
            for batch in t:
                nobs, naction = batch['obs'].to(device), batch['action'].to(device)
                global_cond = nobs.flatten(start_dim=1)
                noise = torch.randn_like(naction)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],), device=device).long()
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond)
                loss = F.mse_loss(noise_pred, noise)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())
        avg_train_loss = train_epoch_loss / len(train_loader)

        # --- Validation ---
        noise_pred_net.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False) as t:
                for batch in t:
                    nobs, naction = batch['obs'].to(device), batch['action'].to(device)
                    global_cond = nobs.flatten(start_dim=1)
                    noise = torch.randn_like(naction)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],), device=device).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond)
                    loss = F.mse_loss(noise_pred, noise)
                    val_epoch_loss += loss.item()
                    t.set_postfix(loss=loss.item())
        avg_val_loss = val_epoch_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(noise_pred_net.state_dict(), 'coolest_model.pth')
            print(f"  New best model saved with val_loss: {best_val_loss:.4f}")

    print("Training finished!")
    
    # =========== 5. INFERENCE & VISUALIZATION ON VALIDATION DATA ===========
    print("Loading best model and running inference on validation data...")
    noise_pred_net.load_state_dict(torch.load('coolest_model.pth'))
    noise_pred_net.eval()
    
    val_batch = next(iter(val_loader))
    nobs, naction_ground_truth = val_batch['obs'].to(device), val_batch['action']
    
    nobs_cond = nobs[0:1]
    global_cond = nobs_cond.flatten(start_dim=1)
    
    noisy_actions = torch.randn((1, pred_horizon, action_dim), device=device)
    noise_scheduler.set_timesteps(num_diffusion_iters)

    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps, desc="Denoising"):
            noise_pred = noise_pred_net(noisy_actions, t, global_cond)
            noisy_actions = noise_scheduler.step(model_output=noise_pred, timestep=t, sample=noisy_actions).prev_sample
            
    naction_gen = noisy_actions.cpu().numpy().squeeze()
    action_gen = unnormalize_data(naction_gen, stats['action'])
    
    naction_gt = naction_ground_truth[0].numpy()
    action_gt = unnormalize_data(naction_gt, stats['action'])
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Generated vs. Ground Truth (Validation Set)', fontsize=16)
    labels = ['X Component', 'Y Component', 'Z Component']
    for i in range(3):
        axs[i].plot(action_gt[:, i], label='Ground Truth', color='blue', linestyle='--')
        axs[i].plot(action_gen[:, i], label='Generated', color='red')
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid(True)
    axs[2].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()