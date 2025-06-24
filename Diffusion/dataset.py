# dataset.py

import os
import pickle
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class DroneDynamicsSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading drone dynamics data for a sequence model.

    This class is designed to work with pre-divided trajectory segments to
    ensure a clean separation between training and validation data, preventing
    any data leakage.
    """
    def __init__(self,
                 dataset_path: str,
                 trajectory_segments: List[Tuple[int, int]],
                 obs_horizon: int,
                 pred_horizon: int,
                 scalers: Dict[str, StandardScaler]):
        """
        Args:
            dataset_path (str): Path to the .npz dataset file.
            trajectory_segments (List[Tuple[int, int]]): A list of (start, end)
                indices defining which trajectories this dataset should use.
            obs_horizon (int): Number of past timesteps for the condition.
            pred_horizon (int): Number of future timesteps to predict.
            scalers (Dict): A dictionary of pre-fitted StandardScaler objects.
        """
        super().__init__()

        # --- 1. Set parameters ---
        self.trajectory_segments = trajectory_segments
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sequence_length = self.obs_horizon + self.pred_horizon - 1

        # --- 2. Load the raw data streams ---
        raw_dataset = np.load(dataset_path)
        self.state_stream = np.concatenate([
            raw_dataset['position'], raw_dataset['velocity'], raw_dataset['orientation']
        ], axis=1).astype(np.float32)
        self.action_stream = raw_dataset['action_t_minus_1'].astype(np.float32)
        self.target_stream = raw_dataset['residual_dynamics'].astype(np.float32)

        # --- 3. Create valid sequence indices from the provided segments ---
        self.indices = self._create_indices_from_segments()

        # --- 4. Store the provided, pre-fitted scalers ---
        self.state_scaler = scalers['state']
        self.action_scaler = scalers['action']
        self.target_scaler = scalers['target']

    def _create_indices_from_segments(self) -> list:
        """Generates a list of valid sequence start indices from trajectory segments."""
        indices = []
        for start_idx, end_idx in self.trajectory_segments:
            segment_len = end_idx - start_idx
            if segment_len >= self.sequence_length:
                last_possible_start = end_idx - self.sequence_length
                indices.extend(range(start_idx, last_possible_start + 1))
        return indices

    def __len__(self) -> int:
        """Returns the total number of valid sequences."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """Retrieves a single training sample."""
        start_idx = self.indices[idx]
        
        state_chunk = self.state_stream[start_idx : start_idx + self.obs_horizon]
        action_chunk = self.action_stream[start_idx : start_idx + self.obs_horizon]

        target_chunk_start = start_idx + self.obs_horizon - 1
        target_chunk_end = target_chunk_start + self.pred_horizon
        target_chunk = self.target_stream[target_chunk_start:target_chunk_end]

        norm_state = self.state_scaler.transform(state_chunk)
        norm_action = self.action_scaler.transform(action_chunk)
        norm_target = self.target_scaler.transform(target_chunk)

        condition = np.concatenate([norm_state.flatten(), norm_action.flatten()], axis=-1)
        target = norm_target.flatten()

        return {
            'condition': torch.from_numpy(condition).float(),
            'target': torch.from_numpy(target).float()
        }


def create_train_val_dataloaders(
    dataset_path: str,
    obs_horizon: int,
    pred_horizon: int,
    batch_size: int,
    val_split_ratio: float = 0.2,
    scaler_dir: str = 'scalers'
) -> Tuple[DataLoader, DataLoader, Dict[str, StandardScaler]]:
    """
    Finds all trajectories in a dataset, splits them into train/validation sets,
    handles scaler fitting and saving, and creates the final DataLoaders.

    Returns:
        A tuple containing (train_dataloader, validation_dataloader, scalers).
    """
    print("--- Preparing Train/Validation DataLoaders ---")
    os.makedirs(scaler_dir, exist_ok=True)
    
    # 1. Find all trajectory segments from the .npz file
    target_stream = np.load(dataset_path)['residual_dynamics']
    nan_rows = np.where(np.isnan(target_stream).any(axis=1))[0]
    all_segments = []
    start_of_ep = 0
    for end_of_ep in nan_rows:
        all_segments.append((start_of_ep, end_of_ep))
        start_of_ep = end_of_ep + 1
    all_segments.append((start_of_ep, len(target_stream)))
    print(f"Found {len(all_segments)} total trajectories in the .npz file.")

    # 2. Shuffle and split trajectories for train/val sets
    random.seed(42)
    random.shuffle(all_segments)
    split_idx = int(len(all_segments) * (1 - val_split_ratio))
    train_segments = all_segments[:split_idx]
    val_segments = all_segments[split_idx:]
    print(f"Splitting into {len(train_segments)} train and {len(val_segments)} validation trajectories.")

    # 3. Fit scalers ONLY on the training data segments
    print("Fitting scalers on training data...")
    raw_dataset = np.load(dataset_path)
    state_stream = np.concatenate([raw_dataset['position'], raw_dataset['velocity'], raw_dataset['orientation']], axis=1)
    action_stream = raw_dataset['action_t_minus_1']
    target_stream = raw_dataset['residual_dynamics']

    train_indices = np.concatenate([np.arange(start, end) for start, end in train_segments])
    
    state_scaler = StandardScaler().fit(state_stream[train_indices])
    action_scaler = StandardScaler().fit(action_stream[train_indices])
    target_scaler = StandardScaler().fit(target_stream[train_indices])

    scalers = {'state': state_scaler, 'action': action_scaler, 'target': target_scaler}
    
    # Save the fitted scalers for future use (e.g., during inference)
    for name, scaler in scalers.items():
        with open(os.path.join(scaler_dir, f'{name}_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    print(f"Fitted and saved scalers to '{scaler_dir}/'")

    # 4. Create Dataset and DataLoader objects
    train_dataset = DroneDynamicsSequenceDataset(
        dataset_path=dataset_path, trajectory_segments=train_segments,
        obs_horizon=obs_horizon, pred_horizon=pred_horizon, scalers=scalers
    )
    val_dataset = DroneDynamicsSequenceDataset(
        dataset_path=dataset_path, trajectory_segments=val_segments,
        obs_horizon=obs_horizon, pred_horizon=pred_horizon, scalers=scalers
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    print(f"Created training dataset with {len(train_dataset)} samples.")
    print(f"Created validation dataset with {len(val_dataset)} samples.")
    print("--- Data preparation complete ---")
    
    return train_dataloader, val_dataloader, scalers


# =================================================================
# Example Usage: This block demonstrates how to use the functions above
# =================================================================
if __name__ == '__main__':
    # --- Parameters ---
    DATA_PATH = 'data/snowy-lake-170_dataset_aligned.npz'
    OBS_HORIZON = 4
    PRED_HORIZON = 8
    VAL_SPLIT = 0.2
    BATCH_SIZE = 128

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Test data file not found at '{DATA_PATH}'")
    else:
        # Create train and validation dataloaders and get the scalers
        train_loader, val_loader, fitted_scalers = create_train_val_dataloaders(
            dataset_path=DATA_PATH,
            obs_horizon=OBS_HORIZON,
            pred_horizon=PRED_HORIZON,
            batch_size=BATCH_SIZE,
            val_split_ratio=VAL_SPLIT
        )

        # --- Test the outputs ---
        print(f"\nTotal training batches: {len(train_loader)}")
        print(f"Total validation batches: {len(val_loader)}")

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        print("\n--- DataLoader Batch Test ---")
        print(f"Train batch condition shape: {train_batch['condition'].shape}")
        print(f"Train batch target shape:   {train_batch['target'].shape}")
        
        # Verify dimensions
        expected_cond_dim = OBS_HORIZON * (10 + 4)
        expected_target_dim = PRED_HORIZON * 3

        assert train_batch['condition'].shape == (BATCH_SIZE, expected_cond_dim)
        assert train_batch['target'].shape == (BATCH_SIZE, expected_target_dim)
        print("\nDimension check PASSED.")