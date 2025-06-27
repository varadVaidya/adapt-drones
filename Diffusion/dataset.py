import os
import pickle
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# =================================================================
# CLASS: DroneDynamicsSequenceDataset (NO CHANGES NEEDED)
# This class is already well-designed to handle our use case.
# It correctly loads data from the `dataset_path` it is given.
# =================================================================
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

# =================================================================
# HELPER FUNCTION (NEW)
# =================================================================
def _find_trajectory_segments(dataset_path: str) -> List[Tuple[int, int]]:
    """Finds all trajectory segments in a given .npz file."""
    target_stream = np.load(dataset_path)['residual_dynamics']
    nan_rows = np.where(np.isnan(target_stream).any(axis=1))[0]
    
    all_segments = []
    start_of_ep = 0
    for end_of_ep in nan_rows:
        # Only add segment if it's not empty
        if end_of_ep > start_of_ep:
            all_segments.append((start_of_ep, end_of_ep))
        start_of_ep = end_of_ep + 1
    
    # Add the final segment after the last NaN
    if start_of_ep < len(target_stream):
        all_segments.append((start_of_ep, len(target_stream)))
        
    return all_segments


# =================================================================
# DATALOADER CREATION FUNCTION (MODIFIED)
# =================================================================
def create_train_val_dataloaders(
    train_dataset_path: str,    # CHANGED: Path to training data
    val_dataset_path: str,      # CHANGED: Path to validation data
    obs_horizon: int,
    pred_horizon: int,
    batch_size: int,
    scaler_dir: str = 'scalers' # REMOVED: val_split_ratio is no longer needed
) -> Tuple[DataLoader, DataLoader, Dict[str, StandardScaler]]:
    """
    Creates train/validation dataloaders from separate data files and
    handles scaler fitting and saving.

    The key logic is that scalers are FIT ONLY on the training data and
    then APPLIED to both the training and validation data.

    Returns:
        A tuple containing (train_dataloader, validation_dataloader, scalers).
    """
    print("--- Preparing Train/Validation DataLoaders from separate files ---")
    os.makedirs(scaler_dir, exist_ok=True)
    
    # 1. Find all trajectory segments from each file
    train_segments = _find_trajectory_segments(train_dataset_path)
    val_segments = _find_trajectory_segments(val_dataset_path)
    print(f"Found {len(train_segments)} trajectories in the training file.")
    print(f"Found {len(val_segments)} trajectories in the validation file.")

    # 2. Fit scalers ONLY on the training data
    print("Fitting scalers on training data...")
    raw_train_dataset = np.load(train_dataset_path)
    # Important: Use the entire training dataset for fitting the scalers
    state_stream_train = np.concatenate([
        raw_train_dataset['position'], raw_train_dataset['velocity'], raw_train_dataset['orientation']
    ], axis=1)
    action_stream_train = raw_train_dataset['action_t_minus_1']
    target_stream_train = raw_train_dataset['residual_dynamics']
    
    # Fit scalers on non-NaN data from the training set
    state_scaler = StandardScaler().fit(state_stream_train[~np.isnan(state_stream_train).any(axis=1)])
    action_scaler = StandardScaler().fit(action_stream_train[~np.isnan(action_stream_train).any(axis=1)])
    target_scaler = StandardScaler().fit(target_stream_train[~np.isnan(target_stream_train).any(axis=1)])

    scalers = {'state': state_scaler, 'action': action_scaler, 'target': target_scaler}
    
    # Save the fitted scalers for future use
    for name, scaler in scalers.items():
        with open(os.path.join(scaler_dir, f'{name}_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    print(f"Fitted and saved scalers to '{scaler_dir}/'")

    # 3. Create Dataset and DataLoader objects for training
    train_dataset = DroneDynamicsSequenceDataset(
        dataset_path=train_dataset_path,
        trajectory_segments=train_segments,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        scalers=scalers  # Use the newly fitted scalers
    )
    
    # 4. Create Dataset and DataLoader objects for validation
    val_dataset = DroneDynamicsSequenceDataset(
        dataset_path=val_dataset_path,
        trajectory_segments=val_segments,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        scalers=scalers  # Use the SAME scalers fitted on the training data
    )
    
    # 5. Create the DataLoader instances
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
# Example Usage: This block demonstrates how to use the new function
# =================================================================
if __name__ == '__main__':
    # --- Parameters ---
    # Define paths for your separate train and validation files
    TRAIN_DATA_PATH = 'data/slow_tcn_train_diff_dataset.npz'
    VAL_DATA_PATH = 'data/slow_tcn_eval_diff_dataset.npz'

    OBS_HORIZON = 4
    PRED_HORIZON = 8
    BATCH_SIZE = 128
    
    # --- Create dummy data for demonstration if files don't exist ---
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VAL_DATA_PATH):
        print("WARNING: Could not find train/val data files. Creating dummy data for demonstration.")
        print("For your actual use, please create these files:\n - {}\n - {}".format(TRAIN_DATA_PATH, VAL_DATA_PATH))
        
        os.makedirs('data', exist_ok=True)
        # Create a dummy dataset and split it to simulate having two files
        dummy_data = {
            'position': np.random.randn(2000, 3), 'velocity': np.random.randn(2000, 3),
            'orientation': np.random.randn(2000, 4), 'action_t_minus_1': np.random.randn(2000, 4),
            'residual_dynamics': np.random.randn(2000, 3)
        }
        # Insert NaNs to simulate trajectory breaks
        dummy_data['residual_dynamics'][499, :] = np.nan
        dummy_data['residual_dynamics'][999, :] = np.nan
        dummy_data['residual_dynamics'][1499, :] = np.nan
        
        # Split into "train" and "validation" files
        np.savez(TRAIN_DATA_PATH, **{k: v[:1500] for k, v in dummy_data.items()})
        np.savez(VAL_DATA_PATH, **{k: v[1500:] for k, v in dummy_data.items()})
        print("Dummy files created.\n")

    # Create train and validation dataloaders and get the scalers
    train_loader, val_loader, fitted_scalers = create_train_val_dataloaders(
        train_dataset_path=TRAIN_DATA_PATH,
        val_dataset_path=VAL_DATA_PATH,
        obs_horizon=OBS_HORIZON,
        pred_horizon=PRED_HORIZON,
        batch_size=BATCH_SIZE
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
    expected_cond_dim = OBS_HORIZON * (10 + 4) # state (3+3+4=10) + action (4)
    expected_target_dim = PRED_HORIZON * 3   # residual_dynamics (3)

    # Note: The last batch might be smaller if the dataset size is not a multiple of BATCH_SIZE
    actual_batch_size = train_batch['condition'].shape[0]
    assert train_batch['condition'].shape == (actual_batch_size, expected_cond_dim)
    assert train_batch['target'].shape == (actual_batch_size, expected_target_dim)
    print("\nDimension check PASSED.")