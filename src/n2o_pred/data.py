import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequenceDataset_ObsStep(Dataset):
    """
    ObsStep dataset for N2O prediction

    Features:
        - Variable length sequences
        - Contains time interval features
    """

    def __init__(self, sequences: list[dict]):
        """
        Args:
            sequences: Processed sequence list
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Static features (numerical + categorical encoded)
        static_feat = np.concatenate([seq['static_numeric'], seq['static_categorical_encoded']])

        # Dynamic features (numerical + fertilization features + fertilization categorical encoded)
        # Dynamic numerical: Temp, Prec, ST, WFPS (4)
        # Fertilization numerical: Split N amount, ferdur, sowdur, time_delta (4)
        # Fertilization categorical: fertilization_class + appl_class (one-hot encoded)
        dynamic_feat = np.concatenate(
            [
                seq['dynamic_numeric'],  # (seq_len, 4)
                seq['fertilization_numeric'],  # (seq_len, 4)
                seq['fertilization_categorical_encoded'][
                    'fertilization_class'
                ],  # (seq_len, n_fert_class)
                seq['fertilization_categorical_encoded']['appl_class'],  # (seq_len, n_appl_class)
            ],
            axis=1,
        )

        return {
            'seq_id': seq['seq_id'],
            'static_features': torch.FloatTensor(static_feat),
            'dynamic_features': torch.FloatTensor(dynamic_feat),
            'target': torch.FloatTensor(seq['target']),
            'target_original': torch.FloatTensor(seq['target_original']),
            'seq_length': seq['seq_length'],
        }


def collate_fn_obs_step(batch):
    """
    ObsStep's collate function handles variable-length sequences.

    Args:
        batch: list of dict from Dataset

    Returns:
        dict with batched and padded tensors
    """
    # Get the maximum sequence length in the batch
    max_len = max(item['seq_length'] for item in batch)
    batch_size = len(batch)

    # Get the feature dimensions
    static_dim = batch[0]['static_features'].shape[0]
    dynamic_dim = batch[0]['dynamic_features'].shape[1]

    # Initialize padded tensors
    static_features = torch.zeros(batch_size, static_dim)
    dynamic_features = torch.zeros(batch_size, max_len, dynamic_dim)
    targets = torch.zeros(batch_size, max_len)
    targets_original = torch.zeros(batch_size, max_len)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    seq_ids = []

    for i, item in enumerate(batch):
        seq_len = item['seq_length']

        static_features[i] = item['static_features']
        dynamic_features[i, :seq_len] = item['dynamic_features']
        targets[i, :seq_len] = item['target']
        targets_original[i, :seq_len] = item['target_original']
        lengths[i] = seq_len
        masks[i, :seq_len] = True

        seq_ids.append(item['seq_id'])

    return {
        'seq_ids': seq_ids,
        'static_features': static_features,
        'dynamic_features': dynamic_features,
        'targets': targets,
        'targets_original': targets_original,
        'lengths': lengths,
        'masks': masks,
    }


class SequenceDataset_DailyStep(Dataset):
    """
    DailyStep dataset for N2O prediction

    Features:
        - Fixed daily step length
        - Contains observed mask
        - Does not contain time interval features
    """

    def __init__(self, sequences: list[dict]):
        """
        Args:
            sequences: Processed sequence list
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Static features (numerical + categorical encoded)
        static_feat = np.concatenate([seq['static_numeric'], seq['static_categorical_encoded']])

        # Dynamic features (numerical + fertilization features + fertilization categorical encoded)
        # Dynamic numerical: Temp, Prec, ST, WFPS (4)
        # Fertilization numerical: Split N amount (1)
        # Fertilization categorical: fertilization_class + appl_class (one-hot encoded)
        dynamic_feat = np.concatenate(
            [
                seq['dynamic_numeric'],  # (seq_len, 4)
                seq['fertilization_numeric'],  # (seq_len, 1)
                seq['fertilization_categorical_encoded'][
                    'fertilization_class'
                ],  # (seq_len, n_fert_class)
                seq['fertilization_categorical_encoded']['appl_class'],  # (seq_len, n_appl_class)
            ],
            axis=1,
        )

        return {
            'seq_id': seq['seq_id'],
            'static_features': torch.FloatTensor(static_feat),
            'dynamic_features': torch.FloatTensor(dynamic_feat),
            'target': torch.FloatTensor(seq['target']),
            'target_original': torch.FloatTensor(seq['target_original']),
            'observed_mask': torch.BoolTensor(seq['observed_mask']),
            'seq_length': seq['seq_length'],
            'min_day': seq['min_day'],
            'max_day': seq['max_day'],
        }


def collate_fn_daily_step(batch):
    """
    DailyStep's collate function handles variable-length sequences.

    Args:
        batch: list of dict from Dataset

    Returns:
        dict with batched and padded tensors
    """
    # Get the maximum sequence length in the batch
    max_len = max(item['seq_length'] for item in batch)
    batch_size = len(batch)

    # Get the feature dimensions
    static_dim = batch[0]['static_features'].shape[0]
    dynamic_dim = batch[0]['dynamic_features'].shape[1]

    # Initialize padded tensors
    static_features = torch.zeros(batch_size, static_dim)
    dynamic_features = torch.zeros(batch_size, max_len, dynamic_dim)
    targets = torch.zeros(batch_size, max_len)
    targets_original = torch.zeros(batch_size, max_len)
    observed_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    padding_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    seq_ids = []
    min_days = []
    max_days = []

    for i, item in enumerate(batch):
        seq_len = item['seq_length']

        static_features[i] = item['static_features']
        dynamic_features[i, :seq_len] = item['dynamic_features']
        targets[i, :seq_len] = item['target']
        targets_original[i, :seq_len] = item['target_original']
        observed_masks[i, :seq_len] = item['observed_mask']
        lengths[i] = seq_len
        padding_masks[i, :seq_len] = True

        seq_ids.append(item['seq_id'])
        min_days.append(item['min_day'])
        max_days.append(item['max_day'])

    return {
        'seq_ids': seq_ids,
        'static_features': static_features,
        'dynamic_features': dynamic_features,
        'targets': targets,
        'targets_original': targets_original,
        'observed_masks': observed_masks,  # Observed mask: only calculate loss at real observation points
        'lengths': lengths,
        'padding_masks': padding_masks,  # Padding mask: used for RNN
        'min_days': min_days,
        'max_days': max_days,
    }


def create_dataloader(
    sequences: list[dict],
    dataset_type: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader

    Args:
        sequences: Sequence list
        dataset_type: 'obs_step' or 'daily_step'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        DataLoader
    """
    if dataset_type == 'obs_step':
        dataset = SequenceDataset_ObsStep(sequences)
        collate_fn = collate_fn_obs_step
    elif dataset_type == 'daily_step':
        dataset = SequenceDataset_DailyStep(sequences)
        collate_fn = collate_fn_daily_step
    else:
        raise ValueError(f'Unknown dataset_type: {dataset_type}')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


if __name__ == '__main__':
    # Test dataset and dataloader
    import pickle
    from pathlib import Path

    print('Testing ObsStep Dataset and DataLoader...')
    data_dir = Path('../datasets/processed')

    # Load a small portion of data for testing
    with open(data_dir / 'sequences_obs_step_train_processed.pkl', 'rb') as f:
        train_obs = pickle.load(f)[:10]  # Only take the first 10 sequences for testing

    # Create dataloader
    dataloader_obs = create_dataloader(
        train_obs, dataset_type='obs_step', batch_size=4, shuffle=False
    )

    # Test a batch
    batch = next(iter(dataloader_obs))
    print(f'Batch keys: {batch.keys()}')
    print(f'Static features shape: {batch["static_features"].shape}')
    print(f'Dynamic features shape: {batch["dynamic_features"].shape}')
    print(f'Targets shape: {batch["targets"].shape}')
    print(f'Lengths: {batch["lengths"]}')
    print(f'Masks shape: {batch["masks"].shape}')
    print('ObsStep test passed!\n')

    print('Testing DailyStep Dataset and DataLoader...')
    with open(data_dir / 'sequences_daily_step_train_processed.pkl', 'rb') as f:
        train_daily = pickle.load(f)[:10]

    dataloader_daily = create_dataloader(
        train_daily, dataset_type='daily_step', batch_size=4, shuffle=False
    )

    batch_daily = next(iter(dataloader_daily))
    print(f'Batch keys: {batch_daily.keys()}')
    print(f'Static features shape: {batch_daily["static_features"].shape}')
    print(f'Dynamic features shape: {batch_daily["dynamic_features"].shape}')
    print(f'Targets shape: {batch_daily["targets"].shape}')
    print(f'Observed masks shape: {batch_daily["observed_masks"].shape}')
    print(f'Padding masks shape: {batch_daily["padding_masks"].shape}')
    print(f'Lengths: {batch_daily["lengths"]}')
    print('DailyStep test passed!')
