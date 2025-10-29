"""
Data loader for precipitation forecasting
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class PrecipitationDataset(Dataset):
    """
    Dataset for precipitation time series forecasting.

    Args:
        features: Input features (T, D)
        targets: Target precipitation values (T,)
        input_len: Length of input sequence
        output_len: Length of output sequence
        stride: Stride for sliding window (default: 1)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        input_len: int,
        output_len: int,
        stride: int = 1
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride

        # Calculate valid indices
        self.indices = []
        for i in range(0, len(features) - input_len - output_len + 1, stride):
            self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            x: Input features (input_len, D)
            y_history: Historical target values (input_len,)
            y_target: Future target values (output_len,)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.input_len

        x = self.features[start_idx:end_idx]
        y_history = self.targets[start_idx:end_idx]
        y_target = self.targets[end_idx:end_idx + self.output_len]

        return x, y_history, y_target


def create_dataloaders(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    input_len: int,
    output_len: int,
    batch_size: int = 64,
    num_workers: int = 4,
    stride: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_features: Training features
        train_targets: Training targets
        val_features: Validation features
        val_targets: Validation targets
        test_features: Test features
        test_targets: Test targets
        input_len: Length of input sequence
        output_len: Length of output sequence
        batch_size: Batch size
        num_workers: Number of workers for data loading
        stride: Stride for sliding window

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = PrecipitationDataset(
        train_features,
        train_targets,
        input_len,
        output_len,
        stride
    )

    val_dataset = PrecipitationDataset(
        val_features,
        val_targets,
        input_len,
        output_len,
        stride
    )

    test_dataset = PrecipitationDataset(
        test_features,
        test_targets,
        input_len,
        output_len,
        stride
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def calculate_zero_inflation_ratio(targets: np.ndarray) -> float:
    """
    Calculate the zero inflation ratio in the target variable.

    Args:
        targets: Target precipitation values

    Returns:
        Zero inflation ratio (percentage of zeros)
    """
    zero_ratio = np.sum(targets == 0) / len(targets)
    return zero_ratio
