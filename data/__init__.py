"""
Data module for loading and preprocessing precipitation data
"""

from .data_loader import PrecipitationDataset, create_dataloaders
from .preprocessing import preprocess_precipitation_data, generate_synthetic_data

__all__ = [
    'PrecipitationDataset',
    'create_dataloaders',
    'preprocess_precipitation_data',
    'generate_synthetic_data'
]
