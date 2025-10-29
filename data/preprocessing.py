"""
Data preprocessing utilities for precipitation data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
import warnings


def preprocess_precipitation_data(
    data: np.ndarray,
    feature_cols: list,
    target_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Preprocess precipitation data for training.

    Args:
        data: Input data as DataFrame or array
        feature_cols: List of feature column names
        target_col: Target column name
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        normalize_features: Whether to normalize features

    Returns:
        train_features, train_targets, val_features, val_targets,
        test_features, test_targets, metadata
    """
    if isinstance(data, pd.DataFrame):
        features = data[feature_cols].values
        targets = data[target_col].values
    else:
        # Assume data is already numpy array
        features = data[:, :-1]
        targets = data[:, -1]

    # Split data
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_features = features[:train_end]
    train_targets = targets[:train_end]

    val_features = features[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_features = features[val_end:]
    test_targets = targets[val_end:]

    # Normalize features
    scaler = None
    if normalize_features:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

    # Compute metadata
    metadata = {
        'train_size': len(train_features),
        'val_size': len(val_features),
        'test_size': len(test_features),
        'num_features': train_features.shape[1],
        'zero_ratio_train': np.sum(train_targets == 0) / len(train_targets),
        'zero_ratio_val': np.sum(val_targets == 0) / len(val_targets),
        'zero_ratio_test': np.sum(test_targets == 0) / len(test_targets),
        'mean_nonzero_train': train_targets[train_targets > 0].mean() if np.any(train_targets > 0) else 0,
        'scaler': scaler
    }

    return (
        train_features, train_targets,
        val_features, val_targets,
        test_features, test_targets,
        metadata
    )


def load_ncep_data(
    file_path: str,
    region: Dict[str, float],
    var_name: str
) -> pd.Series:
    """
    Load and process NCEP/NCAR Reanalysis data.

    Args:
        file_path: Path to NetCDF file
        region: Dictionary with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
        variables: List of variable names to extract

    Returns:
        DataFrame with extracted data
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for loading NetCDF files. Install with: pip install xarray netCDF4")

    # Load NetCDF file
    ds = xr.open_dataset(file_path)

    # Extract region
    ds_region = ds.sel(
        lat=slice(region['lat_min'], region['lat_max']),
        lon=slice(region['lon_min'], region['lon_max'])
    )

    if var_name not in ds_region:
        raise KeyError(f"Variable {var_name} not found in dataset")

    # Compute spatial mean and convert to pandas Series
    series = ds_region[var_name].mean(dim=['lat', 'lon']).to_series()
    series.name = var_name
    return series


def build_sa_dataset_from_netcdf(netcdf_paths: Dict[str, str], region: Dict[str, float], variables: list) -> pd.DataFrame:
    """
    Build South Australia dataset by loading multiple NetCDF variables and aligning by time.
    Args:
        netcdf_paths: dict mapping variable -> file path
        region: dict with lat/lon bounds
        variables: list of variable names to load
    Returns:
        DataFrame indexed by time with selected variables
    """
    series_list = []
    for var in variables:
        path = netcdf_paths.get(var)
        if not path:
            warnings.warn(f"Missing path for variable {var}; skipping")
            continue
        s = load_ncep_data(path, region, var)
        series_list.append(s)

    if not series_list:
        raise ValueError("No variables loaded; check netcdf_paths and variables")

    df = pd.concat(series_list, axis=1).dropna()
    return df


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 5,
    zero_ratio: float = 0.7,
    gamma_shape: float = 2.0,
    gamma_scale: float = 0.05,
    random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic zero-inflated precipitation data.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        zero_ratio: Ratio of zeros in target
        gamma_shape: Shape parameter for Gamma distribution
        gamma_scale: Scale parameter for Gamma distribution
        random_seed: Random seed for reproducibility

    Returns:
        features: Synthetic features (n_samples, n_features)
        targets: Synthetic zero-inflated targets (n_samples,)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate features
    # Periodic component
    t = np.arange(n_samples)
    periodic = np.sin(2 * np.pi * t / 365)[:, np.newaxis]  # Annual cycle

    # Linear trend
    linear = (t / n_samples)[:, np.newaxis]

    # Random features
    random_features = np.random.randn(n_samples, n_features - 2)

    features = np.hstack([periodic, linear, random_features])

    # Generate zero-inflated targets
    # First, generate non-zero values from Gamma distribution
    targets_continuous = np.random.gamma(gamma_shape, gamma_scale, n_samples)

    # Apply zero inflation
    zero_mask = np.random.rand(n_samples) < zero_ratio
    targets = targets_continuous.copy()
    targets[zero_mask] = 0

    # Add correlation with features (especially periodic component)
    correlation_strength = 0.3
    feature_effect = correlation_strength * (periodic.squeeze() + 1) / 2
    targets = targets * (1 + feature_effect)

    return features, targets


def create_sliding_windows(
    data: np.ndarray,
    input_len: int,
    output_len: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data.

    Args:
        data: Time series data (T, D)
        input_len: Length of input window
        output_len: Length of output window
        stride: Stride for sliding

    Returns:
        X: Input windows (N, input_len, D)
        Y: Output windows (N, output_len, D)
    """
    X, Y = [], []

    for i in range(0, len(data) - input_len - output_len + 1, stride):
        X.append(data[i:i + input_len])
        Y.append(data[i + input_len:i + input_len + output_len])

    return np.array(X), np.array(Y)


def compute_statistics(targets: np.ndarray) -> Dict:
    """
    Compute statistics for precipitation data.

    Args:
        targets: Target precipitation values

    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_samples': len(targets),
        'num_zeros': np.sum(targets == 0),
        'num_nonzeros': np.sum(targets > 0),
        'zero_ratio': np.sum(targets == 0) / len(targets),
        'mean': np.mean(targets),
        'std': np.std(targets),
        'min': np.min(targets),
        'max': np.max(targets),
        'median': np.median(targets),
        'q25': np.percentile(targets, 25),
        'q75': np.percentile(targets, 75),
    }

    # Statistics for non-zero values
    if np.any(targets > 0):
        nonzero_targets = targets[targets > 0]
        stats['mean_nonzero'] = np.mean(nonzero_targets)
        stats['std_nonzero'] = np.std(nonzero_targets)
        stats['min_nonzero'] = np.min(nonzero_targets)
        stats['max_nonzero'] = np.max(nonzero_targets)

    return stats


def assemble_features_targets(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame with time index and columns including the target, assemble features and target arrays.
    Adds periodic (annual) and linear trend features.
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col} not in DataFrame")
    t = np.arange(len(df))
    periodic = np.sin(2 * np.pi * t / 365.0)
    linear = t / max(len(df), 1)

    X_other = df.drop(columns=[target_col]).values
    X = np.column_stack([periodic, linear, X_other])
    y = df[target_col].values.astype(float)
    return X, y
