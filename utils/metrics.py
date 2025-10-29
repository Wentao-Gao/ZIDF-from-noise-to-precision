"""
Evaluation metrics for precipitation forecasting
"""

import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Overall metrics
    metrics[f'{prefix}mse'] = mean_squared_error(targets, predictions)
    metrics[f'{prefix}rmse'] = np.sqrt(metrics[f'{prefix}mse'])
    metrics[f'{prefix}mae'] = mean_absolute_error(targets, predictions)
    metrics[f'{prefix}r2'] = r2_score(targets, predictions)

    # Metrics for non-zero values
    nonzero_mask = targets > 0
    if np.any(nonzero_mask):
        metrics[f'{prefix}nonzero_mse'] = mean_squared_error(
            targets[nonzero_mask],
            predictions[nonzero_mask]
        )
        metrics[f'{prefix}nonzero_mae'] = mean_absolute_error(
            targets[nonzero_mask],
            predictions[nonzero_mask]
        )

    # Metrics for zero detection
    pred_zero = predictions < 0.001  # Threshold for considering prediction as zero
    true_zero = targets == 0

    metrics[f'{prefix}zero_precision'] = (
        np.sum(pred_zero & true_zero) / np.sum(pred_zero)
        if np.sum(pred_zero) > 0 else 0
    )
    metrics[f'{prefix}zero_recall'] = (
        np.sum(pred_zero & true_zero) / np.sum(true_zero)
        if np.sum(true_zero) > 0 else 0
    )

    # Mean absolute percentage error (for non-zero values)
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((targets[nonzero_mask] - predictions[nonzero_mask]) /
                               (targets[nonzero_mask] + 1e-8))) * 100
        metrics[f'{prefix}mape'] = mape

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    diffusion_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to use
        diffusion_steps: Number of diffusion steps for ZIDF

    Returns:
        predictions: All predictions
        targets: All targets
        metrics: Dictionary of metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, y_history, y_target in dataloader:
            x = x.to(device)
            y_history = y_history.to(device)
            y_target = y_target.to(device)

            # Make predictions
            if hasattr(model, 'forward_inference'):
                # ZIDF model
                predictions = model.forward_inference(
                    x, y_history, num_diffusion_steps=diffusion_steps
                )
            else:
                # Other models
                predictions = model(x, y_history)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_target.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten if needed
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    return predictions, targets, metrics


def compute_zero_inflation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics specific to zero-inflated data.

    Args:
        predictions: Model predictions
        targets: Ground truth targets

    Returns:
        Dictionary of zero-inflation specific metrics
    """
    metrics = {}

    # Zero inflation ratios
    metrics['pred_zero_ratio'] = np.sum(predictions < 0.001) / len(predictions)
    metrics['true_zero_ratio'] = np.sum(targets == 0) / len(targets)

    # Confusion matrix for zero classification
    pred_zero = predictions < 0.001
    true_zero = targets == 0

    tp = np.sum(pred_zero & true_zero)  # True positives (correctly predicted zeros)
    fp = np.sum(pred_zero & ~true_zero)  # False positives (incorrectly predicted zeros)
    tn = np.sum(~pred_zero & ~true_zero)  # True negatives (correctly predicted non-zeros)
    fn = np.sum(~pred_zero & true_zero)  # False negatives (incorrectly predicted non-zeros)

    metrics['zero_tp'] = tp
    metrics['zero_fp'] = fp
    metrics['zero_tn'] = tn
    metrics['zero_fn'] = fn

    # Accuracy
    metrics['zero_accuracy'] = (tp + tn) / len(predictions)

    # F1 score for zero detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['zero_f1'] = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return metrics
