"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
import seaborn as sns


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_samples: int = 500,
    save_path: Optional[str] = None
):
    """
    Plot predictions vs targets.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        num_samples: Number of samples to plot
        save_path: Path to save figure
    """
    # Sample data if too large
    if len(predictions) > num_samples:
        indices = np.random.choice(len(predictions), num_samples, replace=False)
        predictions = predictions[indices]
        targets = targets[indices]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time series plot
    axes[0, 0].plot(targets, label='Ground Truth', alpha=0.7)
    axes[0, 0].plot(predictions, label='Predictions', alpha=0.7)
    axes[0, 0].set_title('Time Series Comparison')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Precipitation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot
    axes[0, 1].scatter(targets, predictions, alpha=0.5)
    max_val = max(targets.max(), predictions.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    axes[0, 1].set_title('Predictions vs Ground Truth')
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Predictions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Residual plot
    residuals = predictions - targets
    axes[1, 0].scatter(targets, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].set_xlabel('Ground Truth')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1, 1].hist(targets, bins=50, alpha=0.5, label='Ground Truth', density=True)
    axes[1, 1].hist(predictions, bins=50, alpha=0.5, label='Predictions', density=True)
    axes[1, 1].set_title('Distribution Comparison')
    axes[1, 1].set_xlabel('Precipitation')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_zero_inflation_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Analyze and plot zero inflation characteristics.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Zero inflation ratios
    pred_zero_ratio = np.sum(predictions < 0.001) / len(predictions)
    true_zero_ratio = np.sum(targets == 0) / len(targets)

    ratios = [true_zero_ratio, pred_zero_ratio]
    labels = ['Ground Truth', 'Predictions']
    colors = ['#3498db', '#e74c3c']

    axes[0, 0].bar(labels, ratios, color=colors, alpha=0.7)
    axes[0, 0].set_title('Zero Inflation Ratio Comparison')
    axes[0, 0].set_ylabel('Zero Ratio')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    for i, (label, ratio) in enumerate(zip(labels, ratios)):
        axes[0, 0].text(i, ratio + 0.02, f'{ratio:.3f}', ha='center', fontweight='bold')

    # Distribution of non-zero values
    nonzero_targets = targets[targets > 0]
    nonzero_predictions = predictions[predictions > 0.001]

    axes[0, 1].hist(nonzero_targets, bins=50, alpha=0.5, label='Ground Truth', density=True)
    axes[0, 1].hist(nonzero_predictions, bins=50, alpha=0.5, label='Predictions', density=True)
    axes[0, 1].set_title('Distribution of Non-Zero Values')
    axes[0, 1].set_xlabel('Precipitation')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion matrix for zero classification
    pred_zero = predictions < 0.001
    true_zero = targets == 0

    tp = np.sum(pred_zero & true_zero)
    fp = np.sum(pred_zero & ~true_zero)
    tn = np.sum(~pred_zero & ~true_zero)
    fn = np.sum(~pred_zero & true_zero)

    confusion_matrix = np.array([[tp, fp], [fn, tn]])
    confusion_matrix_norm = confusion_matrix / confusion_matrix.sum()

    sns.heatmap(
        confusion_matrix_norm,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=['Pred Zero', 'Pred Non-Zero'],
        yticklabels=['True Zero', 'True Non-Zero'],
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Normalized Confusion Matrix')

    # Error distribution for non-zero values
    if len(nonzero_targets) > 0:
        # Match indices
        nonzero_mask = targets > 0
        errors = predictions[nonzero_mask] - targets[nonzero_mask]

        axes[1, 1].hist(errors, bins=50, alpha=0.7, color='#9b59b6')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Error Distribution (Non-Zero Values)')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        axes[1, 1].text(
            0.05, 0.95,
            f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
            transform=axes[1, 1].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot attention weights heatmap.

    Args:
        attention_weights: Attention weights (num_heads, seq_len, seq_len)
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_heads):
        sns.heatmap(
            attention_weights[i],
            cmap='viridis',
            ax=axes[i],
            cbar=True
        )
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')

    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
