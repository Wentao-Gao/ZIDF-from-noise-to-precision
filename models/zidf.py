"""
Zero Inflation Diffusion Framework (ZIDF)
Main framework implementation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class ZIDF(nn.Module):
    """
    Zero Inflation Diffusion Framework for precipitation forecasting.

    This framework combines:
    1. Gaussian perturbation for smoothing zero-inflated distributions
    2. Transformer-based prediction for capturing temporal patterns
    3. Diffusion-based denoising to restore the original data structure

    Args:
        predictive_model: The base forecasting model (e.g., Non-stationary Transformer)
        diffusion_model: The diffusion model for denoising
        alpha_noise: Noise level for Gaussian perturbation (default: 0.1)
        device: Device to run the model on
    """

    def __init__(
        self,
        predictive_model: nn.Module,
        diffusion_model: nn.Module,
        alpha_noise: float = 0.1,
        device: str = 'cuda'
    ):
        super(ZIDF, self).__init__()

        self.predictive_model = predictive_model
        self.diffusion_model = diffusion_model
        self.alpha_noise = alpha_noise
        self.device = device

    def add_gaussian_noise(
        self,
        y: torch.Tensor,
        mean_y: Optional[float] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Add Gaussian noise to transform zero-inflated distribution.

        Args:
            y: Target values (B, T)
            mean_y: Mean of non-zero values (computed if not provided)

        Returns:
            y_noisy: Noisy target values
            sigma: Standard deviation of added noise
        """
        if mean_y is None:
            # Compute mean of non-zero values
            non_zero_mask = y > 0
            if non_zero_mask.sum() > 0:
                mean_y = y[non_zero_mask].mean().item()
            else:
                mean_y = 1.0  # Fallback value

        sigma = self.alpha_noise * mean_y
        noise = torch.randn_like(y) * sigma
        y_noisy = y + noise

        return y_noisy, sigma

    def forward_train(
        self,
        x: torch.Tensor,
        y_history: torch.Tensor,
        y_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during training.

        Args:
            x: Input features (B, T_in, D)
            y_history: Historical target values (B, T_in)
            y_target: Future target values (B, T_out)

        Returns:
            predictions: Noisy predictions (B, T_out)
            y_target_noisy: Noisy ground truth (B, T_out)
        """
        # Add noise to targets
        y_history_noisy, _ = self.add_gaussian_noise(y_history)
        y_target_noisy, _ = self.add_gaussian_noise(y_target)

        # Predict on smoothed data
        predictions = self.predictive_model(x, y_history_noisy)

        return predictions, y_target_noisy

    def forward_inference(
        self,
        x: torch.Tensor,
        y_history: torch.Tensor,
        num_diffusion_steps: int = 1000
    ) -> torch.Tensor:
        """
        Forward pass during inference with diffusion-based denoising.

        Args:
            x: Input features (B, T_in, D)
            y_history: Historical target values (B, T_in)
            num_diffusion_steps: Number of diffusion steps for denoising

        Returns:
            predictions_clean: Denoised predictions (B, T_out)
        """
        # Add noise to historical data (same as training)
        y_history_noisy, _ = self.add_gaussian_noise(y_history)

        # Get noisy predictions
        predictions_noisy = self.predictive_model(x, y_history_noisy)

        # Denoise using diffusion model
        predictions_clean = self.diffusion_model.denoise(
            predictions_noisy,
            num_steps=num_diffusion_steps
        )

        # Ensure non-negative predictions for precipitation
        predictions_clean = torch.clamp(predictions_clean, min=0.0)

        return predictions_clean

    def forward(
        self,
        x: torch.Tensor,
        y_history: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass that switches between training and inference modes.

        Args:
            x: Input features (B, T_in, D)
            y_history: Historical target values (B, T_in)
            y_target: Future target values (B, T_out), only needed for training
            training: Whether in training mode

        Returns:
            predictions: Model predictions
        """
        if training:
            assert y_target is not None, "y_target is required for training"
            return self.forward_train(x, y_history, y_target)
        else:
            return self.forward_inference(x, y_history)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = 'mse'
    ) -> torch.Tensor:
        """
        Compute prediction loss.

        Args:
            predictions: Model predictions (B, T_out)
            targets: Ground truth targets (B, T_out)
            loss_type: Type of loss ('mse' or 'mae')

        Returns:
            loss: Computed loss value
        """
        if loss_type == 'mse':
            loss = nn.MSELoss()(predictions, targets)
        elif loss_type == 'mae':
            loss = nn.L1Loss()(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss
