"""
Baseline models for comparison
Including ZIP, Hurdle, and ZIG models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import stats


class ZeroInflatedPoisson(nn.Module):
    """
    Zero-Inflated Poisson (ZIP) model for count data.

    The model assumes:
    - Zeros can come from two sources (structural vs sampling zeros)
    - Non-zero values follow a Poisson distribution

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ZeroInflatedPoisson, self).__init__()

        # Logistic regression for zero inflation
        self.zero_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Poisson regression for count
        self.count_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive lambda
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, D)

        Returns:
            pi: Probability of structural zero (B, 1)
            lambda_: Poisson rate parameter (B, 1)
        """
        pi = self.zero_net(x)
        lambda_ = self.count_net(x)

        return pi, lambda_

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input features (B, D)

        Returns:
            predictions: Expected values (B, 1)
        """
        pi, lambda_ = self.forward(x)
        # E[Y] = (1 - pi) * lambda
        predictions = (1 - pi) * lambda_

        return predictions

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            x: Input features (B, D)
            y: Target values (B,)

        Returns:
            loss: Negative log-likelihood
        """
        pi, lambda_ = self.forward(x)
        y = y.unsqueeze(-1)

        # Log-likelihood for zero observations
        log_lik_zero = torch.log(pi + (1 - pi) * torch.exp(-lambda_) + 1e-8)

        # Log-likelihood for non-zero observations
        log_lik_nonzero = (
            torch.log(1 - pi + 1e-8) +
            y * torch.log(lambda_ + 1e-8) -
            lambda_ -
            torch.lgamma(y + 1)
        )

        # Combine based on observed values
        zero_mask = (y == 0).float()
        log_lik = zero_mask * log_lik_zero + (1 - zero_mask) * log_lik_nonzero

        # Negative log-likelihood
        loss = -log_lik.mean()

        return loss


class HurdleModel(nn.Module):
    """
    Hurdle model for zero-inflated data.

    The model has two stages:
    1. Binary model for zero vs non-zero
    2. Truncated distribution for positive values

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(HurdleModel, self).__init__()

        # Binary classifier for zero vs non-zero
        self.binary_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Regression for positive values
        self.positive_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, D)

        Returns:
            p_nonzero: Probability of non-zero (B, 1)
            mu: Mean for positive values (B, 1)
        """
        p_nonzero = self.binary_net(x)
        mu = self.positive_net(x)

        return p_nonzero, mu

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input features (B, D)

        Returns:
            predictions: Expected values (B, 1)
        """
        p_nonzero, mu = self.forward(x)
        # E[Y] = p_nonzero * E[Y | Y > 0]
        predictions = p_nonzero * mu

        return predictions

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.

        Args:
            x: Input features (B, D)
            y: Target values (B,)

        Returns:
            loss: Combined binary and regression loss
        """
        p_nonzero, mu = self.forward(x)
        y = y.unsqueeze(-1)

        # Binary classification loss
        y_binary = (y > 0).float()
        binary_loss = F.binary_cross_entropy(p_nonzero, y_binary)

        # Regression loss for positive values
        positive_mask = (y > 0).float()
        if positive_mask.sum() > 0:
            regression_loss = F.mse_loss(
                mu * positive_mask,
                y * positive_mask,
                reduction='sum'
            ) / (positive_mask.sum() + 1e-8)
        else:
            regression_loss = torch.tensor(0.0, device=x.device)

        # Combined loss
        loss = binary_loss + regression_loss

        return loss


class ZeroInflatedGaussian(nn.Module):
    """
    Zero-Inflated Gaussian (ZIG) model.

    Models zeros separately from continuous positive values
    using a Gaussian distribution.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ZeroInflatedGaussian, self).__init__()

        # Logistic regression for zero inflation
        self.zero_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Gaussian regression for continuous values
        self.mean_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, D)

        Returns:
            pi: Probability of structural zero (B, 1)
            mu: Mean of Gaussian (B, 1)
            logvar: Log variance of Gaussian (B, 1)
        """
        pi = self.zero_net(x)
        mu = self.mean_net(x)
        logvar = self.logvar_net(x)

        return pi, mu, logvar

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            x: Input features (B, D)

        Returns:
            predictions: Expected values (B, 1)
        """
        pi, mu, _ = self.forward(x)
        # E[Y] = (1 - pi) * mu
        predictions = (1 - pi) * mu

        # Ensure non-negative
        predictions = torch.clamp(predictions, min=0.0)

        return predictions

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            x: Input features (B, D)
            y: Target values (B,)

        Returns:
            loss: Negative log-likelihood
        """
        pi, mu, logvar = self.forward(x)
        y = y.unsqueeze(-1)

        var = torch.exp(logvar)

        # Log-likelihood for zero observations
        # p(y=0) = pi + (1-pi) * N(0; mu, var)
        gaussian_at_zero = -0.5 * (mu ** 2) / var - 0.5 * logvar
        log_lik_zero = torch.log(pi + (1 - pi) * torch.exp(gaussian_at_zero) + 1e-8)

        # Log-likelihood for non-zero observations
        # p(y>0) = (1-pi) * N(y; mu, var)
        log_lik_nonzero = (
            torch.log(1 - pi + 1e-8) -
            0.5 * logvar -
            0.5 * ((y - mu) ** 2) / var
        )

        # Combine based on observed values
        zero_mask = (y == 0).float()
        log_lik = zero_mask * log_lik_zero + (1 - zero_mask) * log_lik_nonzero

        # Negative log-likelihood
        loss = -log_lik.mean()

        return loss
