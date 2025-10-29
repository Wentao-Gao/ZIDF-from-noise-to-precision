"""
Diffusion Model for denoising predictions
Implementation based on DDPM (Denoising Diffusion Probabilistic Models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for time series denoising.

    Args:
        input_dim: Dimension of input time series
        hidden_dim: Hidden dimension for U-Net
        num_steps: Total number of diffusion steps (default: 1000)
        beta_start: Starting value for noise schedule (default: 0.0001)
        beta_end: Ending value for noise schedule (default: 0.02)
        device: Device to run the model on
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cuda',
        beta_schedule: str = 'linear'
    ):
        super(DiffusionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.device = device

        # Define noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == 'cosine':
            # Cosine schedule per Nichol & Dhariwal
            timesteps = num_steps
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            alphas_cumprod = (torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999).float()
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # U-Net for noise prediction
        self.noise_predictor = UNet1D(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps
        )

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).

        Args:
            x0: Clean data (B, T)
            t: Time steps (B,)

        Returns:
            xt: Noisy data at time t
            noise: Added noise
        """
        batch_size = x0.shape[0]

        # Get alpha_bar for each sample
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1)

        # Sample noise
        noise = torch.randn_like(x0)

        # Add noise: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt, noise

    def reverse_diffusion_step(
        self,
        xt: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t).

        Args:
            xt: Noisy data at time t (B, T)
            t: Current time step

        Returns:
            x_{t-1}: Less noisy data
        """
        batch_size = xt.shape[0]

        # Create time tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        predicted_noise = self.noise_predictor(xt, t_tensor)

        # Get parameters
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        # Compute mean of p(x_{t-1} | x_t)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = coef1 * (xt - coef2 * predicted_noise)

        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
            x_t_minus_1 = mean + sigma_t * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    def denoise(
        self,
        xt: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full denoising process from noisy predictions to clean predictions.

        Args:
            xt: Noisy predictions (B, T)
            num_steps: Number of reverse diffusion steps (default: self.num_steps)

        Returns:
            x0: Denoised predictions
        """
        if num_steps is None:
            num_steps = self.num_steps

        x = xt.clone()

        # Reverse diffusion from T to 0
        for t in reversed(range(num_steps)):
            x = self.reverse_diffusion_step(x, t)

        return x

    def compute_loss(
        self,
        x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss for diffusion model.

        Args:
            x0: Clean data (B, T)

        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x0.shape[0]

        # Sample random time steps
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)

        # Forward diffusion
        xt, noise = self.forward_diffusion(x0, t)

        # Predict noise
        predicted_noise = self.noise_predictor(xt, t)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss


class UNet1D(nn.Module):
    """
    1D U-Net for noise prediction in diffusion model.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_steps: Total diffusion steps (for time embedding)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 1000
    ):
        super(UNet1D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Time embedding (use discrete timestep indices directly)
        self.time_embed_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Embedding(num_steps, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Encoder
        self.enc1 = ConvBlock(1, hidden_dim // 4)
        self.enc2 = ConvBlock(hidden_dim // 4, hidden_dim // 2)
        self.enc3 = ConvBlock(hidden_dim // 2, hidden_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(hidden_dim, hidden_dim)

        # Decoder
        self.dec3 = ConvBlock(hidden_dim * 2, hidden_dim // 2)
        self.dec2 = ConvBlock(hidden_dim, hidden_dim // 4)
        self.dec1 = ConvBlock(hidden_dim // 2, hidden_dim // 4)

        # Output
        self.output = nn.Conv1d(hidden_dim // 4, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input noisy data (B, T)
            t: Time steps (B,)

        Returns:
            predicted_noise: Predicted noise (B, T)
        """
        # Reshape input: (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)

        # Time embedding: pass discrete indices to Embedding
        # t: (B,) long tensor of timesteps
        t_emb = self.time_embed(t)  # (B, hidden_dim)

        # Encoder
        enc1 = self.enc1(x)  # (B, hidden_dim//4, T)
        enc2 = self.enc2(F.max_pool1d(enc1, 2))  # (B, hidden_dim//2, T//2)
        enc3 = self.enc3(F.max_pool1d(enc2, 2))  # (B, hidden_dim, T//4)

        # Bottleneck with time embedding
        bottleneck = self.bottleneck(F.max_pool1d(enc3, 2))  # (B, hidden_dim, T//8)
        bottleneck = bottleneck + t_emb.unsqueeze(-1)  # Add time embedding

        # Decoder with skip connections
        dec3 = F.interpolate(bottleneck, scale_factor=2, mode='linear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = F.interpolate(dec3, scale_factor=2, mode='linear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = F.interpolate(dec2, scale_factor=2, mode='linear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        output = self.output(dec1)  # (B, 1, T)
        output = output.squeeze(1)  # (B, T)

        return output


class ConvBlock(nn.Module):
    """
    Convolutional block with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.silu(out)

        return out
