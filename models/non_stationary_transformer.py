"""
Non-stationary Transformer
Implementation based on "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class NonStationaryTransformer(nn.Module):
    """
    Non-stationary Transformer for time series forecasting.

    Handles non-stationarity through de-stationary attention mechanism.

    Args:
        input_dim: Number of input features
        output_len: Length of output sequence
        d_model: Dimension of model (default: 512)
        n_heads: Number of attention heads (default: 8)
        e_layers: Number of encoder layers (default: 2)
        d_layers: Number of decoder layers (default: 1)
        d_ff: Dimension of feedforward network (default: 2048)
        dropout: Dropout rate (default: 0.05)
        activation: Activation function (default: 'gelu')
        device: Device to run on
    """

    def __init__(
        self,
        input_dim: int,
        output_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.05,
        activation: str = 'gelu',
        device: str = 'cuda'
    ):
        super(NonStationaryTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_len = output_len
        self.d_model = d_model
        self.device = device

        # Input embedding
        self.enc_embedding = DataEmbedding(input_dim + 1, d_model, dropout)  # +1 for target
        self.dec_embedding = DataEmbedding(1, d_model, dropout)  # Only target for decoder

        # De-stationary components
        self.tau_learner = nn.Linear(1, 1)
        self.delta_learner = nn.Linear(1, 1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DeStationaryAttention(d_model, n_heads),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DeStationaryAttention(d_model, n_heads),
                        d_model,
                        n_heads
                    ),
                    AttentionLayer(
                        DeStationaryAttention(d_model, n_heads),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Output projection
        self.projection = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        y_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (B, T_in, D)
            y_history: Historical target values (B, T_in)

        Returns:
            predictions: Forecasted values (B, T_out)
        """
        batch_size, seq_len, _ = x.shape

        # Concatenate features and target
        y_history_expanded = y_history.unsqueeze(-1)  # (B, T_in, 1)
        x_with_target = torch.cat([x, y_history_expanded], dim=-1)  # (B, T_in, D+1)

        # De-stationarization
        means = x_with_target.mean(dim=1, keepdim=True)
        x_destat = x_with_target - means
        stdev = torch.sqrt(torch.var(x_destat, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_destat = x_destat / stdev

        # Learn tau and delta
        tau = self.tau_learner(means[:, :, -1:])  # (B, 1, 1)
        delta = self.delta_learner(stdev[:, :, -1:])  # (B, 1, 1)

        # Encoder
        enc_out = self.enc_embedding(x_destat)
        enc_out, _ = self.encoder(enc_out)

        # Decoder input: use last value as starting point
        dec_inp = torch.zeros(batch_size, self.output_len, 1).to(self.device)
        dec_inp[:, 0, :] = y_history[:, -1:]

        # Decoder
        dec_out = self.dec_embedding(dec_inp)
        dec_out = self.decoder(dec_out, enc_out)

        # Output projection
        output = self.projection(dec_out)  # (B, T_out, 1)

        # Re-stationarization
        output = output * delta + tau

        return output.squeeze(-1)  # (B, T_out)


class DeStationaryAttention(nn.Module):
    """
    De-stationary Attention mechanism.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super(DeStationaryAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            queries: Query tensor (B, L_q, D)
            keys: Key tensor (B, L_k, D)
            values: Value tensor (B, L_v, D)
            attn_mask: Attention mask

        Returns:
            output: Attention output (B, L_q, D)
            attn_weights: Attention weights
        """
        B, L_q, _ = queries.shape
        _, L_k, _ = keys.shape

        # Linear projections and split into heads
        Q = self.W_q(queries).view(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(B, L_k, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        return output, attn_weights


class AttentionLayer(nn.Module):
    """
    Attention layer wrapper.
    """

    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.attention = attention
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        out, attn = self.attention(queries, keys, values, attn_mask)
        out = self.out_projection(out)
        return out, attn


class EncoderLayer(nn.Module):
    """
    Encoder layer with self-attention and feedforward.
    """

    def __init__(
        self,
        attention,
        d_model,
        d_ff,
        dropout=0.1,
        activation='gelu'
    ):
        super(EncoderLayer, self).__init__()

        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feedforward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    """
    Encoder with multiple layers.
    """

    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Decoder layer with self-attention, cross-attention, and feedforward.
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff,
        dropout=0.1,
        activation='gelu'
    ):
        super(DecoderLayer, self).__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, cross, self_attn_mask=None, cross_attn_mask=None):
        # Self-attention
        new_x, self_attn = self.self_attention(x, x, x, self_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Cross-attention
        new_x, cross_attn = self.cross_attention(x, cross, cross, cross_attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # Feedforward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x = self.norm3(x)

        return x, self_attn, cross_attn


class Decoder(nn.Module):
    """
    Decoder with multiple layers.
    """

    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            x, _, _ = layer(x, cross, self_attn_mask, cross_attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DataEmbedding(nn.Module):
    """
    Data embedding with positional encoding.
    """

    def __init__(self, input_dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(input_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]
