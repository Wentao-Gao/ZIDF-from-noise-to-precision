"""
Adapter for using the official Non-stationary Transformer implementation.

Usage:
- Provide a constructed official model instance to OfficialNSTWrapper.
- The wrapper exposes a unified forward(x, y_history) -> (B, T_out) interface
  used by the rest of this repo.

Note:
- Constructor arguments and exact forward signature of the official model may differ
  by repo/version. This wrapper tries common call patterns and falls back gracefully.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class OfficialNSTWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, output_len: int, device: str = 'cuda'):
        super().__init__()
        self.base_model = base_model
        self.output_len = output_len
        self.device = device

    def forward(self, x: torch.Tensor, y_history: torch.Tensor) -> torch.Tensor:
        """
        Attempts several common forward call patterns:
        1) base_model(x, y_history)
        2) base_model(x)
        3) base_model(enc_input=x, dec_input=y_history) (by kwargs)
        """
        out = None
        # Try (x, y_history)
        try:
            out = self.base_model(x, y_history)
        except Exception:
            pass
        # Try (x)
        if out is None:
            try:
                out = self.base_model(x)
            except Exception:
                pass
        # Try kwargs
        if out is None:
            try:
                out = self.base_model(enc_input=x, dec_input=y_history)
            except Exception:
                raise RuntimeError(
                    "Official NST forward signature not recognized. Please adjust the adapter."
                )

        # Ensure shape (B, T_out)
        if out.dim() == 3 and out.size(-1) == 1:
            out = out.squeeze(-1)
        if out.size(-1) != self.output_len:
            # If output length differs, take last T_out as a heuristic
            out = out[..., -self.output_len:]
        return out

