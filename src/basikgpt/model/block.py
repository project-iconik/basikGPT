"""Pre-Norm Transformer Decoder Block for basikGPT.

Assembles LayerNorm, Causal Multi-Head Self-Attention, and MLP sublayers
with residual skip connections in the canonical GPT-2 Pre-Norm topology.
"""

import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.model.attention import CausalSelfAttention
from basikgpt.model.mlp import MLP


class TransformerBlock(nn.Module):
    """Canonical Pre-Norm Transformer Decoder Block for GPT-2.

    Implements the standard GPT-2 block architecture where LayerNorm is applied
    *before* each sublayer (Pre-LayerNorm), and the sublayer output is added to the
    un-normalized residual stream:

        x_1 = x + Attention(LayerNorm_1(x))
        x_2 = x_1 + MLP(LayerNorm_2(x_1))

    Tensor Dimension Notation:
        B: Batch size (number of sequences in batch)
        T: Sequence length (time steps)
        C: Model dimension (config.d_model)

    Topology (Pre-Norm):
        x: (B, T, C)
          │
          ├──► LayerNorm 1 ──► CausalSelfAttention ──► (B, T, C)
          │         │
          ▼         ▼
          + ◄───────┘ (Residual Connection 1)
          │
          ├──► LayerNorm 2 ──► MLP ──────────────────► (B, T, C)
          │         │
          ▼         ▼
          + ◄───────┘ (Residual Connection 2)
          │
          ▼
        Output: (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # Pre-attention LayerNorm
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        # Causal Multi-Head Self-Attention (supports 'eager' and 'sdpa' backends)
        self.attn = CausalSelfAttention(config)

        # Pre-MLP LayerNorm
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        # Position-wise Feed-Forward Network (MLP)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Pre-Norm Transformer Block.

        Args:
            x: Hidden state tensor of shape (B, T, C).

        Returns:
            Updated hidden state tensor of shape (B, T, C).
        """
        # Sublayer 1: Pre-Norm Attention with Residual Addition
        # x: (B, T, C) + Attention(LayerNorm_1((B, T, C))) -> (B, T, C)
        x = x + self.attn(self.ln_1(x))

        # Sublayer 2: Pre-Norm MLP with Residual Addition
        # x: (B, T, C) + MLP(LayerNorm_2((B, T, C))) -> (B, T, C)
        x = x + self.mlp(self.ln_2(x))

        return x
