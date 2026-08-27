"""Causal Multi-Head Self-Attention for basikGPT.

Implements the educational reference Eager causal multi-head self-attention module
using explicit PyTorch tensor operations.
"""

import math
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention module for GPT-2.

    This module performs autoregressive multi-head self-attention over an input sequence
    of hidden states `(B, T, C)`. It enforces causality using a lower-triangular mask
    so each token can only attend to itself and preceding tokens.

    Tensor Dimension Notation:
        B: Batch size (number of sequences in batch)
        T: Sequence length (number of time steps, T <= context_length)
        C: Model / embedding dimension (d_model)
        H: Number of attention heads (n_heads)
        D: Head dimension (head_dim = C // H)

    Architecture:
        1. Fused QKV Linear projection: (B, T, C) -> (B, T, 3 * C)
        2. Chunk into Query, Key, Value: (B, T, C) each
        3. Head split & transpose: (B, T, C) -> (B, H, T, D)
        4. Scaled Dot-Product: (Q @ K^T) / sqrt(D) -> (B, H, T, T)
        5. Causal Masking: Mask future positions (j > i) with -infinity
        6. Softmax along key dimension: dim=-1 -> (B, H, T, T)
        7. Attention Dropout: (B, H, T, T)
        8. Value Aggregation: attn_weights @ V -> (B, H, T, D)
        9. Head Merge: (B, H, T, D) -> (B, T, C)
        10. Output Linear projection: (B, T, C) -> (B, T, C)
        11. Residual Dropout: (B, T, C) -> (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.context_length = config.context_length

        # Fused projection for Query, Key, and Value (3 * d_model output)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection back to residual hidden dimension
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropouts for attention matrix probabilities and output residual stream
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask: shape (1, 1, context_length, context_length)
        # registered as non-persistent buffer so it moves with .to(device)
        # without being saved into model state_dict.
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        ).view(1, 1, config.context_length, config.context_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Causal Multi-Head Self-Attention.

        Args:
            x: Input tensor of shape (B, T, C) representing hidden states.

        Returns:
            Output tensor of shape (B, T, C) after attention and output projection.

        Raises:
            ValueError: If input sequence length T exceeds config.context_length,
                        or if feature dimension C does not match config.d_model.
        """
        B, T, C = x.shape

        if T > self.context_length:
            raise ValueError(
                f"Input sequence length T={T} exceeds maximum configured context length {self.context_length}."
            )
        if C != self.d_model:
            raise ValueError(
                f"Input feature dimension C={C} does not match model d_model={self.d_model}."
            )

        # Step 1: Fused QKV projection: (B, T, C) -> (B, T, 3 * C)
        qkv = self.qkv_proj(x)

        # Step 2: Separate Query, Key, and Value: each (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 3: Reshape for multi-head attention: (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 4: Scaled dot-product attention scores: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        # Scaled by 1 / sqrt(D) to normalize variance of dot-products to ~1.0
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Step 5: Apply causal mask to prevent attending to future tokens (j > i)
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # Step 6: Softmax along key position dimension (dim=-1) to obtain probability distribution
        attn_weights = torch.softmax(scores, dim=-1)

        # Step 7: Attention probability dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Step 8: Value aggregation: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        y = attn_weights @ v

        # Step 9: Head merge: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        # .contiguous() is required because transpose(1, 2) alters memory strides
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 10: Output linear projection: (B, T, C) -> (B, T, C)
        y = self.out_proj(y)

        # Step 11: Residual dropout
        y = self.resid_dropout(y)

        return y
