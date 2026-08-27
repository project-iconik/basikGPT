"""Causal Multi-Head Self-Attention for basikGPT.

Provides both an educational reference Eager attention implementation and an optimized
PyTorch SDPA (Scaled Dot-Product Attention) backend.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basikgpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention module for GPT-2.

    This module performs autoregressive multi-head self-attention over an input sequence
    of hidden states `(B, T, C)`. It supports two backends:
    - `"eager"`: Educational reference implementation using explicit matrix multiplication,
      scaling, lower-triangular causal masking, softmax, and dropout.
    - `"sdpa"`: Optimized pretraining backend using PyTorch's `F.scaled_dot_product_attention`.

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
        4. Attention Backend:
           - eager: (Q @ K^T) / sqrt(D) -> mask -> softmax -> dropout -> @ V -> (B, H, T, D)
           - sdpa: F.scaled_dot_product_attention(Q, K, V, is_causal=True) -> (B, H, T, D)
        5. Head Merge: (B, H, T, D) -> (B, T, C)
        6. Output Linear projection: (B, T, C) -> (B, T, C)
        7. Residual Dropout: (B, T, C) -> (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.context_length = config.context_length
        self.dropout = config.dropout
        self.attention_backend = config.attention_backend

        # Shared fused projection for Query, Key, and Value (3 * d_model output)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Shared output projection back to residual hidden dimension
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropouts for attention matrix probabilities and output residual stream
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask: shape (1, 1, context_length, context_length)
        # registered as non-persistent buffer for eager attention reference so it moves
        # with .to(device) without polluting the model state_dict.
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        ).view(1, 1, config.context_length, config.context_length)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """Educational reference scaled dot-product attention with manual causal masking."""
        # 1. Scaled dot-product attention scores: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # 2. Slice and apply causal mask to prevent attending to future tokens (j > i)
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # 3. Softmax along key position dimension (dim=-1) to obtain probability distribution
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Attention probability dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 5. Weighted value aggregation: (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        y = attn_weights @ v
        return y

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized scaled dot-product attention using PyTorch SDPA primitive."""
        # Ensure dropout is strictly 0.0 during evaluation mode
        dropout_p = self.dropout if self.training else 0.0

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Causal Multi-Head Self-Attention.

        Args:
            x: Input tensor of shape (B, T, C) representing hidden states.

        Returns:
            Output tensor of shape (B, T, C) after attention and output projection.

        Raises:
            ValueError: If input sequence length T exceeds config.context_length,
                        or if feature dimension C does not match config.d_model,
                        or if an unrecognized attention backend is configured.
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

        # Step 4: Core Attention Computation dispatched by configured backend
        if self.attention_backend == "eager":
            y = self._eager_attention(q, k, v, T)
        elif self.attention_backend == "sdpa":
            y = self._sdpa_attention(q, k, v)
        else:
            raise ValueError(f"Unsupported attention backend: '{self.attention_backend}'.")

        # Step 5: Head merge: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        # .contiguous() is required because transpose(1, 2) alters memory strides
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 6: Output linear projection: (B, T, C) -> (B, T, C)
        y = self.out_proj(y)

        # Step 7: Residual dropout
        y = self.resid_dropout(y)

        return y
