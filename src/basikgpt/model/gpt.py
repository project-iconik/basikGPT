"""Full GPT-2 Decoder-Only Transformer Model for basikGPT.

Assembles Token & Positional Embeddings, Transformer Decoder Blocks,
Final LayerNorm, and the Language Model Head with Weight Tying.
"""

import math
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.model.block import TransformerBlock


class GPT(nn.Module):
    """Canonical Decoder-Only Transformer Language Model (GPT-2).

    This model processes a 2D tensor of discrete token IDs `(B, T)` and computes
    un-normalized log-probabilities (logits) over the vocabulary `(B, T, V)` for
    next-token prediction.

    Tensor Dimension Notation:
        B: Batch size (number of sequences in batch)
        T: Sequence length (number of time steps, T <= context_length)
        C: Model dimension (config.d_model, e.g. 768 for GPT-2 Small)
        V: Vocabulary size (config.vocab_size, e.g. 50,257 for GPT-2 Small)
        L: Number of layers (config.n_layers, e.g. 12 for GPT-2 Small)

    Full Forward Flow:
        input_ids: (B, T) [dtype: torch.long]
          ├── wte(input_ids): (B, T, C)              [Token Embedding]
          └── wpe(positions): (T, C)                 [Learned Positional Embedding]
          ▼
        Broadcast Addition: x = wte + wpe: (B, T, C)
          ▼
        Embedding Dropout: (B, T, C)
          ▼
        TransformerBlock x L: (B, T, C) -> (B, T, C) [12 Pre-Norm Decoder Blocks]
          ▼
        Final LayerNorm ln_f: (B, T, C)              [Final Normalization]
          ▼
        Language Model Head lm_head: (B, T, V)       [Tied Weight Projection]
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Token Embeddings: maps token ID [0..V-1] to C-dimensional vector
        self.wte = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Learned Absolute Positional Embeddings: maps position [0..context_length-1] to C-dim vector
        self.wpe = nn.Embedding(config.context_length, config.d_model)

        # 3. Post-embedding dropout
        self.drop = nn.Dropout(config.dropout)

        # 4. Sequential stack of L independent Transformer decoder blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # 5. Final LayerNorm before projection to logits
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)

        # 6. Language Model Head: projects hidden state C to vocabulary distribution V
        # Bias is False in standard GPT-2 architecture
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 7. Standard GPT-2 Weight Initialization:
        # Initialize all Linear, Embedding, and LayerNorm modules with N(0, initializer_range^2) and (1.0, 0.0)
        self.apply(self._init_weights)

        # 8. Apply GPT-2 special scaled initialization for residual projection layers:
        # std = initializer_range / sqrt(2 * n_layers) prevents variance accumulation across deep residual streams
        residual_std = self.config.initializer_range / math.sqrt(2 * self.config.n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc_out.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=residual_std)

        # 9. Weight Tying (Weight Sharing):
        # Ties output lm_head weight matrix to input token embedding matrix wte.weight.
        # Calling after initialization ensures single RNG initialization on the shared memory.
        self.tie_weights()

    def tie_weights(self) -> None:
        """Ties the language model head weight to the token embedding weight matrix."""
        self.lm_head.weight = self.wte.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes model weights following the canonical GPT-2 normal distribution specification."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Calculates total unique parameter count of the instantiated model.

        Because self.lm_head.weight is tied to self.wte.weight, PyTorch's parameter
        iterator naturally deduplicates the tied matrix.

        Args:
            non_embedding: If True, excludes token embedding (wte) and positional embedding (wpe).

        Returns:
            Total unique parameter count as integer.
        """
        params = list(self.parameters())
        if non_embedding:
            params = [p for p in params if p is not self.wpe.weight and p is not self.wte.weight]

        return sum(p.numel() for p in params)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the full GPT language model.

        Args:
            input_ids: 2D integer tensor of token IDs with shape (B, T).

        Returns:
            logits: Output logits tensor with shape (B, T, V).

        Raises:
            TypeError: If input_ids is a floating-point tensor.
            ValueError: If input tensor is not 2D or sequence length T exceeds context_length.
        """
        if input_ids.is_floating_point():
            raise TypeError(
                f"Expected integer tensor for input_ids, got floating point dtype {input_ids.dtype}."
            )

        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected 2D input tensor of shape (batch_size, sequence_length), "
                f"got {input_ids.ndim}D tensor with shape {tuple(input_ids.shape)}."
            )

        B, T = input_ids.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Input sequence length T={T} exceeds maximum configured context length {self.config.context_length}."
            )

        # Step 1: Generate position indices [0, 1, ..., T-1] matching input device
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # Step 2: Retrieve embeddings
        # tok_emb: (B, T, C)
        tok_emb = self.wte(input_ids)
        # pos_emb: (T, C)
        pos_emb = self.wpe(positions)

        # Step 3: Broadcast addition (B, T, C) + (T, C) -> (B, T, C) followed by dropout
        x = self.drop(tok_emb + pos_emb)

        # Step 4: Pass through L Transformer decoder blocks
        for block in self.blocks:
            x = block(x)

        # Step 5: Final LayerNorm
        x = self.ln_f(x)

        # Step 6: Language model head projection -> (B, T, V)
        logits = self.lm_head(x)

        return logits

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Forward pass of the GPT language model supporting per-layer Key/Value caching.

        Args:
            input_ids: 2D integer tensor of token IDs with shape (B, T), where T is prompt length (Prefill) or 1 (Decode).
            past_key_values: Optional tuple of length n_layers containing (past_k, past_v) tuples.

        Returns:
            Tuple of (logits, present_key_values) where logits has shape (B, T, V) and
            present_key_values is a tuple of length n_layers containing updated (k, v) cache tensors.

        Raises:
            TypeError: If input_ids is floating-point.
            ValueError: If input_ids is not 2D, or if total sequence length exceeds context_length,
                        or if past_key_values length does not match n_layers.
        """
        if input_ids.is_floating_point():
            raise TypeError(
                f"Expected integer tensor for input_ids, got floating point dtype {input_ids.dtype}."
            )

        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected 2D input tensor of shape (batch_size, sequence_length), "
                f"got {input_ids.ndim}D tensor with shape {tuple(input_ids.shape)}."
            )

        B, T = input_ids.shape

        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(
                    f"Expected past_key_values of length {self.config.n_layers}, got length {len(past_key_values)}."
                )
            past_len = past_key_values[0][0].shape[-2]

        total_len = past_len + T
        if total_len > self.config.context_length:
            raise ValueError(
                f"Total sequence length total_len={total_len} exceeds maximum configured context length {self.config.context_length}."
            )

        # Step 1: Position indices with exact offset: [past_len, past_len + 1, ..., past_len + T - 1]
        positions = torch.arange(past_len, total_len, dtype=torch.long, device=input_ids.device)

        # Step 2: Retrieve embeddings
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(positions)
        x = self.drop(tok_emb + pos_emb)

        # Step 3: Pass through L Transformer decoder blocks with KV cache
        present_key_values = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, layer_present = block.forward_cached(x, past_kv=layer_past)
            present_key_values.append(layer_present)

        # Step 4: Final LayerNorm
        x = self.ln_f(x)

        # Step 5: Language model head projection -> (B, T, V)
        logits = self.lm_head(x)

        return logits, tuple(present_key_values)
