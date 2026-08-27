"""Full GPT-2 Decoder-Only Transformer Model for basikGPT.

Assembles Token & Positional Embeddings, Transformer Decoder Blocks,
Final LayerNorm, and the Language Model Head with Weight Tying.
"""

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
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # 6. Language Model Head: projects hidden state C to vocabulary distribution V
        # Bias is False in standard GPT-2 architecture
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 7. Weight Tying (Weight Sharing):
        # Ties the output lm_head weight matrix to the input token embedding matrix wte.weight.
        # This reduces parameter count by V * C and shares semantic representations.
        self.lm_head.weight = self.wte.weight

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
            ValueError: If input tensor is not 2D or sequence length T exceeds context_length.
        """
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
