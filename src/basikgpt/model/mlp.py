"""Feed-Forward Multi-Layer Perceptron (MLP) module for basikGPT.

Implements the standard GPT-2 two-layer position-wise feed-forward network with
tanh-approximated GELU non-linear activation.
"""

import torch
import torch.nn as nn

from basikgpt.config import GPTConfig


class MLP(nn.Module):
    """Position-wise Feed-Forward Network (MLP) for GPT-2.

    Applies two linear transformations with a non-linear GELU activation in between,
    operating independently and identically on each token position in the sequence:

        MLP(x) = Dropout(Linear_2(GELU_tanh(Linear_1(x))))

    Tensor Dimension Notation:
        B: Batch size (number of sequences in batch)
        T: Sequence length (number of time steps)
        C: Model / hidden dimension (config.d_model, e.g., 768 for GPT-2 Small)
        F: Feed-forward inner dimension (config.d_ff, e.g., 3072 for GPT-2 Small = 4 * C)

    Architecture Progression:
        x: (B, T, C)
          -> fc_in: (B, T, F)
          -> GELU(approximate="tanh"): (B, T, F)
          -> fc_out: (B, T, C)
          -> Dropout: (B, T, C)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        # First linear projection expanding hidden dimension from C to F (e.g. 768 -> 3072)
        self.fc_in = nn.Linear(config.d_model, config.d_ff, bias=config.bias)

        # GPT-2 uses the tanh approximation of GELU:
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        self.activation = nn.GELU(approximate="tanh")

        # Second linear projection contracting hidden dimension back from F to C (e.g. 3072 -> 768)
        self.fc_out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

        # Output dropout applied before residual addition in the TransformerBlock
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP module.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor of shape (B, T, C).
        """
        # Step 1: Up-projection: (B, T, C) -> (B, T, F)
        h = self.fc_in(x)

        # Step 2: Non-linear activation: (B, T, F) -> (B, T, F)
        h = self.activation(h)

        # Step 3: Down-projection: (B, T, F) -> (B, T, C)
        out = self.fc_out(h)

        # Step 4: Dropout: (B, T, C) -> (B, T, C)
        out = self.dropout(out)

        return out
