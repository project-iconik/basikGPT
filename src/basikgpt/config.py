"""Model configuration dataclass for basikGPT.

Defines architectural hyperparameters, validation rules, and presets for GPT-2 Small
and related decoder-only Transformer configurations.
"""

from dataclasses import dataclass
from typing import Literal, Self

type AttentionBackend = Literal["eager", "sdpa"]


@dataclass(frozen=True, slots=True)
class GPTConfig:
    """Configuration specification for basikGPT decoder-only Transformer.

    Attributes:
        vocab_size: Total vocabulary size of the tokenizer (GPT-2 default: 50,257).
        context_length: Maximum context sequence length / block size (GPT-2 default: 1,024).
        n_layers: Number of Transformer decoder layers (GPT-2 Small: 12).
        n_heads: Number of parallel attention heads (GPT-2 Small: 12).
        d_model: Hidden embedding dimension (GPT-2 Small: 768).
        d_ff: Inner dimension of the feed-forward MLP network (GPT-2 Small: 3,072 = 4 * 768).
        dropout: Dropout probability applied to residual connections, embeddings, and attention (default: 0.1).
        layer_norm_eps: Small epsilon constant for LayerNorm numerical stability (default: 1e-5).
        bias: Whether to use learnable additive bias in Linear projections and LayerNorms (GPT-2 default: True).
        attention_backend: Attention computation backend to use ("eager" for educational/reference, "sdpa" for fast training).
        initializer_range: Standard deviation of the normal distribution for weight initialization (GPT-2 default: 0.02).
    """

    vocab_size: int = 50_257
    context_length: int = 1_024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3_072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    bias: bool = True
    attention_backend: AttentionBackend = "eager"
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        """Validates configuration parameters to prevent invalid model architectures."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive integer, got {self.vocab_size}.")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be a positive integer, got {self.context_length}.")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be a positive integer, got {self.n_layers}.")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be a positive integer, got {self.n_heads}.")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {self.d_model}.")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be a positive integer, got {self.d_ff}.")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}), "
                f"but remainder is {self.d_model % self.n_heads}."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in the half-open interval [0.0, 1.0), got {self.dropout}.")
        if self.layer_norm_eps <= 0:
            raise ValueError(f"layer_norm_eps must be strictly positive, got {self.layer_norm_eps}.")
        if self.attention_backend not in ("eager", "sdpa"):
            raise ValueError(
                f"attention_backend must be either 'eager' or 'sdpa', got '{self.attention_backend}'."
            )
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be strictly positive, got {self.initializer_range}."
            )

    @property
    def head_dim(self) -> int:
        """Calculates dimension of each individual attention head: d_model // n_heads."""
        return self.d_model // self.n_heads

    def num_embedding_parameters(self) -> int:
        """Calculates total unique parameters in Token Embedding (wte) and Positional Embedding (wpe).

        - Token Embedding (wte): vocab_size * d_model
        - Positional Embedding (wpe): context_length * d_model
        """
        return (self.vocab_size + self.context_length) * self.d_model

    def num_transformer_parameters(self) -> int:
        """Calculates total parameters in all Transformer blocks and final LayerNorm.

        Per Transformer block:
        - ln_1 (LayerNorm): d_model * (2 if bias else 1)
        - c_attn (QKV linear projection): d_model * (3 * d_model) + ((3 * d_model) if bias else 0)
        - c_proj (Attn output projection): d_model * d_model + (d_model if bias else 0)
        - ln_2 (LayerNorm): d_model * (2 if bias else 1)
        - c_fc (MLP expansion): d_model * d_ff + (d_ff if bias else 0)
        - c_proj (MLP contraction): d_ff * d_model + (d_model if bias else 0)

        Final LayerNorm:
        - ln_f: d_model * (2 if bias else 1)
        """
        bias_multiplier = 2 if self.bias else 1

        # LayerNorm parameters (weights + optional biases)
        ln_params = self.d_model * bias_multiplier

        # Attention parameters (QKV projection + output projection)
        qkv_weights = self.d_model * (3 * self.d_model)
        qkv_biases = (3 * self.d_model) if self.bias else 0
        attn_out_weights = self.d_model * self.d_model
        attn_out_biases = self.d_model if self.bias else 0
        attn_params = qkv_weights + qkv_biases + attn_out_weights + attn_out_biases

        # MLP parameters (expansion + contraction)
        mlp_fc_weights = self.d_model * self.d_ff
        mlp_fc_biases = self.d_ff if self.bias else 0
        mlp_proj_weights = self.d_ff * self.d_model
        mlp_proj_biases = self.d_model if self.bias else 0
        mlp_params = mlp_fc_weights + mlp_fc_biases + mlp_proj_weights + mlp_proj_biases

        # Per-block total
        per_block_params = ln_params + attn_params + ln_params + mlp_params

        # Final LayerNorm
        final_ln_params = self.d_model * bias_multiplier

        return (self.n_layers * per_block_params) + final_ln_params

    def num_total_parameters(self, tied_weights: bool = True) -> int:
        """Calculates total unique parameter count analytically without GPU memory allocation.

        Args:
            tied_weights: If True, LM Head shares weight matrix with Token Embedding (standard GPT-2).
                         If False, LM Head adds an independent weight matrix of shape (vocab_size, d_model).
        """
        base = self.num_embedding_parameters() + self.num_transformer_parameters()
        if tied_weights:
            return base
        return base + (self.vocab_size * self.d_model)

    def to_dict(self) -> dict[str, object]:
        """Serializes configuration hyperparameters into a JSON-serializable dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "bias": self.bias,
            "attention_backend": self.attention_backend,
            "initializer_range": self.initializer_range,
        }

    @classmethod
    def gpt2_small(cls, **kwargs) -> Self:
        """Canonical GPT-2 Small preset (~124M parameters)."""
        defaults = dict(
            vocab_size=50_257,
            context_length=1_024,
            n_layers=12,
            n_heads=12,
            d_model=768,
            d_ff=3_072,
            dropout=0.1,
            layer_norm_eps=1e-5,
            bias=True,
            attention_backend="eager",
            initializer_range=0.02,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gpt2_medium(cls, **kwargs) -> Self:
        """GPT-2 Medium preset (~350M parameters)."""
        defaults = dict(
            vocab_size=50_257,
            context_length=1_024,
            n_layers=24,
            n_heads=16,
            d_model=1_024,
            d_ff=4_096,
            dropout=0.1,
            layer_norm_eps=1e-5,
            bias=True,
            attention_backend="eager",
            initializer_range=0.02,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gpt2_large(cls, **kwargs) -> Self:
        """GPT-2 Large preset (~774M parameters)."""
        defaults = dict(
            vocab_size=50_257,
            context_length=1_024,
            n_layers=36,
            n_heads=20,
            d_model=1_280,
            d_ff=5_120,
            dropout=0.1,
            layer_norm_eps=1e-5,
            bias=True,
            attention_backend="eager",
            initializer_range=0.02,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def gpt2_xl(cls, **kwargs) -> Self:
        """GPT-2 XL preset (~1.5B parameters)."""
        defaults = dict(
            vocab_size=50_257,
            context_length=1_024,
            n_layers=48,
            n_heads=25,
            d_model=1_600,
            d_ff=6_400,
            dropout=0.1,
            layer_norm_eps=1e-5,
            bias=True,
            attention_backend="eager",
            initializer_range=0.02,
        )
        defaults.update(kwargs)
        return cls(**defaults)
