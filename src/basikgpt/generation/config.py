"""Generation configuration dataclass for autoregressive decoding."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Immutable configuration for autoregressive token decoding and sampling.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate (must be > 0).
        temperature: Logits scaling temperature (must be > 0.0). T < 1.0 is sharper, T > 1.0 is flatter.
        top_k: Number of highest probability vocabulary tokens to keep for sampling (None disables top-k).
        top_p: Cumulative probability threshold for nucleus sampling (None disables top-p, 0.0 < top_p <= 1.0).
        do_sample: If True, uses stochastic sampling (temperature/top-k/top-p); if False, uses greedy argmax.
        seed: Optional integer random seed for reproducible sampling.
        stop_on_eot: If True, halts generation when the EOT token is emitted.
        eot_token_id: Token ID marking the end of text (defaults to GPT-2 canonical 50,256).
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False
    seed: int | None = None
    stop_on_eot: bool = True
    eot_token_id: int = 50256

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive (> 0), got {self.max_new_tokens}")

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive (> 0.0), got {self.temperature}")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive (> 0) if specified, got {self.top_k}")

        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in the range (0.0, 1.0], got {self.top_p}")

        if self.eot_token_id < 0:
            raise ValueError(f"eot_token_id must be non-negative (>= 0), got {self.eot_token_id}")
