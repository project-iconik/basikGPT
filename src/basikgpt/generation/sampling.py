"""Token sampling, logits filtering, temperature scaling, top-k, and nucleus (top-p) functions."""

import torch
from basikgpt.generation.config import GenerationConfig


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Selects the next token ID from next-position logits according to GenerationConfig.

    Filtering & Selection Flow:
        1. If not do_sample: return greedy argmax across vocabulary.
        2. If do_sample:
           a. Temperature scaling: logits / temperature
           b. Top-k filtering: mask tokens below the k-th highest logit to -inf
           c. Top-p (nucleus) filtering: mask tail tokens where cumulative probability > top_p to -inf
           d. Softmax normalization over remaining tokens
           e. Categorical sampling via torch.multinomial

    Args:
        logits: 2D Tensor of raw logits of shape (batch_size, vocab_size).
        config: GenerationConfig instance specifying sampling hyperparameters.
        generator: Optional torch.Generator for deterministic/isolated RNG.

    Returns:
        Tensor of selected token IDs with shape (batch_size, 1).
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor of shape (batch_size, vocab_size), got shape {tuple(logits.shape)}")

    # 1. Greedy Decoding (Deterministic Argmax)
    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature Scaling
    scaled_logits = logits.clone() / config.temperature

    # 3. Top-k Filtering
    if config.top_k is not None and config.top_k > 0:
        k = min(config.top_k, scaled_logits.size(-1))
        topk_vals, _ = torch.topk(scaled_logits, k, dim=-1)
        kth_threshold = topk_vals[..., -1, None]
        scaled_logits = torch.where(scaled_logits < kth_threshold, torch.tensor(-float("Inf"), device=scaled_logits.device), scaled_logits)

    # 4. Top-p (Nucleus) Filtering
    if config.top_p is not None and config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > config.top_p

        # Shift indices to always keep the first token that exceeds the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter mask back to original logit positions
        indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled_logits = torch.where(indices_to_remove, torch.tensor(-float("Inf"), device=scaled_logits.device), scaled_logits)

    # 5. Softmax & Categorical Sampling
    probs = torch.softmax(scaled_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_tokens
