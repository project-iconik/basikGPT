"""Autoregressive text generation loops (Naive full-prefix and KV Cached decoding) for basikGPT."""

from typing import Any
import torch
import torch.nn as nn
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.sampling import sample_next_token


def generate_naive(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generates continuation tokens by re-evaluating the full prefix at each step (O(N^2) reference baseline).

    Args:
        model: Pretrained GPT model instance.
        input_ids: 2D Tensor of integer token IDs of shape (batch_size, sequence_length).
        config: Optional GenerationConfig instance.

    Returns:
        Tensor of shape (batch_size, prompt_length + num_generated_tokens).

    Raises:
        ValueError: If input_ids is not 2D or prompt length exceeds context length.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids tensor of shape (batch_size, sequence_length), got shape {tuple(input_ids.shape)}")

    gen_cfg = config or GenerationConfig()
    batch_size, prompt_length = input_ids.shape

    # Context Length Guardrails
    model_cfg = getattr(model, "config", None)
    context_length = getattr(model_cfg, "context_length", 1024) if model_cfg else 1024

    if prompt_length > context_length:
        raise ValueError(
            f"Prompt length ({prompt_length}) exceeds model maximum context_length ({context_length})"
        )

    effective_max_new_tokens = min(gen_cfg.max_new_tokens, context_length - prompt_length)
    if effective_max_new_tokens <= 0:
        return input_ids

    # Isolated Generator for Reproducibility
    generator: torch.Generator | None = None
    if gen_cfg.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(gen_cfg.seed)

    was_training = model.training
    model.eval()

    curr_ids = input_ids.clone()

    with torch.inference_mode():
        for _ in range(effective_max_new_tokens):
            logits = model(curr_ids)
            next_token_logits = logits[:, -1, :]
            next_token = sample_next_token(next_token_logits, gen_cfg, generator=generator)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if gen_cfg.stop_on_eot and (next_token == gen_cfg.eot_token_id).all():
                break

    if was_training:
        model.train()

    return curr_ids


def generate_cached(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generates continuation tokens using Key-Value Caching to avoid recomputing past token projections.

    Execution Flow:
        1. Prefill Phase: Forwards full prompt sequence to build initial per-layer K/V caches.
        2. Decode Phase: Forwards only the newest single token (1, 1) per step, appending new K/V to cache.

    Args:
        model: Pretrained GPT model instance supporting forward_cached().
        input_ids: 2D Tensor of integer token IDs of shape (batch_size, sequence_length).
        config: Optional GenerationConfig instance.

    Returns:
        Tensor of shape (batch_size, prompt_length + num_generated_tokens).

    Raises:
        ValueError: If input_ids is not 2D or prompt length exceeds context length.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids tensor of shape (batch_size, sequence_length), got shape {tuple(input_ids.shape)}")

    gen_cfg = config or GenerationConfig()
    batch_size, prompt_length = input_ids.shape

    # Context Length Guardrails
    model_cfg = getattr(model, "config", None)
    context_length = getattr(model_cfg, "context_length", 1024) if model_cfg else 1024

    if prompt_length > context_length:
        raise ValueError(
            f"Prompt length ({prompt_length}) exceeds model maximum context_length ({context_length})"
        )

    effective_max_new_tokens = min(gen_cfg.max_new_tokens, context_length - prompt_length)
    if effective_max_new_tokens <= 0:
        return input_ids

    # Isolated Generator for Reproducibility
    generator: torch.Generator | None = None
    if gen_cfg.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(gen_cfg.seed)

    was_training = model.training
    model.eval()

    curr_ids = input_ids.clone()

    with torch.inference_mode():
        # Phase 1: Prompt Prefill
        logits, past_key_values = model.forward_cached(curr_ids, past_key_values=None)
        next_token_logits = logits[:, -1, :]
        next_token = sample_next_token(next_token_logits, gen_cfg, generator=generator)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        # Phase 2: Single-Token Cached Decode Loop
        if not (gen_cfg.stop_on_eot and (next_token == gen_cfg.eot_token_id).all()):
            for _ in range(1, effective_max_new_tokens):
                logits, past_key_values = model.forward_cached(next_token, past_key_values=past_key_values)
                next_token_logits = logits[:, -1, :]
                next_token = sample_next_token(next_token_logits, gen_cfg, generator=generator)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)

                if gen_cfg.stop_on_eot and (next_token == gen_cfg.eot_token_id).all():
                    break

    if was_training:
        model.train()

    return curr_ids


def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Unified autoregressive text generation entrypoint for basikGPT.

    Args:
        model: Pretrained GPT model instance.
        input_ids: 2D Tensor of integer token IDs of shape (batch_size, sequence_length).
        config: Optional GenerationConfig instance.
        use_cache: If True, uses fast Key-Value Caching (generate_cached);
                   if False, uses naive full-prefix recomputation (generate_naive).

    Returns:
        Tensor of shape (batch_size, prompt_length + num_generated_tokens).
    """
    if use_cache and hasattr(model, "forward_cached"):
        return generate_cached(model=model, input_ids=input_ids, config=config)
    return generate_naive(model=model, input_ids=input_ids, config=config)
