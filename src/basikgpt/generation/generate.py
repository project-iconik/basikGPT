"""Autoregressive text generation loop for basikGPT."""

from typing import Any
import torch
import torch.nn as nn
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.sampling import sample_next_token


def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generates a continuation token sequence autoregressively from a given input prompt.

    At each decoding step:
        1. Forwards the full prefix sequence through the model: `logits = model(input_ids)`
        2. Extracts the logits of the last token position: `next_token_logits = logits[:, -1, :]`
        3. Samples the next token ID via `sample_next_token` (greedy or temperature/top-k/top-p)
        4. Appends the new token ID to the sequence: `input_ids = cat([input_ids, next_token], dim=1)`
        5. Checks for early termination on canonical EOT token if `stop_on_eot=True`.

    Args:
        model: Pretrained GPT model instance.
        input_ids: 2D Tensor of integer token IDs of shape (batch_size, sequence_length).
        config: Optional GenerationConfig (uses default greedy config if None).

    Returns:
        Tensor of shape (batch_size, prompt_length + num_generated_tokens) containing the full sequence.

    Raises:
        ValueError: If input_ids is not 2D, or if prompt length exceeds model context length.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"Expected 2D input_ids tensor of shape (batch_size, sequence_length), got shape {tuple(input_ids.shape)}")

    gen_cfg = config or GenerationConfig()
    batch_size, prompt_length = input_ids.shape

    # 1. Model Context Length Guardrails
    model_cfg = getattr(model, "config", None)
    context_length = getattr(model_cfg, "context_length", 1024) if model_cfg else 1024

    if prompt_length > context_length:
        raise ValueError(
            f"Prompt length ({prompt_length}) exceeds model maximum context_length ({context_length})"
        )

    effective_max_new_tokens = min(gen_cfg.max_new_tokens, context_length - prompt_length)
    if effective_max_new_tokens <= 0:
        return input_ids

    # 2. Generator Setup for Isolated Reproducibility
    generator: torch.Generator | None = None
    if gen_cfg.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(gen_cfg.seed)

    # 3. Autoregressive Loop
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
