"""Learning rate scheduling with linear warmup and cosine decay."""

import math
import torch.optim
from basikgpt.training.config import TrainingConfig


def get_learning_rate_at_step(
    step: int,
    config: TrainingConfig,
) -> float:
    """Computes the target learning rate at a given optimizer step index.

    Default schedule (schedule_origin_step is None):
        1. Linear warmup (step < warmup_steps) from ~0 up to `learning_rate`.
        2. Cosine decay down to `min_learning_rate` through `max_steps`.
        3. Floor at `min_learning_rate` after `max_steps`.

    Continuation schedule (`schedule_origin_step` set):
        Warmup is relative to the origin. Linear from `min_learning_rate` to
        `learning_rate` over `warmup_steps`, then cosine to `min_learning_rate`
        at `max_steps`. Steps before the origin stay at the floor.

    Args:
        step: Current global optimizer step index (0-indexed).
        config: TrainingConfig instance containing schedule parameters.

    Returns:
        Learning rate float value for the current step.
    """
    if step < 0:
        raise ValueError(f"Step index must be non-negative, got {step}")

    origin = config.schedule_origin_step
    if origin:
        return _continuation_learning_rate(step, config, origin)
    return _from_scratch_learning_rate(step, config)


def _from_scratch_learning_rate(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        if config.warmup_steps == 0:
            return config.learning_rate
        # For step 0..warmup_steps-1: (step + 1) / warmup_steps reaches 1.0 at step == warmup_steps - 1
        return config.learning_rate * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.min_learning_rate

    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps <= 0:
        return config.min_learning_rate

    decay_ratio = (step - config.warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def _continuation_learning_rate(step: int, config: TrainingConfig, origin: int) -> float:
    if step < origin:
        return config.min_learning_rate

    warmup_end = origin + config.warmup_steps
    if step < warmup_end:
        if config.warmup_steps == 0:
            return config.learning_rate
        local = step - origin
        frac = (local + 1) / config.warmup_steps
        return config.min_learning_rate + frac * (config.learning_rate - config.min_learning_rate)

    if step >= config.max_steps:
        return config.min_learning_rate

    decay_steps = config.max_steps - warmup_end
    if decay_steps <= 0:
        return config.min_learning_rate

    decay_ratio = (step - warmup_end) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    """Updates the learning rate across all parameter groups in the optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
