"""Learning rate scheduling with linear warmup and cosine decay."""

import math
import torch.optim
from basikgpt.training.config import TrainingConfig


def get_learning_rate_at_step(
    step: int,
    config: TrainingConfig,
) -> float:
    """Computes the target learning rate at a given optimizer step index.

    Schedule Phases:
        1. Linear Warmup (step < warmup_steps):
           Linearly scales learning rate from ~0 up to `config.learning_rate`.
        2. Cosine Decay (warmup_steps <= step <= max_steps):
           Decays learning rate following a cosine curve down to `config.min_learning_rate`.
        3. Minimum LR Floor (step > max_steps):
           Maintains constant `config.min_learning_rate`.

    Args:
        step: Current global optimizer step index (0-indexed).
        config: TrainingConfig instance containing schedule parameters.

    Returns:
        Learning rate float value for the current step.
    """
    if step < 0:
        raise ValueError(f"Step index must be non-negative, got {step}")

    # Phase 1: Linear Warmup
    if step < config.warmup_steps:
        if config.warmup_steps == 0:
            return config.learning_rate
        # For step 0..warmup_steps-1: (step + 1) / warmup_steps reaches 1.0 at step == warmup_steps - 1
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Phase 3: Post-max_steps floor
    if step >= config.max_steps:
        return config.min_learning_rate

    # Phase 2: Cosine Decay
    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps <= 0:
        return config.min_learning_rate

    decay_ratio = (step - config.warmup_steps) / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    """Updates the learning rate across all parameter groups in the optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
