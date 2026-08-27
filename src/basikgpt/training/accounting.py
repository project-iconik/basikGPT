"""Analytical token accounting, step calculation, and budget planning for basikGPT pretraining.

Provides exact token arithmetic for micro-batches, gradient accumulation, distributed world size,
and token-budget-driven optimizer step calculations.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenBudgetPlan:
    """Analytical specification and step breakdown for a target pretraining token budget.

    Attributes:
        requested_token_budget: User-requested nominal token count target.
        micro_batch_size: Number of sequences per forward/backward micro-step (B).
        context_length: Sequence length in tokens (T).
        grad_accum_steps: Number of micro-batches accumulated per optimizer step (G).
        world_size: Number of distributed data-parallel workers (W, default: 1).
        tokens_per_micro_batch: Total tokens processed in a single micro-batch (B * T).
        tokens_per_optimizer_step: Total tokens processed across all workers per optimizer step (B * T * G * W).
        optimizer_steps: Number of global optimizer steps required to reach or exceed target budget.
        actual_token_budget: Actual total tokens that will be processed across all planned steps.
        overshoot_tokens: Difference between actual processed tokens and requested budget (actual - requested).
    """

    requested_token_budget: int
    micro_batch_size: int
    context_length: int
    grad_accum_steps: int
    world_size: int
    tokens_per_micro_batch: int
    tokens_per_optimizer_step: int
    optimizer_steps: int
    actual_token_budget: int
    overshoot_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes budget plan into a dictionary."""
        return asdict(self)


def calculate_training_steps(
    target_tokens: int,
    micro_batch_size: int,
    context_length: int,
    grad_accum_steps: int,
    world_size: int = 1,
) -> TokenBudgetPlan:
    """Calculates the exact number of global optimizer steps required for a target token budget.

    Token Accounting Formulation:
        1. Tokens per micro-batch:
           tokens_per_micro_batch = B * T
        2. Tokens per optimizer step:
           tokens_per_optimizer_step = B * T * G * W
        3. Global optimizer steps (Ceiling Policy):
           optimizer_steps = ceil(target_tokens / tokens_per_optimizer_step)
        4. Actual processed token count:
           actual_token_budget = optimizer_steps * tokens_per_optimizer_step
        5. Overshoot tokens:
           overshoot_tokens = actual_token_budget - target_tokens

    Args:
        target_tokens: Total desired nominal training token budget.
        micro_batch_size: Micro-batch size per device (B).
        context_length: Sequence length / block size (T).
        grad_accum_steps: Gradient accumulation steps (G).
        world_size: Number of distributed workers (W, default: 1).

    Returns:
        TokenBudgetPlan instance containing complete step arithmetic and breakdown.

    Raises:
        ValueError: If any input argument is non-positive (<= 0).
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    tokens_per_micro_batch = micro_batch_size * context_length
    tokens_per_optimizer_step = tokens_per_micro_batch * grad_accum_steps * world_size

    # Ceiling policy ensures we never terminate below the requested target token budget
    optimizer_steps = math.ceil(target_tokens / tokens_per_optimizer_step)
    actual_token_budget = optimizer_steps * tokens_per_optimizer_step
    overshoot_tokens = actual_token_budget - target_tokens

    return TokenBudgetPlan(
        requested_token_budget=target_tokens,
        micro_batch_size=micro_batch_size,
        context_length=context_length,
        grad_accum_steps=grad_accum_steps,
        world_size=world_size,
        tokens_per_micro_batch=tokens_per_micro_batch,
        tokens_per_optimizer_step=tokens_per_optimizer_step,
        optimizer_steps=optimizer_steps,
        actual_token_budget=actual_token_budget,
        overshoot_tokens=overshoot_tokens,
    )


def calculate_eval_batches(
    eval_tokens: int,
    micro_batch_size: int,
    context_length: int,
) -> int:
    """Number of validation micro-batches that cover exactly `eval_tokens`.

    Requires `eval_tokens` to be divisible by `B * T` so Candidate A/B can share
    the same evaluated token count with different micro-batch sizes.
    """
    if eval_tokens <= 0:
        raise ValueError(f"eval_tokens must be positive, got {eval_tokens}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    tokens_per_batch = micro_batch_size * context_length
    if eval_tokens % tokens_per_batch != 0:
        raise ValueError(
            f"eval_tokens ({eval_tokens}) must be divisible by B*T "
            f"({micro_batch_size}*{context_length}={tokens_per_batch}) so candidates "
            "evaluate an identical token count."
        )
    return eval_tokens // tokens_per_batch


def calculate_warmup_steps(max_steps: int, fraction: float = 0.10) -> int:
    """Linear-warmup step count as a fraction of optimizer steps, at least 1."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    steps = round(fraction * max_steps)
    return max(1, int(steps))


def calculate_tokens_seen(
    optimizer_steps: int,
    micro_batch_size: int,
    context_length: int,
    grad_accum_steps: int,
    world_size: int = 1,
) -> int:
    """Calculates cumulative tokens processed after a given number of optimizer steps."""
    if optimizer_steps < 0:
        raise ValueError(f"optimizer_steps must be non-negative, got {optimizer_steps}")
    if micro_batch_size <= 0 or context_length <= 0 or grad_accum_steps <= 0 or world_size <= 0:
        raise ValueError("Batching dimensions must all be positive integers.")

    return optimizer_steps * micro_batch_size * context_length * grad_accum_steps * world_size


def calculate_compile_break_even_tokens(
    compile_overhead_seconds: float,
    baseline_tokens_per_second: float,
    compiled_tokens_per_second: float,
) -> float | None:
    """Tokens needed for compiled throughput to recover compile overhead C.

    Let C be compile overhead in seconds, R0 baseline tokens/sec, R1 compiled
    tokens/sec. Times to process N tokens:

        T0 = N / R0
        T1 = C + N / R1

    Break-even T1 = T0 yields:

        N = C * R0 * R1 / (R1 - R0)

    Returns None when compiled is not faster (R1 <= R0), including equal rates.
    Zero overhead with a faster compiled path returns 0.0.

    Raises:
        ValueError: If overhead is negative or either throughput is not positive.
    """
    if compile_overhead_seconds < 0:
        raise ValueError(
            f"compile_overhead_seconds must be non-negative, got {compile_overhead_seconds}"
        )
    if baseline_tokens_per_second <= 0:
        raise ValueError(
            f"baseline_tokens_per_second must be positive, got {baseline_tokens_per_second}"
        )
    if compiled_tokens_per_second <= 0:
        raise ValueError(
            f"compiled_tokens_per_second must be positive, got {compiled_tokens_per_second}"
        )
    if compiled_tokens_per_second <= baseline_tokens_per_second:
        return None
    if compile_overhead_seconds == 0:
        return 0.0
    return (
        compile_overhead_seconds
        * baseline_tokens_per_second
        * compiled_tokens_per_second
        / (compiled_tokens_per_second - baseline_tokens_per_second)
    )
