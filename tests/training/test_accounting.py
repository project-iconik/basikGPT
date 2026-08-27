"""Unit tests for analytical token accounting and budget calculations in basikGPT."""

import pytest
from basikgpt.training.accounting import (
    TokenBudgetPlan,
    calculate_compile_break_even_tokens,
    calculate_eval_batches,
    calculate_tokens_seen,
    calculate_training_steps,
    calculate_warmup_steps,
)


def test_token_accounting_arithmetic() -> None:
    """Verifies that micro-batch and optimizer-step token calculations match theoretical formulas."""
    plan = calculate_training_steps(
        target_tokens=10_000,
        micro_batch_size=2,
        context_length=128,
        grad_accum_steps=4,
        world_size=1,
    )
    assert plan.tokens_per_micro_batch == 256
    assert plan.tokens_per_optimizer_step == 1_024
    assert plan.optimizer_steps == 10
    assert plan.actual_token_budget == 10_240
    assert plan.overshoot_tokens == 240


def test_step_calculation_ceiling_and_overshoot() -> None:
    """Verifies ceiling behavior when target tokens are not an exact multiple of tokens per step."""
    # Not exact multiple: 10,001 / 1024 = 9.766 -> 10 steps (10,240 tokens, +239 overshoot)
    plan_ceil = calculate_training_steps(
        target_tokens=10_001,
        micro_batch_size=2,
        context_length=128,
        grad_accum_steps=4,
        world_size=1,
    )
    assert plan_ceil.optimizer_steps == 10
    assert plan_ceil.actual_token_budget == 10_240
    assert plan_ceil.overshoot_tokens == 239

    # Exact multiple: 10,240 / 1024 = 10 steps (10,240 tokens, +0 overshoot)
    plan_exact = calculate_training_steps(
        target_tokens=10_240,
        micro_batch_size=2,
        context_length=128,
        grad_accum_steps=4,
        world_size=1,
    )
    assert plan_exact.optimizer_steps == 10
    assert plan_exact.actual_token_budget == 10_240
    assert plan_exact.overshoot_tokens == 0


def test_world_size_scaling() -> None:
    """Verifies that distributed world_size scales token capacity proportionally."""
    plan_w1 = calculate_training_steps(target_tokens=100_000, micro_batch_size=4, context_length=128, grad_accum_steps=2, world_size=1)
    plan_w2 = calculate_training_steps(target_tokens=100_000, micro_batch_size=4, context_length=128, grad_accum_steps=2, world_size=2)
    plan_w4 = calculate_training_steps(target_tokens=100_000, micro_batch_size=4, context_length=128, grad_accum_steps=2, world_size=4)

    assert plan_w1.tokens_per_optimizer_step == 1_024
    assert plan_w2.tokens_per_optimizer_step == 2_048
    assert plan_w4.tokens_per_optimizer_step == 4_096

    assert plan_w1.optimizer_steps == 98
    assert plan_w2.optimizer_steps == 49
    assert plan_w4.optimizer_steps == 25


def test_invalid_inputs_raise_errors() -> None:
    """Verifies that non-positive dimensions raise ValueError fail-fast."""
    with pytest.raises(ValueError, match="target_tokens must be positive"):
        calculate_training_steps(0, 2, 64, 2)

    with pytest.raises(ValueError, match="micro_batch_size must be positive"):
        calculate_training_steps(1000, 0, 64, 2)

    with pytest.raises(ValueError, match="context_length must be positive"):
        calculate_training_steps(1000, 2, 0, 2)

    with pytest.raises(ValueError, match="grad_accum_steps must be positive"):
        calculate_training_steps(1000, 2, 64, 0)

    with pytest.raises(ValueError, match="world_size must be positive"):
        calculate_training_steps(1000, 2, 64, 2, world_size=0)


def test_chinchilla_2_5b_planning_arithmetic() -> None:
    """Verifies the analytical calculations for the long-term ~2.5B token target budget."""
    target_tokens = 2_500_000_000

    # Single-device setup: B=4, T=1024, G=8, W=1
    plan_single = calculate_training_steps(
        target_tokens=target_tokens,
        micro_batch_size=4,
        context_length=1024,
        grad_accum_steps=8,
        world_size=1,
    )
    assert plan_single.tokens_per_optimizer_step == 32_768
    assert plan_single.optimizer_steps == 76_294
    assert plan_single.actual_token_budget == 2_500_001_792
    assert plan_single.overshoot_tokens == 1_792

    # 4-device DDP setup: B=4, T=1024, G=8, W=4
    plan_4gpu = calculate_training_steps(
        target_tokens=target_tokens,
        micro_batch_size=4,
        context_length=1024,
        grad_accum_steps=8,
        world_size=4,
    )
    assert plan_4gpu.tokens_per_optimizer_step == 131_072
    assert plan_4gpu.optimizer_steps == 19_074
    assert plan_4gpu.actual_token_budget == 2_500_067_328
    assert plan_4gpu.overshoot_tokens == 67_328


def test_calculate_tokens_seen() -> None:
    """Verifies cumulative token count helper."""
    assert calculate_tokens_seen(optimizer_steps=10, micro_batch_size=2, context_length=64, grad_accum_steps=2, world_size=1) == 2_560
    assert calculate_tokens_seen(optimizer_steps=0, micro_batch_size=2, context_length=64, grad_accum_steps=2) == 0

    with pytest.raises(ValueError):
        calculate_tokens_seen(-1, 2, 64, 2)


def test_compile_break_even_tokens() -> None:
    """N = C * R0 * R1 / (R1 - R0). Compiled-not-faster and invalid inputs are safe."""
    n = calculate_compile_break_even_tokens(
        compile_overhead_seconds=10.0,
        baseline_tokens_per_second=1000.0,
        compiled_tokens_per_second=2000.0,
    )
    assert n == pytest.approx(20_000.0)

    assert (
        calculate_compile_break_even_tokens(
            compile_overhead_seconds=5.0,
            baseline_tokens_per_second=1000.0,
            compiled_tokens_per_second=1000.0,
        )
        is None
    )
    assert (
        calculate_compile_break_even_tokens(
            compile_overhead_seconds=5.0,
            baseline_tokens_per_second=2000.0,
            compiled_tokens_per_second=1000.0,
        )
        is None
    )
    assert (
        calculate_compile_break_even_tokens(
            compile_overhead_seconds=0.0,
            baseline_tokens_per_second=1000.0,
            compiled_tokens_per_second=2000.0,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="compile_overhead_seconds"):
        calculate_compile_break_even_tokens(-1.0, 1000.0, 2000.0)
    with pytest.raises(ValueError, match="baseline_tokens_per_second"):
        calculate_compile_break_even_tokens(1.0, 0.0, 2000.0)
    with pytest.raises(ValueError, match="compiled_tokens_per_second"):
        calculate_compile_break_even_tokens(1.0, 1000.0, -5.0)


def test_eval_batches_require_identical_token_count() -> None:
    """B=8 and B=16 can share 131,072 eval tokens with different batch counts."""
    eval_tokens = 131_072
    batches_a = calculate_eval_batches(eval_tokens, micro_batch_size=8, context_length=1024)
    batches_b = calculate_eval_batches(eval_tokens, micro_batch_size=16, context_length=1024)
    assert batches_a == 16
    assert batches_b == 8
    assert batches_a * 8 * 1024 == batches_b * 16 * 1024 == eval_tokens
    with pytest.raises(ValueError, match="divisible"):
        calculate_eval_batches(1000, micro_batch_size=8, context_length=1024)


def test_warmup_steps_fraction() -> None:
    assert calculate_warmup_steps(16, fraction=0.10) == 2
    assert calculate_warmup_steps(153, fraction=0.10) == 15
    assert calculate_warmup_steps(1, fraction=0.10) == 1
    with pytest.raises(ValueError):
        calculate_warmup_steps(0)
