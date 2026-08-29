"""Unit tests for linear warmup and cosine decay learning rate scheduler."""

import pytest
import torch
import torch.nn as nn
from basikgpt.training.config import TrainingConfig
from basikgpt.training.scheduler import get_learning_rate_at_step, update_learning_rate


def test_scheduler_curve_and_boundaries() -> None:
    """Verifies that LR schedule adheres strictly to warmup, peak, cosine decay, and min-LR floor."""
    lr = 1e-3
    min_lr = 1e-4
    warmup_steps = 100
    max_steps = 1000

    cfg = TrainingConfig(
        learning_rate=lr,
        min_learning_rate=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )

    # 1. Start of warmup (step 0): LR > 0 and <= peak/warmup
    lr_0 = get_learning_rate_at_step(0, cfg)
    assert lr_0 == pytest.approx(lr / warmup_steps)

    # 2. Mid-warmup monotonicity
    lr_mid_warmup = get_learning_rate_at_step(warmup_steps // 2, cfg)
    assert lr_0 < lr_mid_warmup < lr

    # 3. End of warmup (step == warmup_steps - 1) reaches peak lr
    lr_peak = get_learning_rate_at_step(warmup_steps - 1, cfg)
    assert lr_peak == pytest.approx(lr)

    # 4. At warmup_steps (cosine decay start): LR is peak lr
    lr_decay_start = get_learning_rate_at_step(warmup_steps, cfg)
    assert lr_decay_start == pytest.approx(lr)

    # 5. Mid decay: strictly between min_lr and peak lr
    lr_mid_decay = get_learning_rate_at_step((warmup_steps + max_steps) // 2, cfg)
    assert min_lr < lr_mid_decay < lr

    # 6. At max_steps: exactly min_lr
    lr_end = get_learning_rate_at_step(max_steps, cfg)
    assert lr_end == pytest.approx(min_lr)

    # 7. Beyond max_steps: stays at min_lr floor
    lr_beyond = get_learning_rate_at_step(max_steps + 500, cfg)
    assert lr_beyond == pytest.approx(min_lr)


def test_continuation_schedule_rewarm_then_cosine() -> None:
    """Continuation LR starts at min_lr, re-warms to peak, then cosine to min_lr at max_steps."""
    origin = 38_147
    warmup = 1_000
    max_steps = 76_294
    peak = 3e-4
    min_lr = 6e-5
    cfg = TrainingConfig(
        learning_rate=peak,
        min_learning_rate=min_lr,
        warmup_steps=warmup,
        max_steps=max_steps,
        schedule_origin_step=origin,
    )

    assert get_learning_rate_at_step(origin - 1, cfg) == pytest.approx(min_lr)
    lr_first = get_learning_rate_at_step(origin, cfg)
    assert lr_first == pytest.approx(min_lr + (peak - min_lr) / warmup)
    lr_peak = get_learning_rate_at_step(origin + warmup - 1, cfg)
    assert lr_peak == pytest.approx(peak)
    lr_cosine_start = get_learning_rate_at_step(origin + warmup, cfg)
    assert lr_cosine_start == pytest.approx(peak)
    lr_mid = get_learning_rate_at_step((origin + warmup + max_steps) // 2, cfg)
    assert min_lr < lr_mid < peak
    assert get_learning_rate_at_step(max_steps, cfg) == pytest.approx(min_lr)
    assert get_learning_rate_at_step(max_steps + 10, cfg) == pytest.approx(min_lr)


def test_update_learning_rate_on_optimizer() -> None:
    """Verifies that update_learning_rate mutates all param_groups in the optimizer."""
    p = nn.Parameter(torch.randn(2, 2))
    opt = torch.optim.SGD([{"params": [p], "lr": 0.5}], lr=0.5)

    update_learning_rate(opt, 0.00123)
    for group in opt.param_groups:
        assert group["lr"] == 0.00123
