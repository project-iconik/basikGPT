"""Unit tests for AdamW optimizer parameter grouping and weight tying invariants."""

import pytest
import torch
from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.optimizer import configure_optimizers


def test_optimizer_parameter_grouping_invariants() -> None:
    """Verifies that all parameters are cleanly partitioned into decay vs no-decay groups."""
    cfg = GPTConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
    )
    model = GPT(cfg)
    train_cfg = TrainingConfig(weight_decay=0.1, learning_rate=1e-3)

    optimizer = configure_optimizers(model, train_cfg)

    # 1. Exactly 2 parameter groups
    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decay_params = set(decay_group["params"])
    no_decay_params = set(no_decay_group["params"])

    # 2. Invariant: Disjoint sets (decay ∩ no_decay = ∅)
    assert decay_params.isdisjoint(no_decay_params)

    # 3. Invariant: Completeness (decay ∪ no_decay = all unique parameters)
    all_unique_params = set(model.parameters())
    assert (decay_params | no_decay_params) == all_unique_params

    # 4. Invariant: Weight tying single registration
    # lm_head.weight is wte.weight -> should be in decay_params exactly once
    assert model.wte.weight in decay_params
    assert sum(p is model.wte.weight for p in decay_group["params"]) == 1
    assert sum(p is model.wte.weight for p in no_decay_group["params"]) == 0

    # 5. Dimensionality checks
    for p in decay_group["params"]:
        assert p.dim() >= 2, f"Expected 2D+ parameter in decay group, got shape {p.shape}"
    for p in no_decay_group["params"]:
        assert p.dim() < 2, f"Expected 1D parameter in no-decay group, got shape {p.shape}"


def test_optimizer_zero_grad_set_to_none() -> None:
    """Verifies that zero_grad(set_to_none=True) deallocates gradient buffers."""
    cfg = GPTConfig(
        vocab_size=64,
        context_length=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
    )
    model = GPT(cfg)
    optimizer = configure_optimizers(model, TrainingConfig())

    input_ids = torch.randint(0, 64, (2, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Gradients populated
    assert model.wte.weight.grad is not None

    optimizer.zero_grad(set_to_none=True)
    assert model.wte.weight.grad is None
