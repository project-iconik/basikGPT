"""Integration tests for baseline Trainer, tiny-batch overfitting, and accumulation equivalence."""

import math
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.trainer import Trainer


def make_tiny_model_and_data(
    vocab_size: int = 64,
    context_length: int = 16,
    num_samples: int = 4,
    seed: int = 42,
) -> tuple[GPT, TensorDataset, TensorDataset]:
    """Helper creating a miniature GPT model and synthetic deterministic TensorDatasets."""
    torch.manual_seed(seed)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        attention_backend="eager",
    )
    model = GPT(cfg)

    # Autoregressive tokens where y is strictly shifted by 1
    raw_tokens = torch.randint(0, vocab_size, (num_samples, context_length + 1), dtype=torch.long)
    x = raw_tokens[:, :-1]
    y = raw_tokens[:, 1:]

    train_ds = TensorDataset(x, y)
    val_ds = TensorDataset(x[:2], y[:2])
    return model, train_ds, val_ds


def test_tiny_batch_overfit(tmp_path: Path) -> None:
    """Verifies that the training engine successfully drives down training loss on a tiny batch.

    Validates the entire end-to-end forward -> loss -> backward -> clip -> optimizer -> update cycle.
    """
    model, train_ds, val_ds = make_tiny_model_and_data(vocab_size=64, context_length=16, num_samples=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    train_cfg = TrainingConfig(
        learning_rate=1e-2,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup_steps=5,
        max_steps=60,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=30,
        eval_batches=1,
        checkpoint_interval=30,
        log_interval=1,
        output_dir=str(tmp_path),
        device="cpu",
    )

    trainer = Trainer(model, train_cfg, train_loader, val_loader)
    history = trainer.train()

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]

    # Initial loss for uniform distribution over 64 tokens is -ln(1/64) ≈ 4.16
    expected_initial = -math.log(1.0 / 64)
    assert abs(initial_loss - expected_initial) < 1.0, f"Expected initial loss near {expected_initial:.2f}, got {initial_loss:.2f}"

    # Overfitting criterion: Final loss must reduce by at least 80% (final_loss < 0.20 * initial_loss)
    assert final_loss < 0.20 * initial_loss, f"Loss did not decrease sufficiently: initial={initial_loss:.4f}, final={final_loss:.4f}"
    print(f"\n[Overfit Test] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Reduction: {(1 - final_loss/initial_loss)*100:.1f}%)")


def test_gradient_accumulation_equivalence() -> None:
    """Verifies that 1 step of batch_size=4 matches 2 micro-steps of batch_size=2 with grad_accum=2."""
    vocab_size, context_length = 64, 16
    torch.manual_seed(100)

    # 4 fixed samples
    raw_tokens = torch.randint(0, vocab_size, (4, context_length + 1), dtype=torch.long)
    x = raw_tokens[:, :-1]
    y = raw_tokens[:, 1:]

    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=1,
        n_heads=2,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        attention_backend="eager",
    )
    torch.manual_seed(999)
    model1 = GPT(cfg)

    # Model 2: identical initial weights
    torch.manual_seed(999)
    model2 = GPT(cfg)

    train_cfg1 = TrainingConfig(learning_rate=1e-3, weight_decay=0.0, batch_size=4, gradient_accumulation_steps=1, max_steps=1, warmup_steps=0, max_grad_norm=None)
    train_cfg2 = TrainingConfig(learning_rate=1e-3, weight_decay=0.0, batch_size=2, gradient_accumulation_steps=2, max_steps=1, warmup_steps=0, max_grad_norm=None)

    ds1 = TensorDataset(x, y)
    loader1 = DataLoader(ds1, batch_size=4, shuffle=False)
    trainer1 = Trainer(model1, train_cfg1, loader1)

    ds2 = TensorDataset(x, y)
    loader2 = DataLoader(ds2, batch_size=2, shuffle=False)
    trainer2 = Trainer(model2, train_cfg2, loader2)

    # Execute single step
    iter1 = iter(trainer1._infinite_loader(loader1))
    iter2 = iter(trainer2._infinite_loader(loader2))

    res1 = trainer1.train_step(iter1)
    res2 = trainer2.train_step(iter2)

    # Losses should match
    torch.testing.assert_close(torch.tensor(res1["loss"]), torch.tensor(res2["loss"]), rtol=1e-5, atol=1e-5)

    # Updated weights should match
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4, msg=f"Accumulation divergence at {n1}")


def test_validation_mode_non_contamination() -> None:
    """Verifies that evaluate() runs in eval mode without populating gradients or modifying step count."""
    model, _, val_ds = make_tiny_model_and_data(vocab_size=32, context_length=8, num_samples=4)
    val_loader = DataLoader(val_ds, batch_size=2)

    train_cfg = TrainingConfig(batch_size=2, warmup_steps=2, max_steps=10)
    trainer = Trainer(model, train_cfg, DataLoader(val_ds, batch_size=2), val_loader)

    initial_step = trainer.global_step
    initial_tokens = trainer.tokens_seen

    # Run evaluation
    val_loss = trainer.evaluate(num_batches=2)

    assert val_loss > 0
    assert trainer.global_step == initial_step, "Validation must not increment global_step!"
    assert trainer.tokens_seen == initial_tokens, "Validation must not increment tokens_seen!"
    assert model.training is True, "Model must be restored to training mode after evaluate()!"

    for name, p in model.named_parameters():
        assert p.grad is None, f"Gradient buffer contaminated at {name} during validation!"


def test_resume_continuation(tmp_path: Path) -> None:
    """Verifies that 2 consecutive steps match 1 step -> save -> load -> 1 step on identical data stream."""
    vocab_size, context_length = 32, 8
    torch.manual_seed(42)
    # 1 identical batch of 2 samples repeated
    raw = torch.randint(0, vocab_size, (2, context_length + 1), dtype=torch.long)
    ds = TensorDataset(raw[:, :-1], raw[:, 1:])

    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64, dropout=0.0)

    # Run A: 2 continuous steps
    torch.manual_seed(777)
    modelA = GPT(cfg)
    train_cfgA = TrainingConfig(learning_rate=1e-3, max_steps=2, warmup_steps=0, batch_size=2, gradient_accumulation_steps=1, output_dir=str(tmp_path / "runA"))
    trainerA = Trainer(modelA, train_cfgA, DataLoader(ds, batch_size=2, shuffle=False))
    trainerA.train()

    # Run B: Step 1 -> Checkpoint -> Load -> Step 2
    torch.manual_seed(777)
    modelB = GPT(cfg)
    train_cfgB1 = TrainingConfig(learning_rate=1e-3, max_steps=1, warmup_steps=0, batch_size=2, gradient_accumulation_steps=1, output_dir=str(tmp_path / "runB"))
    trainerB1 = Trainer(modelB, train_cfgB1, DataLoader(ds, batch_size=2, shuffle=False))
    trainerB1.train()

    ckpt_path = tmp_path / "runB" / "step-00000001.pt"
    assert ckpt_path.exists()

    # Create fresh Model C and resume for step 2
    modelC = GPT(cfg)
    train_cfgC = TrainingConfig(learning_rate=1e-3, max_steps=2, warmup_steps=0, batch_size=2, gradient_accumulation_steps=1, output_dir=str(tmp_path / "runB"))
    trainerC = Trainer(modelC, train_cfgC, DataLoader(ds, batch_size=2, shuffle=False))
    trainerC.train(resume_from=ckpt_path)

    # Model A and Model C should have identical weights after 2 steps
    for (nA, pA), (nC, pC) in zip(modelA.named_parameters(), modelC.named_parameters()):
        torch.testing.assert_close(pA, pC, rtol=1e-5, atol=1e-5, msg=f"Resume divergence at parameter {nA}")
