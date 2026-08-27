"""Unit and integration tests for random seed management and deterministic training."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.reproducibility import seed_everything
from basikgpt.training.trainer import Trainer


def test_seed_everything_initialization_reproducibility() -> None:
    """Verifies that two models initialized after seed_everything(1337) have identical weights."""
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
        dropout=0.0,
    )

    seed_everything(1337)
    model1 = GPT(cfg)

    seed_everything(1337)
    model2 = GPT(cfg)

    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, rtol=0.0, atol=0.0, msg=f"Weight mismatch at {n1}")


def test_seed_difference_produces_different_weights() -> None:
    """Verifies that different seeds produce distinct initial parameters."""
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
        dropout=0.0,
    )

    seed_everything(100)
    model1 = GPT(cfg)

    seed_everything(200)
    model2 = GPT(cfg)

    mismatches = 0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            mismatches += 1
    assert mismatches > 0, "Different seeds should produce different model parameters"


def test_deterministic_tiny_training_trajectory(tmp_path: Path) -> None:
    """Verifies that two training runs with identical seed, data, and config produce identical loss trajectories and weights."""
    vocab_size, context_length = 32, 8
    num_samples = 4

    # Generate fixed deterministic dataset
    torch.manual_seed(42)
    raw_tokens = torch.randint(0, vocab_size, (num_samples, context_length + 1), dtype=torch.long)
    x = raw_tokens[:, :-1]
    y = raw_tokens[:, 1:]

    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
    )

    # Run 1
    seed_everything(777)
    model1 = GPT(cfg)
    train_cfg1 = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=5,
        batch_size=2,
        gradient_accumulation_steps=1,
        seed=777,
        device="cpu",
        precision="fp32",
        output_dir=str(tmp_path / "run1"),
    )
    loader1 = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    trainer1 = Trainer(model1, train_cfg1, loader1, overwrite=True)
    history1 = trainer1.train()

    # Run 2
    seed_everything(777)
    model2 = GPT(cfg)
    train_cfg2 = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=5,
        batch_size=2,
        gradient_accumulation_steps=1,
        seed=777,
        device="cpu",
        precision="fp32",
        output_dir=str(tmp_path / "run2"),
    )
    loader2 = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    trainer2 = Trainer(model2, train_cfg2, loader2, overwrite=True)
    history2 = trainer2.train()

    assert len(history1) == len(history2)
    for h1, h2 in zip(history1, history2):
        assert h1["step"] == h2["step"]
        assert h1["loss"] == pytest.approx(h2["loss"], rel=1e-5)
        assert h1["grad_norm"] == pytest.approx(h2["grad_norm"], rel=1e-5)

    # Final weights equality
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5, msg=f"Weight divergence at {n1}")
