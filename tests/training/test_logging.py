"""Unit tests for structured JSONL metrics logging and unscaled loss recording."""

import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.trainer import Trainer


def test_structured_metrics_jsonl_schema(tmp_path: Path) -> None:
    """Verifies that metrics.jsonl contains valid JSON objects with typed train and val records."""
    vocab_size, context_length = 32, 8
    torch.manual_seed(42)
    raw = torch.randint(0, vocab_size, (8, context_length + 1), dtype=torch.long)
    train_ds = TensorDataset(raw[:, :-1], raw[:, 1:])
    val_ds = TensorDataset(raw[:2, :-1], raw[:2, 1:])

    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)

    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=2,
        eval_batches=1,
        log_interval=2,
        output_dir=str(tmp_path),
    )

    trainer = Trainer(model, train_cfg, DataLoader(train_ds, batch_size=2), DataLoader(val_ds, batch_size=2), overwrite=True)
    trainer.train()

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()

    records = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) >= 2

    # Check train record
    train_records = [r for r in records if r["type"] == "train"]
    assert len(train_records) > 0
    for tr in train_records:
        assert "step" in tr
        assert "tokens_seen" in tr
        assert "loss" in tr
        assert "learning_rate" in tr
        assert "grad_norm" in tr
        assert "tokens_per_sec" in tr
        assert "elapsed_seconds" in tr
        assert tr["loss"] > 0

    # Check val record
    val_records = [r for r in records if r["type"] == "val"]
    assert len(val_records) > 0
    for vr in val_records:
        assert "step" in vr
        assert "tokens_seen" in vr
        assert "val_loss" in vr
        assert "elapsed_seconds" in vr
        assert vr["val_loss"] > 0


def test_train_loss_logged_is_unscaled(tmp_path: Path) -> None:
    """Verifies that during gradient accumulation (G > 1), the logged train loss is the true unscaled mean Cross-Entropy."""
    vocab_size, context_length = 32, 8
    torch.manual_seed(42)
    raw = torch.randint(0, vocab_size, (8, context_length + 1), dtype=torch.long)
    train_ds = TensorDataset(raw[:, :-1], raw[:, 1:])

    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)

    # Accumulation = 4
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=1,
        batch_size=2,
        gradient_accumulation_steps=4,
        log_interval=1,
        output_dir=str(tmp_path),
    )

    trainer = Trainer(model, train_cfg, DataLoader(train_ds, batch_size=2), overwrite=True)
    history = trainer.train()

    logged_loss = history[0]["loss"]
    # Theoretical cross-entropy on uniform random 32 tokens is ~ -ln(1/32) = 3.465
    # If loss were scaled by 1/4, it would be ~ 0.866.
    assert logged_loss > 2.0, f"Expected unscaled cross entropy (~3.46), got {logged_loss:.4f} (looks mistakenly scaled)"
