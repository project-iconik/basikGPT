"""Unit tests for structured JSONL metrics logging and unscaled loss recording."""

import json
import math
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
    prev_tokens = 0
    param_count = model.num_parameters()
    for tr in train_records:
        assert "step" in tr
        assert "tokens_seen" in tr
        assert "loss" in tr
        assert "learning_rate" in tr
        assert "grad_norm" in tr
        assert "grad_clipped" in tr
        assert isinstance(tr["grad_clipped"], bool)
        assert tr["grad_clipped"] is (tr["grad_norm"] > 1.0)
        assert "estimated_flops" in tr
        tokens_delta = tr["tokens_seen"] - prev_tokens
        assert tr["estimated_flops"] == 6 * param_count * tokens_delta
        assert "tokens_per_sec" in tr
        assert "elapsed_seconds" in tr
        assert tr["loss"] > 0
        prev_tokens = tr["tokens_seen"]

    # Check val record
    val_records = [r for r in records if r["type"] == "val"]
    assert len(val_records) > 0
    for vr in val_records:
        assert "step" in vr
        assert "tokens_seen" in vr
        assert "val_loss" in vr
        assert "val_perplexity" in vr
        assert "elapsed_seconds" in vr
        assert vr["val_loss"] > 0
        assert vr["val_perplexity"] == pytest.approx(math.exp(vr["val_loss"]))

    run_meta = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run_meta["extra"]["parameter_count"] == param_count
    assert run_meta["extra"]["tokens_per_optimizer_step"] == 2 * 8 * 1


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
    assert history[0]["grad_clipped"] == (history[0]["grad_norm"] > 1.0)


def test_val_metrics_logged_when_eval_not_aligned_with_log(tmp_path: Path) -> None:
    """Verifies val records are written on every eval step even if it is not a log step."""
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
        max_steps=6,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=2,
        eval_batches=1,
        log_interval=3,
        checkpoint_interval=100,
        output_dir=str(tmp_path),
    )

    trainer = Trainer(
        model,
        train_cfg,
        DataLoader(train_ds, batch_size=2),
        DataLoader(val_ds, batch_size=2),
        overwrite=True,
    )
    trainer.train()

    records = []
    with open(tmp_path / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    val_steps = {r["step"] for r in records if r["type"] == "val"}
    assert val_steps == {2, 4, 6}

    train_steps = {r["step"] for r in records if r["type"] == "train"}
    assert train_steps == {1, 3, 6}


def test_grad_clipped_false_when_clipping_disabled(tmp_path: Path) -> None:
    """When max_grad_norm is None, grad_clipped is always False and grad_norm is still recorded."""
    vocab_size, context_length = 32, 8
    torch.manual_seed(42)
    raw = torch.randint(0, vocab_size, (8, context_length + 1), dtype=torch.long)
    train_ds = TensorDataset(raw[:, :-1], raw[:, 1:])

    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        log_interval=1,
        eval_interval=100,
        checkpoint_interval=100,
        output_dir=str(tmp_path),
    )
    Trainer(model, train_cfg, DataLoader(train_ds, batch_size=2), overwrite=True).train()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [r for r in records if r["type"] == "train"]
    assert train_records
    for tr in train_records:
        assert tr["grad_clipped"] is False
        assert tr["grad_norm"] > 0


def test_step_one_train_record_logged_regardless_of_interval(tmp_path: Path) -> None:
    """Step 1 is always written to metrics.jsonl even when log_interval skips it."""
    vocab_size, context_length = 32, 8
    torch.manual_seed(7)
    raw = torch.randint(0, vocab_size, (8, context_length + 1), dtype=torch.long)
    train_ds = TensorDataset(raw[:, :-1], raw[:, 1:])
    val_ds = TensorDataset(raw[:2, :-1], raw[:2, 1:])
    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=2,
        eval_batches=1,
        log_interval=10,
        checkpoint_interval=100,
        eval_at_start=True,
        output_dir=str(tmp_path),
    )
    Trainer(
        GPT(cfg),
        train_cfg,
        DataLoader(train_ds, batch_size=2),
        DataLoader(val_ds, batch_size=2),
        overwrite=True,
    ).train()

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [r for r in records if r["type"] == "train"]
    assert train_records[0]["step"] == 1
    assert {r["step"] for r in train_records} >= {1, 4}

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["first_train_step"] == 1
    assert summary["first_train_loss"] == pytest.approx(train_records[0]["loss"])
    assert summary["last_train_step"] == 4
    assert summary["min_val_loss"] is not None
    assert summary["min_val_perplexity"] == pytest.approx(math.exp(summary["min_val_loss"]))
    assert summary["min_val_step"] is not None
    assert summary["tokens_per_parameter"] == pytest.approx(
        summary["tokens_seen"] / summary["parameter_count"]
    )
    assert summary["gpu_hours"] == pytest.approx(summary["elapsed_seconds"] / 3600.0)
    assert summary["uniform_ce_reference"] == pytest.approx(math.log(vocab_size))
    assert summary["eval_tokens"] == 2 * 8 * 1
    assert summary["parameter_count"] > 0

    run_meta = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    extra = run_meta["extra"]
    assert extra["tie_word_embeddings"] is True
    assert extra["eval_tokens"] == 16
    assert extra["uniform_ce_reference"] == pytest.approx(math.log(vocab_size))
    assert extra["parameter_breakdown"]["measured_unique"] == extra["parameter_count"]
    assert extra["optimizer_param_groups"]["decay_parameters"] > 0
    assert extra["optimizer_param_groups"]["no_decay_parameters"] > 0
    assert extra["packed_data"]["train_sequences"] == 8
    assert extra["token_budget"]["actual_token_budget"] == 4 * 16
    assert extra["head_dim"] == 8

