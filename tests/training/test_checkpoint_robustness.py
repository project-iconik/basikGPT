"""Unit tests for checkpoint schema versioning, integrity validation, and architecture mismatch detection."""

from pathlib import Path
import pytest
import torch
from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from basikgpt.training.config import TrainingConfig


def test_checkpoint_schema_version_included(tmp_path: Path) -> None:
    """Verifies that save_checkpoint records CHECKPOINT_SCHEMA_VERSION in the saved payload."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    model = GPT(cfg)
    train_cfg = TrainingConfig()
    opt = torch.optim.AdamW(model.parameters())

    ckpt_path = save_checkpoint(
        tmp_path / "test_v1.pt",
        model=model,
        optimizer=opt,
        global_step=1,
        tokens_seen=32,
        training_config=train_cfg,
        model_config=cfg,
    )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "schema_version" in payload
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_unsupported_version_rejected(tmp_path: Path) -> None:
    """Verifies that load_checkpoint rejects unsupported future schema versions."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    model = GPT(cfg)
    train_cfg = TrainingConfig()
    opt = torch.optim.AdamW(model.parameters())

    ckpt_path = save_checkpoint(
        tmp_path / "future_v99.pt",
        model=model,
        optimizer=opt,
        global_step=1,
        tokens_seen=32,
        training_config=train_cfg,
        model_config=cfg,
    )

    # Mutate version to future unsupported version
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload["schema_version"] = 999
    torch.save(payload, ckpt_path)

    with pytest.raises(ValueError, match="Unsupported checkpoint schema version '999'"):
        load_checkpoint(ckpt_path, model, opt)


def test_checkpoint_missing_required_fields_rejected(tmp_path: Path) -> None:
    """Verifies that load_checkpoint fails if required metadata fields are missing."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    model = GPT(cfg)
    train_cfg = TrainingConfig()
    opt = torch.optim.AdamW(model.parameters())

    ckpt_path = save_checkpoint(
        tmp_path / "valid.pt",
        model=model,
        optimizer=opt,
        global_step=1,
        tokens_seen=32,
        training_config=train_cfg,
        model_config=cfg,
    )

    # Delete 'tokens_seen'
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del payload["tokens_seen"]
    bad_ckpt = tmp_path / "corrupted_missing_field.pt"
    torch.save(payload, bad_ckpt)

    with pytest.raises(ValueError, match="missing required fields: tokens_seen"):
        load_checkpoint(bad_ckpt, model, opt)


def test_checkpoint_corrupted_file_error(tmp_path: Path) -> None:
    """Verifies that loading garbage bytes raises RuntimeError with informative message."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters())

    corrupt_file = tmp_path / "corrupt.pt"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_DATA_1234567890")

    with pytest.raises(RuntimeError, match="corrupted or unreadable file"):
        load_checkpoint(corrupt_file, model, opt)


def test_checkpoint_architecture_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that resuming a checkpoint with mismatched model architecture (e.g. d_model) is rejected."""
    cfg1 = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    cfg2 = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=64, d_ff=256)

    model1 = GPT(cfg1)
    train_cfg = TrainingConfig()
    opt1 = torch.optim.AdamW(model1.parameters())

    ckpt_path = save_checkpoint(
        tmp_path / "model1.pt",
        model=model1,
        optimizer=opt1,
        global_step=10,
        tokens_seen=320,
        training_config=train_cfg,
        model_config=cfg1,
    )

    # Attempt to load into model2 (different d_model)
    model2 = GPT(cfg2)
    opt2 = torch.optim.AdamW(model2.parameters())

    with pytest.raises(ValueError, match="Checkpoint architecture is incompatible with current model"):
        load_checkpoint(ckpt_path, model2, opt2, expected_model_config=cfg2)
