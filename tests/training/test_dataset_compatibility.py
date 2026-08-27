"""Unit tests for dataset manifest compatibility and run directory collision prevention."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.compatibility import validate_dataset_model_compatibility
from basikgpt.training.config import TrainingConfig
from basikgpt.training.trainer import Trainer


def test_vocab_size_mismatch_rejected() -> None:
    """Verifies that dataset manifest with mismatched vocab_size is rejected."""
    model_cfg = GPTConfig.gpt2_small(context_length=1024)
    bad_manifest = {
        "tokenizer": {
            "vocab_size": 32000,
            "encoding_name": "llama",
        }
    }

    with pytest.raises(ValueError, match="Dataset vocabulary size \\(32,000\\) does not match model vocab_size \\(50,257\\)"):
        validate_dataset_model_compatibility(model_cfg, bad_manifest)


def test_tokenizer_encoding_mismatch_rejected() -> None:
    """Verifies that non-gpt2 tokenizer with vocab_size 50257 is rejected."""
    model_cfg = GPTConfig.gpt2_small(context_length=1024)
    bad_manifest = {
        "tokenizer": {
            "vocab_size": 50257,
            "encoding_name": "cl100k_base",
        }
    }

    with pytest.raises(ValueError, match="is incompatible with canonical GPT-2 configuration"):
        validate_dataset_model_compatibility(model_cfg, bad_manifest)


def test_context_length_exceeds_model_max_rejected() -> None:
    """Verifies that requesting context length longer than model max raises ValueError."""
    model_cfg = GPTConfig(vocab_size=128, context_length=64, n_layers=1, n_heads=2, d_model=16, d_ff=64)

    with pytest.raises(ValueError, match="exceeds model maximum context length"):
        validate_dataset_model_compatibility(model_cfg, requested_context_length=128)


def test_run_directory_collision_prevention(tmp_path: Path) -> None:
    """Verifies that running Trainer against an existing non-empty directory without overwrite=True raises FileExistsError."""
    vocab_size, context_length = 32, 8
    raw = torch.randint(0, vocab_size, (4, context_length + 1), dtype=torch.long)
    ds = TensorDataset(raw[:, :-1], raw[:, 1:])

    cfg = GPTConfig(vocab_size=vocab_size, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)
    train_cfg = TrainingConfig(output_dir=str(tmp_path / "colliding_run"), max_steps=1, warmup_steps=0)

    # First run succeeds
    trainer1 = Trainer(model, train_cfg, DataLoader(ds, batch_size=2))
    trainer1.train()

    # Second run without overwrite raises FileExistsError
    with pytest.raises(FileExistsError, match="already exists and contains training artifacts"):
        Trainer(model, train_cfg, DataLoader(ds, batch_size=2), overwrite=False)

    # With overwrite=True, it succeeds
    trainer3 = Trainer(model, train_cfg, DataLoader(ds, batch_size=2), overwrite=True)
    assert trainer3 is not None
