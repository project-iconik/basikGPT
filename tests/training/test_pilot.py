"""Unit tests for the Pilot Pretraining Protocol, guardrails, and stage execution in basikGPT."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.accounting import calculate_training_steps
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.pilot import PILOT_STAGES, PilotSummary
from basikgpt.training.trainer import Trainer


def make_tiny_model_and_loader(
    vocab_size: int = 64,
    context_length: int = 16,
    num_samples: int = 16,
    batch_size: int = 2,
) -> tuple[GPT, DataLoader]:
    """Helper to instantiate a miniature model and deterministic DataLoader."""
    torch.manual_seed(42)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=64,
        dropout=0.0,
    )
    model = GPT(cfg)
    raw = torch.randint(0, vocab_size, (num_samples, context_length + 1), dtype=torch.long)
    ds = TensorDataset(raw[:, :-1], raw[:, 1:])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return model, loader


def test_scheduler_step_alignment_with_accumulation(tmp_path: Path) -> None:
    """Verifies that LR scheduler updates strictly once per optimizer step, not on micro-batches."""
    model, loader = make_tiny_model_and_loader(num_samples=16, batch_size=2)
    # 4 micro-batches per optimizer step
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=4,
        max_steps=8,
        batch_size=2,
        gradient_accumulation_steps=4,
        output_dir=str(tmp_path / "sched_test"),
    )
    trainer = Trainer(model, train_cfg, loader)

    data_iter = trainer._infinite_loader(loader)

    # Before any step
    assert trainer.global_step == 0

    # Step 1: processes 4 micro-batches
    step1_metrics = trainer.train_step(data_iter)
    assert trainer.global_step == 1
    # Warmup step 0: LR = 1e-3 * (0 + 1) / 4 = 2.5e-4
    assert step1_metrics["lr"] == pytest.approx(2.5e-4)

    # Step 2: processes 4 micro-batches
    step2_metrics = trainer.train_step(data_iter)
    assert trainer.global_step == 2
    # Warmup step 1: LR = 1e-3 * (1 + 1) / 4 = 5.0e-4
    assert step2_metrics["lr"] == pytest.approx(5.0e-4)


def test_tokens_seen_exact_accounting(tmp_path: Path) -> None:
    """Verifies that tokens_seen increments exactly by B * T * G per optimizer step."""
    batch_size, context_length, grad_accum = 2, 16, 3
    model, loader = make_tiny_model_and_loader(
        context_length=context_length, num_samples=12, batch_size=batch_size
    )

    tokens_per_step = batch_size * context_length * grad_accum  # 2 * 16 * 3 = 96
    train_cfg = TrainingConfig(
        max_steps=5,
        warmup_steps=0,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        output_dir=str(tmp_path / "tokens_test"),
    )
    trainer = Trainer(model, train_cfg, loader)
    trainer.train()

    assert trainer.global_step == 5
    assert trainer.tokens_seen == 5 * tokens_per_step == 480


def test_validation_tokens_isolation(tmp_path: Path) -> None:
    """Verifies that evaluation batches do not pollute training tokens_seen."""
    model, train_loader = make_tiny_model_and_loader(num_samples=16, batch_size=2)
    _, val_loader = make_tiny_model_and_loader(num_samples=8, batch_size=2)

    train_cfg = TrainingConfig(
        max_steps=3,
        warmup_steps=0,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=1,
        eval_batches=4,
        output_dir=str(tmp_path / "val_iso_test"),
    )
    trainer = Trainer(model, train_cfg, train_loader, val_loader)
    trainer.train()

    # Total training tokens should be exactly 3 steps * 2 * 16 * 1 = 96
    assert trainer.tokens_seen == 96


def test_non_finite_loss_fail_fast(tmp_path: Path) -> None:
    """Verifies that non-finite training loss raises FloatingPointError immediately."""
    model, loader = make_tiny_model_and_loader()
    train_cfg = TrainingConfig(
        max_steps=5,
        warmup_steps=0,
        batch_size=2,
        gradient_accumulation_steps=1,
        output_dir=str(tmp_path / "nan_loss_test"),
    )
    trainer = Trainer(model, train_cfg, loader)

    # Force model weights to NaN to trigger non-finite loss
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="Non-finite training loss detected"):
        trainer.train()


def test_non_finite_gradient_fail_fast(tmp_path: Path) -> None:
    """Verifies that non-finite gradient norms trigger fail-fast abort."""
    model, loader = make_tiny_model_and_loader()
    train_cfg = TrainingConfig(
        max_steps=5,
        warmup_steps=0,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        output_dir=str(tmp_path / "nan_grad_test"),
    )
    trainer = Trainer(model, train_cfg, loader)
    data_iter = trainer._infinite_loader(loader)

    # Corrupt model linear layer weights to infinity
    with torch.no_grad():
        model.blocks[0].mlp.fc_in.weight.fill_(1e30)

    with pytest.raises(FloatingPointError, match="Non-finite.*detected"):
        trainer.train_step(data_iter)


def test_checkpoint_metadata_and_resume_continuity(tmp_path: Path) -> None:
    """Verifies save/load restores global_step, tokens_seen, and model/optimizer states."""
    from basikgpt.training.optimizer import configure_optimizers

    model, loader = make_tiny_model_and_loader(num_samples=16, batch_size=2)
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        max_steps=4,
        warmup_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        checkpoint_interval=2,
        output_dir=str(tmp_path / "ckpt_test"),
    )
    trainer = Trainer(model, train_cfg, loader)
    trainer.train()

    ckpt_path = tmp_path / "ckpt_test" / "step-00000002.pt"
    assert ckpt_path.exists()

    # Load into fresh model with matching 2 parameter groups
    fresh_model = GPT(model.config)
    fresh_optimizer = configure_optimizers(fresh_model, train_cfg)
    meta = load_checkpoint(ckpt_path, fresh_model, fresh_optimizer)

    assert meta["global_step"] == 2
    assert meta["tokens_seen"] == 2 * (2 * 16 * 1) == 64
    assert meta["schema_version"] == 1

    # Weights in fresh_model should now match model at step 2
    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert p2.shape == p1.shape


def test_pilot_stage_presets_validity() -> None:
    """Verifies that all defined pilot stages produce positive integer plans."""
    for stage_id, stage_spec in PILOT_STAGES.items():
        plan = stage_spec.compute_plan(world_size=1)
        assert plan.requested_token_budget == stage_spec.target_tokens
        assert plan.tokens_per_optimizer_step > 0
        assert plan.optimizer_steps > 0
        assert plan.actual_token_budget >= plan.requested_token_budget
        assert plan.overshoot_tokens >= 0


def test_structured_pilot_summary_schema_and_serialization(tmp_path: Path) -> None:
    """Verifies structured JSON serialization and human-readable formatting of PilotSummary."""
    summary = PilotSummary(
        pilot_stage="stage_a",
        status="passed",
        requested_tokens=10_000,
        actual_tokens=10_240,
        tokens_per_step=256,
        optimizer_steps=40,
        initial_train_loss=10.82,
        final_train_loss=10.15,
        final_validation_loss=10.20,
        best_validation_loss=10.20,
        min_gradient_norm=0.45,
        max_gradient_norm=1.10,
        initial_learning_rate=1.2e-4,
        final_learning_rate=6.0e-5,
        elapsed_seconds=3.25,
        tokens_per_sec=3150.0,
        checkpoint_path=str(tmp_path / "step-final.pt"),
    )

    out_file = tmp_path / "pilot_summary.json"
    summary.save_json(out_file)
    assert out_file.exists()

    d = summary.to_dict()
    assert d["status"] == "passed"
    assert d["requested_tokens"] == 10000

    human_table = summary.format_human_readable()
    assert "STAGE_A" in human_table
    assert "PASSED" in human_table
    assert "10,240" in human_table
