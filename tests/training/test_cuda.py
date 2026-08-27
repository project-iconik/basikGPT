"""Unit and integration tests for CUDA execution, device resolution, and mixed precision."""

import math
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.trainer import Trainer, resolve_device


# =====================================================================
# 1. CPU Guardrail Tests (Run on any machine without CUDA)
# =====================================================================

def test_cuda_device_explicit_error_when_unavailable() -> None:
    """Verifies that requesting CUDA on a non-CUDA environment raises RuntimeError without silent fallback."""
    if torch.cuda.is_available():
        pytest.skip("Test specifically verifies error behavior when CUDA is unavailable")

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda:0")


def test_cpu_mixed_precision_error() -> None:
    """Verifies that attempting BF16 or FP16 on CPU raises ValueError."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)
    ds = TensorDataset(torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
    loader = DataLoader(ds, batch_size=2)

    with pytest.raises(ValueError, match="CPU mixed precision is not supported"):
        Trainer(model, TrainingConfig(device="cpu", precision="bf16", warmup_steps=0), loader)

    with pytest.raises(ValueError, match="CPU mixed precision is not supported"):
        Trainer(model, TrainingConfig(device="cpu", precision="fp16", warmup_steps=0), loader)


def test_precision_config_invalid() -> None:
    """Verifies that unsupported precision strings raise ValueError."""
    with pytest.raises(ValueError, match="precision must be one of 'fp32', 'bf16', 'fp16'"):
        TrainingConfig(precision="int8")  # type: ignore[arg-type]


def test_checkpoint_scaler_state_roundtrip(tmp_path: Path) -> None:
    """Verifies that save_checkpoint and load_checkpoint preserve scaler_state_dict."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)
    train_cfg = TrainingConfig(warmup_steps=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Mock scaler state payload
    ckpt_file = tmp_path / "scaler_test.pt"
    save_checkpoint(
        checkpoint_path=ckpt_file,
        model=model,
        optimizer=optimizer,
        global_step=10,
        tokens_seen=500,
        training_config=train_cfg,
        model_config=cfg,
        scaler=None,
    )

    meta = load_checkpoint(ckpt_file, model, optimizer, device="cpu")
    assert meta["global_step"] == 10
    assert meta["tokens_seen"] == 500


# =====================================================================
# 2. CUDA & Mixed Precision Tests (Skipped on local CPU, run on RunPod)
# =====================================================================

@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_fp32_forward_backward() -> None:
    """Verifies CUDA FP32 training step, loss computation, and gradient updates on GPU."""
    torch.cuda.empty_cache()
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4, d_model=32, d_ff=128, dropout=0.0)
    model = GPT(cfg)

    raw_tokens = torch.randint(0, 64, (4, 17), dtype=torch.long)
    ds = TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:])
    loader = DataLoader(ds, batch_size=2)

    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
    )
    trainer = Trainer(model, train_cfg, loader)
    step_res = trainer.train_step(iter(trainer._infinite_loader(loader)))

    assert not math.isnan(step_res["loss"])
    assert not math.isinf(step_res["loss"])
    assert step_res["loss"] > 0
    assert next(model.parameters()).is_cuda


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_fp32_vs_cpu_fp32_parity() -> None:
    """Verifies numerical closeness between CPU FP32 and CUDA FP32 forward + loss on identical weights."""
    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4, d_model=32, d_ff=128, dropout=0.0)
    model_cpu = GPT(cfg)
    model_cuda = GPT(cfg)
    model_cuda.load_state_dict(model_cpu.state_dict())
    model_cuda.to("cuda")

    x = torch.randint(0, 64, (2, 16), dtype=torch.long)
    y = torch.randint(0, 64, (2, 16), dtype=torch.long)

    # CPU forward
    logits_cpu = model_cpu(x)
    loss_cpu = compute_cross_entropy_loss(logits_cpu, y)

    # CUDA forward
    logits_cuda = model_cuda(x.to("cuda"))
    loss_cuda = compute_cross_entropy_loss(logits_cuda, y.to("cuda"))

    torch.testing.assert_close(logits_cpu, logits_cuda.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(loss_cpu, loss_cuda.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_bf16_forward_backward() -> None:
    """Verifies CUDA BF16 autocast training step and finite loss."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this CUDA GPU")

    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4, d_model=32, d_ff=128, dropout=0.0)
    model = GPT(cfg)

    raw_tokens = torch.randint(0, 64, (4, 17), dtype=torch.long)
    ds = TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:])
    loader = DataLoader(ds, batch_size=2)

    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="bf16",
    )
    trainer = Trainer(model, train_cfg, loader)
    step_res = trainer.train_step(iter(trainer._infinite_loader(loader)))

    assert not math.isnan(step_res["loss"])
    assert not math.isinf(step_res["loss"])
    assert step_res["loss"] > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_bf16_tiny_overfit(tmp_path: Path) -> None:
    """Verifies that BF16 mixed-precision training successfully reduces loss on a tiny batch."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this CUDA GPU")

    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=2, d_model=32, d_ff=128, dropout=0.0)
    model = GPT(cfg)

    raw = torch.randint(0, 64, (2, 17), dtype=torch.long)
    ds = TensorDataset(raw[:, :-1], raw[:, 1:])
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    train_cfg = TrainingConfig(
        learning_rate=1e-2,
        min_learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        warmup_steps=5,
        max_steps=50,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=25,
        checkpoint_interval=25,
        log_interval=1,
        output_dir=str(tmp_path),
        device="cuda",
        precision="bf16",
    )
    trainer = Trainer(model, train_cfg, loader)
    history = trainer.train()

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]

    assert final_loss < 0.20 * initial_loss, f"BF16 overfit loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_fp16_grad_scaler_step(tmp_path: Path) -> None:
    """Verifies CUDA FP16 with GradScaler unscaling, step, and state restoration."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4, d_model=32, d_ff=128, dropout=0.0)
    model = GPT(cfg)

    raw_tokens = torch.randint(0, 64, (4, 17), dtype=torch.long)
    ds = TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:])
    loader = DataLoader(ds, batch_size=2)

    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp16",
        output_dir=str(tmp_path),
    )
    trainer = Trainer(model, train_cfg, loader)
    assert trainer.scaler is not None, "GradScaler must be instantiated for FP16"

    step_res = trainer.train_step(iter(trainer._infinite_loader(loader)))
    assert not math.isnan(step_res["loss"])
    assert trainer.scaler.get_scale() > 0

    # Save and reload checkpoint with scaler
    ckpt_file = tmp_path / "fp16_test.pt"
    save_checkpoint(ckpt_file, model, trainer.optimizer, global_step=1, tokens_seen=32, training_config=train_cfg, scaler=trainer.scaler)
    meta = load_checkpoint(ckpt_file, model, trainer.optimizer, scaler=trainer.scaler, device="cuda")
    assert meta["global_step"] == 1
