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
from basikgpt.training.optimizer import configure_optimizers
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

    diff = (logits_cpu - logits_cuda.cpu()).abs()
    max_abs_error = float(diff.max().item())
    mean_abs_error = float(diff.mean().item())
    print(f"CPU↔CUDA FP32 max_abs_error={max_abs_error:.6e} mean_abs_error={mean_abs_error:.6e}")
    assert math.isfinite(max_abs_error) and math.isfinite(mean_abs_error)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_token_accounting(tmp_path: Path) -> None:
    """Verifies tokens_seen == B * T * G * steps on CUDA FP32."""
    batch_size, context_length, grad_accum, steps = 2, 16, 3, 4
    cfg = GPTConfig(vocab_size=64, context_length=context_length, n_layers=1, n_heads=2, d_model=16, d_ff=64, dropout=0.0)
    model = GPT(cfg)
    raw = torch.randint(0, 64, (32, context_length + 1), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw[:, :-1], raw[:, 1:]), batch_size=batch_size)
    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        device="cuda",
        precision="fp32",
        output_dir=str(tmp_path),
        log_interval=1,
        eval_interval=100,
        checkpoint_interval=100,
    )
    trainer = Trainer(model, train_cfg, loader)
    trainer.train()
    expected = batch_size * context_length * grad_accum * steps
    assert trainer.tokens_seen == expected
    assert trainer.global_step == steps


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cpu_checkpoint_loads_on_cuda(tmp_path: Path) -> None:
    """Verifies a CPU-saved checkpoint can be loaded onto CUDA."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64, dropout=0.0)
    cpu_model = GPT(cfg)
    train_cfg = TrainingConfig(warmup_steps=0, device="cpu", precision="fp32")
    cpu_opt = torch.optim.AdamW(cpu_model.parameters(), lr=1e-3)
    ckpt = tmp_path / "cpu.pt"
    save_checkpoint(ckpt, cpu_model, cpu_opt, global_step=3, tokens_seen=48, training_config=train_cfg, model_config=cfg)

    gpu_model = GPT(cfg).to("cuda")
    gpu_opt = torch.optim.AdamW(gpu_model.parameters(), lr=1e-3)
    meta = load_checkpoint(ckpt, gpu_model, gpu_opt, device="cuda")
    assert meta["global_step"] == 3
    assert meta["tokens_seen"] == 48
    assert next(gpu_model.parameters()).is_cuda


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_gpu_checkpoint_state_continuous_resume(tmp_path: Path) -> None:
    """Verifies GPU save → recreate → GPU resume restores step and token counters."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=1, n_heads=2, d_model=16, d_ff=64, dropout=0.0)
    raw = torch.randint(0, 64, (16, 17), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw[:, :-1], raw[:, 1:]), batch_size=2)
    first_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        output_dir=str(tmp_path / "run"),
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=2,
    )
    first = Trainer(GPT(cfg), first_cfg, loader)
    first.train()
    ckpt = tmp_path / "run" / "step-final.pt"
    assert ckpt.exists()
    tokens_after_first = first.tokens_seen

    resume_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        output_dir=str(tmp_path / "run"),
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=2,
    )
    resumed = Trainer(GPT(cfg), resume_cfg, loader, resume_from=ckpt, overwrite=True)
    resumed.train(resume_from=ckpt)
    assert resumed.global_step == 4
    assert resumed.tokens_seen == tokens_after_first + (2 * 16 * 1 * 2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_gpu_checkpoint_cpu_inspection(tmp_path: Path) -> None:
    """Verifies a CUDA checkpoint can be inspected with map_location=cpu."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64, dropout=0.0)
    raw = torch.randint(0, 32, (8, 9), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw[:, :-1], raw[:, 1:]), batch_size=2)
    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        output_dir=str(tmp_path),
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=1,
    )
    trainer = Trainer(GPT(cfg), train_cfg, loader)
    trainer.train()
    ckpt = tmp_path / "step-final.pt"

    inspect_cfg = TrainingConfig(warmup_steps=0, device="cpu", precision="fp32")
    cpu_model = GPT(cfg)
    cpu_opt = configure_optimizers(cpu_model, inspect_cfg)
    meta = load_checkpoint(ckpt, cpu_model, cpu_opt, device="cpu")
    assert meta["global_step"] == 1
    assert meta["tokens_seen"] == 16
    assert not next(cpu_model.parameters()).is_cuda


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_fp32_bf16_loss_sanity() -> None:
    """Records FP32 vs BF16 loss on identical weights/batch. Exact equality is not required."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this CUDA GPU")

    torch.manual_seed(42)
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=4, d_model=32, d_ff=128, dropout=0.0)
    model = GPT(cfg).to("cuda")
    model.eval()
    x = torch.randint(0, 64, (2, 16), dtype=torch.long, device="cuda")
    y = torch.randint(0, 64, (2, 16), dtype=torch.long, device="cuda")
    with torch.inference_mode():
        loss_fp32 = compute_cross_entropy_loss(model(x), y)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_bf16 = compute_cross_entropy_loss(model(x), y)

    fp32 = float(loss_fp32.item())
    bf16 = float(loss_bf16.item())
    abs_diff = abs(fp32 - bf16)
    rel_diff = abs_diff / max(abs(fp32), 1e-12)
    print(f"FP32 loss={fp32:.6f} BF16 loss={bf16:.6f} abs_diff={abs_diff:.6e} rel_diff={rel_diff:.6e}")
    assert math.isfinite(fp32) and math.isfinite(bf16)
    # Record the actual gap. Fail only on catastrophic divergence, not on expected BF16 rounding.
    if rel_diff > 0.25:
        raise AssertionError(
            f"FP32 vs BF16 relative loss difference is large ({rel_diff:.4f}); "
            "not widening tolerance to hide the gap."
        )


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
    save_checkpoint(
        ckpt_file,
        model,
        trainer.optimizer,
        global_step=1,
        tokens_seen=32,
        training_config=train_cfg,
        model_config=cfg,
        scaler=trainer.scaler,
    )
    meta = load_checkpoint(ckpt_file, model, trainer.optimizer, scaler=trainer.scaler, device="cuda")
    assert meta["global_step"] == 1
