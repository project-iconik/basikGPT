"""GPU qualification helpers: environment, smoke runs, numerical sanity, capacity probe, summary JSON.

These helpers configure the existing unified Trainer. They are not a separate training engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import shutil
import time
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import atomic_save_json
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.reproducibility import (
    collect_cuda_device_metadata,
    get_git_metadata,
    get_system_metadata,
    seed_everything,
)
from basikgpt.training.trainer import Trainer


OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError, RuntimeError)


def collect_gpu_environment() -> dict[str, Any]:
    """Collects git + CUDA/GPU metadata for qualification reports. No secrets."""
    git = get_git_metadata()
    system = get_system_metadata()
    cuda = collect_cuda_device_metadata()
    return {
        "git": git,
        "system": system,
        "cuda": cuda,
    }


def _tiny_config(context_length: int = 64, attention_backend: str = "sdpa") -> GPTConfig:
    return GPTConfig(
        vocab_size=50257,
        context_length=min(context_length, 64),
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        attention_backend=attention_backend,
        dropout=0.0,
    )


def _model_config(preset: str, context_length: int, attention_backend: str) -> GPTConfig:
    if preset == "tiny":
        return _tiny_config(context_length=context_length, attention_backend=attention_backend)
    return GPTConfig.gpt2_small(
        context_length=context_length,
        attention_backend=attention_backend,
        dropout=0.0,
    )


def make_synthetic_loader(
    vocab_size: int,
    context_length: int,
    batch_size: int,
    num_samples: int,
    seed: int = 1337,
) -> DataLoader:
    """Builds an in-memory next-token DataLoader for GPU smokes and probes."""
    torch.manual_seed(seed)
    raw = torch.randint(0, vocab_size, (num_samples, context_length + 1), dtype=torch.long)
    dataset = TensorDataset(raw[:, :-1], raw[:, 1:])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def _empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _classify_oom_stage(exc: BaseException) -> str:
    message = str(exc).lower()
    if "allocate" in message or "out of memory" in message:
        return "cuda_oom"
    return type(exc).__name__


def run_training_smoke(
    *,
    output_dir: Path | str,
    preset: str,
    device: str,
    precision: str,
    attention_backend: str = "sdpa",
    context_length: int = 1024,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    max_steps: int = 8,
    warmup_steps: int = 2,
    seed: int = 1337,
    eval_interval: int | None = None,
    checkpoint_interval: int | None = None,
) -> dict[str, Any]:
    """Runs a short unified-Trainer smoke on synthetic tokens."""
    _empty_cuda()
    seed_everything(seed)
    model_cfg = _model_config(preset, context_length, attention_backend)
    context_length = model_cfg.context_length
    samples_needed = max(batch_size * grad_accum_steps * (max_steps + 4), batch_size * 4)
    train_loader = make_synthetic_loader(
        vocab_size=model_cfg.vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        num_samples=samples_needed,
        seed=seed,
    )
    val_loader = make_synthetic_loader(
        vocab_size=model_cfg.vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        num_samples=max(batch_size * 2, 2),
        seed=seed + 1,
    )
    model = GPT(model_cfg)
    train_cfg = TrainingConfig(
        learning_rate=6e-4,
        min_learning_rate=6e-5,
        warmup_steps=min(warmup_steps, max_steps),
        max_steps=max_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        eval_interval=eval_interval or max(1, max_steps // 2),
        eval_batches=1,
        checkpoint_interval=checkpoint_interval or max_steps,
        log_interval=1,
        device=device,
        precision=precision,  # type: ignore[arg-type]
        output_dir=str(output_dir),
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        overwrite=True,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)

    history = trainer.train()
    if device.startswith("cuda"):
        torch.cuda.synchronize(trainer.device)

    losses = [row["loss"] for row in history]
    grad_norms = [row["grad_norm"] for row in history]
    lrs = [row["learning_rate"] for row in history]
    tokens_per_step = batch_size * context_length * grad_accum_steps
    result: dict[str, Any] = {
        "preset": preset,
        "parameter_count": model.num_parameters(),
        "device": str(trainer.device),
        "precision": precision,
        "attention_backend": attention_backend,
        "micro_batch_size": batch_size,
        "context_length": context_length,
        "gradient_accumulation_steps": grad_accum_steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "optimizer_steps": trainer.global_step,
        "tokens_seen": trainer.tokens_seen,
        "expected_tokens_seen": tokens_per_step * trainer.global_step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "validation_loss": trainer.last_val_loss,
        "min_gradient_norm": min(grad_norms) if grad_norms else None,
        "max_gradient_norm": max(grad_norms) if grad_norms else None,
        "mean_gradient_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "initial_learning_rate": lrs[0] if lrs else None,
        "final_learning_rate": lrs[-1] if lrs else None,
        "checkpoint_path": str(Path(output_dir) / "step-final.pt"),
        "loss_finite": all(math.isfinite(v) for v in losses),
        "grad_finite": all(math.isfinite(v) for v in grad_norms),
        "token_accounting_ok": trainer.tokens_seen == tokens_per_step * trainer.global_step,
    }
    if trainer.device.type == "cuda":
        result["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated(trainer.device))
        result["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved(trainer.device))
    return result


def compare_cpu_cuda_fp32(*, seed: int = 42) -> dict[str, Any]:
    """Compares tiny-model CPU FP32 vs CUDA FP32 logits. Bitwise equality is not required."""
    seed_everything(seed)
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        attention_backend="sdpa",
    )
    model_cpu = GPT(cfg)
    model_cuda = GPT(cfg)
    model_cuda.load_state_dict(model_cpu.state_dict())
    model_cuda.to("cuda")
    model_cpu.eval()
    model_cuda.eval()

    x = torch.randint(0, 64, (2, 16), dtype=torch.long)
    y = torch.randint(0, 64, (2, 16), dtype=torch.long)
    with torch.inference_mode():
        logits_cpu = model_cpu(x)
        loss_cpu = compute_cross_entropy_loss(logits_cpu, y)
        logits_cuda = model_cuda(x.to("cuda"))
        loss_cuda = compute_cross_entropy_loss(logits_cuda, y.to("cuda"))

    diff = (logits_cpu - logits_cuda.cpu()).abs()
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "loss_cpu": float(loss_cpu.item()),
        "loss_cuda": float(loss_cuda.cpu().item()),
        "loss_abs_diff": abs(float(loss_cpu.item()) - float(loss_cuda.cpu().item())),
    }


def compare_fp32_bf16_loss(*, seed: int = 42) -> dict[str, Any]:
    """Measures FP32 vs BF16 loss on identical weights and batch. Exact equality is not required."""
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 numerical sanity but this GPU does not support bfloat16.")

    seed_everything(seed)
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        attention_backend="sdpa",
    )
    model = GPT(cfg).to("cuda")
    model.eval()
    x = torch.randint(0, 64, (2, 16), dtype=torch.long, device="cuda")
    y = torch.randint(0, 64, (2, 16), dtype=torch.long, device="cuda")

    with torch.inference_mode():
        logits_fp32 = model(x)
        loss_fp32 = compute_cross_entropy_loss(logits_fp32, y)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_bf16 = model(x)
            loss_bf16 = compute_cross_entropy_loss(logits_bf16, y)

    fp32 = float(loss_fp32.item())
    bf16 = float(loss_bf16.item())
    abs_diff = abs(fp32 - bf16)
    rel_diff = abs_diff / max(abs(fp32), 1e-12)
    return {
        "loss_fp32": fp32,
        "loss_bf16": bf16,
        "absolute_difference": abs_diff,
        "relative_difference": rel_diff,
        "both_finite": math.isfinite(fp32) and math.isfinite(bf16),
    }


def verify_checkpoint_portability(output_dir: Path | str, seed: int = 1337) -> dict[str, Any]:
    """Verifies CPU→GPU load, GPU→GPU state-continuous resume, and GPU→CPU inspection."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        attention_backend="sdpa",
    )
    loader = make_synthetic_loader(64, 16, batch_size=2, num_samples=32, seed=seed)

    # A. CPU checkpoint → GPU load
    cpu_model = GPT(cfg)
    cpu_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cpu",
        precision="fp32",
        output_dir=str(out / "cpu_src"),
        seed=seed,
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=2,
    )
    cpu_trainer = Trainer(cpu_model, cpu_cfg, loader, overwrite=True)
    cpu_trainer.train()
    cpu_ckpt = out / "cpu_src" / "step-final.pt"

    gpu_model_a = GPT(cfg).to("cuda")
    gpu_opt_a = configure_optimizers(
        gpu_model_a,
        TrainingConfig(warmup_steps=0, device="cuda", precision="fp32", output_dir=str(out / "gpu_a")),
    )
    meta_a = load_checkpoint(cpu_ckpt, gpu_model_a, gpu_opt_a, device="cuda")
    cpu_to_gpu = {
        "global_step": meta_a["global_step"],
        "tokens_seen": meta_a["tokens_seen"],
        "first_param_device": str(next(gpu_model_a.parameters()).device),
        "ok": next(gpu_model_a.parameters()).is_cuda and meta_a["global_step"] == 2,
    }

    # B. GPU steps 0→2 save, recreate, resume 2→4
    gpu_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        output_dir=str(out / "gpu_run"),
        seed=seed,
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=2,
    )
    gpu_trainer = Trainer(GPT(cfg), gpu_cfg, loader, overwrite=True)
    gpu_trainer.train()
    gpu_ckpt = out / "gpu_run" / "step-final.pt"
    gpu_ckpt_pre_resume = out / "gpu_ckpt_before_resume.pt"
    shutil.copy2(gpu_ckpt, gpu_ckpt_pre_resume)
    resume_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        output_dir=str(out / "gpu_run"),
        seed=seed,
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=2,
    )
    resume_trainer = Trainer(
        GPT(cfg),
        resume_cfg,
        loader,
        resume_from=gpu_ckpt,
        overwrite=True,
    )
    resume_trainer.train(resume_from=gpu_ckpt)
    gpu_resume = {
        "resumed_from_step": 2,
        "final_global_step": resume_trainer.global_step,
        "final_tokens_seen": resume_trainer.tokens_seen,
        "expected_tokens_seen": 2 * 16 * 1 * 4,
        "ok": resume_trainer.global_step == 4 and resume_trainer.tokens_seen == 2 * 16 * 4,
        "resume_class": "state-continuous",
    }

    # C. GPU checkpoint → CPU inspection (pre-resume snapshot; resume overwrites step-final.pt)
    inspect_model = GPT(cfg)
    inspect_opt = configure_optimizers(
        inspect_model,
        TrainingConfig(warmup_steps=0, device="cpu", precision="fp32", output_dir=str(out / "cpu_inspect")),
    )
    meta_c = load_checkpoint(gpu_ckpt_pre_resume, inspect_model, inspect_opt, device="cpu")
    gpu_to_cpu = {
        "checkpoint": str(gpu_ckpt_pre_resume),
        "global_step": meta_c["global_step"],
        "tokens_seen": meta_c["tokens_seen"],
        "first_param_device": str(next(inspect_model.parameters()).device),
        "ok": (not next(inspect_model.parameters()).is_cuda) and meta_c["global_step"] == 2,
    }

    return {
        "cpu_to_gpu": cpu_to_gpu,
        "gpu_resume": gpu_resume,
        "gpu_to_cpu": gpu_to_cpu,
        "checkpoint_resume_verified": bool(
            cpu_to_gpu["ok"] and gpu_resume["ok"] and gpu_to_cpu["ok"]
        ),
    }


def run_benchmark(
    *,
    preset: str,
    precision: str,
    attention_backend: str,
    context_length: int,
    batch_size: int,
    grad_accum_steps: int,
    warmup_steps: int,
    measured_steps: int,
    output_dir: Path | str,
    seed: int = 1337,
) -> dict[str, Any]:
    """Times measured optimizer steps with CUDA synchronize. Timing warmup != LR warmup."""
    _empty_cuda()
    seed_everything(seed)
    model_cfg = _model_config(preset, context_length, attention_backend)
    context_length = model_cfg.context_length
    tokens_per_step = batch_size * context_length * grad_accum_steps
    samples = batch_size * grad_accum_steps * (warmup_steps + measured_steps + 4)
    loader = make_synthetic_loader(
        vocab_size=model_cfg.vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        num_samples=max(samples, batch_size),
        seed=seed,
    )
    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=warmup_steps + measured_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        device="cuda",
        precision=precision,  # type: ignore[arg-type]
        output_dir=str(output_dir),
        seed=seed,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=10_000,
    )
    trainer = Trainer(GPT(model_cfg), train_cfg, loader, overwrite=True)
    data_iter = trainer._infinite_loader(loader)
    for _ in range(warmup_steps):
        trainer.train_step(data_iter)

    torch.cuda.reset_peak_memory_stats(trainer.device)
    torch.cuda.synchronize(trainer.device)
    start = time.perf_counter()
    losses: list[float] = []
    for _ in range(measured_steps):
        losses.append(trainer.train_step(data_iter)["loss"])
    torch.cuda.synchronize(trainer.device)
    elapsed = time.perf_counter() - start
    measured_tokens = measured_steps * tokens_per_step
    return {
        "preset": preset,
        "parameter_count": trainer.model.num_parameters() if hasattr(trainer.model, "num_parameters") else None,
        "precision": precision,
        "attention_backend": attention_backend,
        "micro_batch_size": batch_size,
        "context_length": context_length,
        "gradient_accumulation_steps": grad_accum_steps,
        "tokens_per_optimizer_step": tokens_per_step,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "total_measured_tokens": measured_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": measured_tokens / max(elapsed, 1e-6),
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(trainer.device)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(trainer.device)),
        "gpu_name": torch.cuda.get_device_name(trainer.device),
    }


def probe_microbatch_capacity(
    *,
    precision: str,
    attention_backend: str = "sdpa",
    context_length: int = 1024,
    grad_accum_steps: int = 1,
    batch_candidates: list[int] | None = None,
    steps: int = 2,
    seed: int = 1337,
    output_dir: Path | str = "runs/capacity_probe",
) -> list[dict[str, Any]]:
    """Probes GPT-2 Small micro-batch capacity. OOM is recorded as a measurement, not auto-retried at smaller B."""
    if batch_candidates is None:
        batch_candidates = [1, 2, 4, 8, 16]
    results: list[dict[str, Any]] = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    for batch_size in batch_candidates:
        _empty_cuda()
        record: dict[str, Any] = {
            "gpu": gpu_name,
            "precision": precision,
            "micro_batch_size": batch_size,
            "context_length": context_length,
            "gradient_accumulation_steps": grad_accum_steps,
            "status": "PASS",
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
            "oom_stage": None,
        }
        try:
            seed_everything(seed)
            model_cfg = GPTConfig.gpt2_small(
                context_length=context_length,
                attention_backend=attention_backend,
                dropout=0.0,
            )
            samples = max(batch_size * grad_accum_steps * (steps + 2), batch_size)
            loader = make_synthetic_loader(
                vocab_size=model_cfg.vocab_size,
                context_length=context_length,
                batch_size=batch_size,
                num_samples=samples,
                seed=seed,
            )
            train_cfg = TrainingConfig(
                warmup_steps=0,
                max_steps=steps,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                device="cuda",
                precision=precision,  # type: ignore[arg-type]
                output_dir=str(Path(output_dir) / f"{precision}_B{batch_size}"),
                seed=seed,
                log_interval=10_000,
                eval_interval=10_000,
                checkpoint_interval=10_000,
            )
            trainer = Trainer(GPT(model_cfg), train_cfg, loader, overwrite=True)
            data_iter = trainer._infinite_loader(loader)
            torch.cuda.reset_peak_memory_stats(trainer.device)
            torch.cuda.synchronize(trainer.device)
            last_loss = None
            for _ in range(steps):
                last_loss = trainer.train_step(data_iter)["loss"]
            torch.cuda.synchronize(trainer.device)
            record["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated(trainer.device))
            record["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved(trainer.device))
            record["loss"] = last_loss
            del trainer
            _empty_cuda()
        except OOM_EXCEPTIONS as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            record["status"] = "OOM"
            record["oom_stage"] = _classify_oom_stage(exc)
            if torch.cuda.is_available():
                try:
                    record["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated())
                    record["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved())
                except Exception:
                    pass
            _empty_cuda()
            results.append(record)
            # Larger B will also OOM; still record remaining candidates as skipped-after-oom.
            for larger in batch_candidates[batch_candidates.index(batch_size) + 1 :]:
                results.append(
                    {
                        "gpu": gpu_name,
                        "precision": precision,
                        "micro_batch_size": larger,
                        "context_length": context_length,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "status": "OOM",
                        "peak_allocated_vram_bytes": None,
                        "peak_reserved_vram_bytes": None,
                        "oom_stage": "not_attempted_after_smaller_oom",
                    }
                )
            break
        results.append(record)
    return results


def largest_passing_batch(probe_rows: list[dict[str, Any]]) -> int | None:
    """Returns the largest micro-batch that PASSed, or None."""
    passing = [int(row["micro_batch_size"]) for row in probe_rows if row.get("status") == "PASS"]
    return max(passing) if passing else None


def save_gpu_qualification_summary(path: Path | str, payload: dict[str, Any]) -> Path:
    """Writes the machine-readable GPU qualification summary."""
    return atomic_save_json(path, payload)
