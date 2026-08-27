"""Single-GPU performance probes: compile timing, SDPA backends, and batch sweeps.

Not a production benchmark framework. Helpers wrap the existing Trainer.
The Milestone 14 uncompiled `gpu_qualification.run_benchmark` path is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import time
import gc
import torch

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.accounting import calculate_compile_break_even_tokens
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.compile import state_dict_has_compile_wrapper_keys, unwrap_compiled_model
from basikgpt.training.config import TrainingConfig
from basikgpt.training.gpu_qualification import OOM_EXCEPTIONS, make_synthetic_loader
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import atomic_save_json
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.reproducibility import get_git_metadata, seed_everything
from basikgpt.training.sdpa import list_probe_sdpa_backends, normalize_sdpa_kernel_name
from basikgpt.training.trainer import Trainer


BYTES_PER_GIB = 1024 ** 3


def empty_cuda() -> None:
    """Releases cached CUDA memory and resets peak stats so runs do not leak peaks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _model_config(preset: str, context_length: int, attention_backend: str) -> GPTConfig:
    if preset == "tiny":
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
    return GPTConfig.gpt2_small(
        context_length=context_length,
        attention_backend=attention_backend,
        dropout=0.0,
    )


def _env_fields() -> dict[str, Any]:
    git = get_git_metadata()
    payload: dict[str, Any] = {
        "git_sha": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        payload["compute_capability"] = f"{cap[0]}.{cap[1]}"
    else:
        payload["gpu"] = None
        payload["compute_capability"] = None
    return payload


def _classify_status(exc: BaseException, *, compiled: bool) -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "OOM"
    if compiled:
        return "COMPILE_FAILED"
    return type(exc).__name__


def _vram_efficiency(tokens_per_second: float | None, peak_allocated: int | None) -> float | None:
    if tokens_per_second is None or peak_allocated is None or peak_allocated <= 0:
        return None
    return tokens_per_second / (peak_allocated / BYTES_PER_GIB)


def run_performance_benchmark(
    *,
    output_dir: Path | str,
    preset: str = "gpt2_small",
    precision: str = "bf16",
    attention_backend: str = "sdpa",
    sdpa_kernel: str = "auto",
    compiled: bool = False,
    compile_mode: str = "default",
    context_length: int = 1024,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    warmup_steps: int = 5,
    measured_steps: int = 20,
    seed: int = 1337,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Times one controlled training configuration, including compile first-step cost.

    Distinguishes:
        time_to_first_optimizer_step  (includes tracing/compilation)
        warmup_elapsed
        steady_state_tokens_per_second  (measured window only)
        end_to_end_tokens_per_second_including_compile
    """
    empty_cuda()
    seed_everything(seed)
    sdpa_kernel = normalize_sdpa_kernel_name(sdpa_kernel)
    model_cfg = _model_config(preset, context_length, attention_backend)
    context_length = model_cfg.context_length
    tokens_per_step = batch_size * context_length * grad_accum_steps
    total_steps = warmup_steps + measured_steps
    samples = max(batch_size * grad_accum_steps * (total_steps + 4), batch_size)
    loader = make_synthetic_loader(
        vocab_size=model_cfg.vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        num_samples=samples,
        seed=seed,
    )
    name = experiment_name or (
        f"{preset}_{precision}_B{batch_size}_G{grad_accum_steps}"
        f"{'_compile_' + compile_mode if compiled else '_eager'}"
        f"_{attention_backend}_{sdpa_kernel}"
    )
    record: dict[str, Any] = {
        **_env_fields(),
        "experiment_name": name,
        "preset": preset,
        "precision": precision,
        "compiled": compiled,
        "compile_mode": compile_mode if compiled else None,
        "attention_backend": attention_backend,
        "sdpa_forced_backend": None if sdpa_kernel == "auto" else sdpa_kernel,
        "sdpa_kernel": sdpa_kernel,
        "B": batch_size,
        "T": context_length,
        "G": grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "compile_seconds": None,
        "time_to_first_optimizer_step": None,
        "warmup_elapsed": None,
        "steady_state_tokens_per_second": None,
        "end_to_end_tokens_per_second_including_compile": None,
        "ms_per_step": None,
        "peak_allocated": None,
        "peak_reserved": None,
        "peak_allocated_compile_phase": None,
        "tokens_per_sec_per_gib_allocated": None,
        "mean_loss": None,
        "status": "PASS",
        "error": None,
    }

    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=max(total_steps, 1),
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        device="cuda",
        precision=precision,  # type: ignore[arg-type]
        compile=compiled,
        compile_mode=compile_mode,
        sdpa_kernel=sdpa_kernel,
        output_dir=str(output_dir),
        seed=seed,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=10_000,
    )

    trainer: Trainer | None = None
    try:
        trainer = Trainer(GPT(model_cfg), train_cfg, loader, overwrite=True)
        record["parameter_count"] = trainer.raw_model.num_parameters() if hasattr(trainer.raw_model, "num_parameters") else None
        data_iter = trainer._infinite_loader(loader)

        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
            torch.cuda.synchronize(trainer.device)

        e2e_start = time.perf_counter()
        first = trainer.train_step(data_iter)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        time_to_first = time.perf_counter() - e2e_start
        record["time_to_first_optimizer_step"] = time_to_first
        if trainer.device.type == "cuda":
            record["peak_allocated_compile_phase"] = int(torch.cuda.max_memory_allocated(trainer.device))

        remaining_warmup = max(warmup_steps - 1, 0)
        measured_from_first = warmup_steps <= 0
        for _ in range(remaining_warmup):
            trainer.train_step(data_iter)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        warmup_elapsed = time.perf_counter() - e2e_start
        record["warmup_elapsed"] = warmup_elapsed

        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
            torch.cuda.synchronize(trainer.device)

        losses: list[float] = [first["loss"]] if measured_from_first else []
        remaining_measured = measured_steps - (1 if measured_from_first else 0)
        measured_start = time.perf_counter()
        for _ in range(max(remaining_measured, 0)):
            losses.append(trainer.train_step(data_iter)["loss"])
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        measured_elapsed = time.perf_counter() - measured_start
        if measured_from_first:
            measured_elapsed = time.perf_counter() - e2e_start
        e2e_elapsed = time.perf_counter() - e2e_start

        measured_tokens = measured_steps * tokens_per_step
        e2e_tokens = (warmup_steps + measured_steps) * tokens_per_step
        steady = measured_tokens / max(measured_elapsed, 1e-6)
        record["steady_state_tokens_per_second"] = steady
        record["end_to_end_tokens_per_second_including_compile"] = e2e_tokens / max(e2e_elapsed, 1e-6)
        record["ms_per_step"] = (measured_elapsed / max(measured_steps, 1)) * 1000.0
        record["mean_loss"] = sum(losses) / len(losses) if losses else None
        mean_steady_step = measured_elapsed / max(measured_steps, 1)
        record["compile_seconds"] = max(time_to_first - mean_steady_step, 0.0) if compiled else 0.0
        if trainer.device.type == "cuda":
            steady_alloc = int(torch.cuda.max_memory_allocated(trainer.device))
            steady_reserved = int(torch.cuda.max_memory_reserved(trainer.device))
            compile_alloc = record.get("peak_allocated_compile_phase")
            alloc_candidates = [value for value in (steady_alloc, compile_alloc) if isinstance(value, int)]
            record["peak_allocated"] = max(alloc_candidates) if alloc_candidates else steady_alloc
            record["peak_reserved"] = steady_reserved
            record["peak_allocated_steady_state"] = steady_alloc
        record["tokens_per_sec_per_gib_allocated"] = _vram_efficiency(
            record["steady_state_tokens_per_second"], record["peak_allocated"]
        )
        if not math.isfinite(first["loss"]) or any(not math.isfinite(x) for x in losses):
            record["status"] = "NON_FINITE"
        del trainer
        empty_cuda()
        return record
    except OOM_EXCEPTIONS as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
            if compiled:
                record["status"] = "COMPILE_FAILED"
                record["error"] = str(exc)
                empty_cuda()
                return record
            raise
        record["status"] = "OOM"
        record["error"] = str(exc)
        if torch.cuda.is_available():
            try:
                record["peak_allocated"] = int(torch.cuda.max_memory_allocated())
                record["peak_reserved"] = int(torch.cuda.max_memory_reserved())
            except Exception:
                pass
        empty_cuda()
        return record
    except Exception as exc:
        record["status"] = _classify_status(exc, compiled=compiled)
        if compiled and record["status"] not in ("OOM", "COMPILE_FAILED"):
            record["status"] = "COMPILE_FAILED"
        record["error"] = str(exc)
        empty_cuda()
        return record


def probe_sdpa_backend(
    backend_name: str,
    *,
    output_dir: Path | str,
    batch_size: int = 8,
    context_length: int = 1024,
    precision: str = "bf16",
    warmup_steps: int = 3,
    measured_steps: int = 10,
    seed: int = 1337,
) -> dict[str, Any]:
    """Forces one SDPA backend exclusively. Fallback success is not counted as support."""
    kernel = normalize_sdpa_kernel_name(backend_name)
    available = {normalize_sdpa_kernel_name(name) for name in list_probe_sdpa_backends()}
    if kernel not in available and kernel != "auto":
        return {
            **_env_fields(),
            "sdpa_forced_backend": kernel,
            "status": "UNSUPPORTED",
            "error": f"{backend_name} is not present in this PyTorch SDPBackend enum",
            "B": batch_size,
            "T": context_length,
            "G": 1,
            "compiled": False,
        }
    result = run_performance_benchmark(
        output_dir=output_dir,
        precision=precision,
        attention_backend="sdpa",
        sdpa_kernel=kernel,
        compiled=False,
        context_length=context_length,
        batch_size=batch_size,
        grad_accum_steps=1,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        seed=seed,
        experiment_name=f"sdpa_{kernel}_B{batch_size}",
    )
    # Exclusive kernel raise paths are classified as UNSUPPORTED, not a generic crash.
    if result["status"] not in ("PASS", "OOM", "NON_FINITE") and kernel != "auto":
        result["status"] = "UNSUPPORTED"
    return result


def compare_compiled_logits(
    *,
    device: str = "cuda",
    seed: int = 1337,
) -> dict[str, Any]:
    """Compares uncompiled vs compiled FP32 logits on a tiny deterministic model.

    Bitwise equality is not required. Non-finite or catastrophic divergence fails.
    """
    seed_everything(seed)
    cfg = GPTConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        attention_backend="sdpa",
        dropout=0.0,
    )
    raw = GPT(cfg).to(device)
    compiled_src = GPT(cfg).to(device)
    compiled_src.load_state_dict(raw.state_dict())
    from basikgpt.training.compile import compile_model

    compiled = compile_model(compiled_src, mode="default")
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    raw.eval()
    compiled.eval()
    with torch.inference_mode():
        logits_raw = raw(x)
        logits_compiled = compiled(x)
    max_abs = float((logits_raw.float() - logits_compiled.float()).abs().max().item())
    finite = bool(torch.isfinite(logits_raw).all() and torch.isfinite(logits_compiled).all())
    return {
        "max_abs_diff": max_abs,
        "finite": finite,
        "ok": finite and max_abs < 1e-2,
    }


def verify_compiled_checkpoint_roundtrip(
    *,
    output_dir: Path | str,
    device: str = "cuda",
    precision: str = "bf16",
    seed: int = 1337,
) -> dict[str, Any]:
    """Compiled train → checkpoint without wrapper keys → uncompiled load + compiled resume."""
    empty_cuda()
    seed_everything(seed)
    cfg = GPTConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        attention_backend="sdpa",
        dropout=0.0,
    )
    loader = make_synthetic_loader(cfg.vocab_size, cfg.context_length, batch_size=2, num_samples=16, seed=seed)
    out = Path(output_dir)
    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        precision=precision,  # type: ignore[arg-type]
        compile=True,
        compile_mode="default",
        output_dir=str(out),
        seed=seed,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=1,
    )
    trainer = Trainer(GPT(cfg), train_cfg, loader, overwrite=True)
    data_iter = trainer._infinite_loader(loader)
    trainer.train_step(data_iter)
    ckpt = out / "compiled_roundtrip.pt"
    save_checkpoint(
        ckpt,
        trainer.raw_model,
        trainer.optimizer,
        global_step=1,
        tokens_seen=trainer.tokens_seen,
        training_config=train_cfg,
        model_config=cfg,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    has_wrapper = state_dict_has_compile_wrapper_keys(payload["model_state_dict"])

    fresh = GPT(cfg)
    opt = configure_optimizers(fresh, TrainingConfig(warmup_steps=0, device="cpu", compile=False))
    load_checkpoint(ckpt, fresh, opt, device="cpu")

    resume_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        precision=precision,  # type: ignore[arg-type]
        compile=True,
        compile_mode="default",
        output_dir=str(out / "resume"),
        seed=seed,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=10_000,
    )
    resumed = Trainer(GPT(cfg), resume_cfg, loader, overwrite=True)
    load_checkpoint(ckpt, resumed.raw_model, resumed.optimizer, scaler=resumed.scaler, device=resumed.device)
    step = resumed.train_step(resumed._infinite_loader(loader))
    return {
        "wrapper_keys_in_checkpoint": has_wrapper,
        "uncompiled_load_ok": True,
        "compiled_resume_loss_finite": math.isfinite(step["loss"]),
        "unwrapped_is_raw": unwrap_compiled_model(trainer.model) is trainer.raw_model,
    }


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()


def save_performance_summary(path: Path | str, payload: dict[str, Any]) -> Path:
    return atomic_save_json(path, payload)


def attach_speedup(record: dict[str, Any], baseline_tokens_per_second: float | None) -> dict[str, Any]:
    """Adds speedup vs a same-milestone baseline throughput. Does not mutate status."""
    steady = record.get("steady_state_tokens_per_second")
    if (
        baseline_tokens_per_second is None
        or baseline_tokens_per_second <= 0
        or not isinstance(steady, (int, float))
        or record.get("status") != "PASS"
    ):
        record["speedup_vs_baseline"] = None
        record["break_even_tokens"] = None
        return record
    record["speedup_vs_baseline"] = steady / baseline_tokens_per_second
    compile_seconds = record.get("compile_seconds") or 0.0
    if record.get("compiled"):
        record["break_even_tokens"] = calculate_compile_break_even_tokens(
            compile_overhead_seconds=float(compile_seconds),
            baseline_tokens_per_second=float(baseline_tokens_per_second),
            compiled_tokens_per_second=float(steady),
        )
    else:
        record["break_even_tokens"] = None
    return record
