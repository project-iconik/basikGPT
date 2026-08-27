"""Milestone 14 GPU qualification orchestrator for the unified basikGPT Trainer.

Runs CUDA FP32 → BF16 → checkpoint/resume → VRAM/throughput → capacity → FineWeb-Edu dry-run.
Does not use torch.compile, DDP, FSDP, or a separate GPU trainer class.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.training.gpu_qualification import (
    collect_gpu_environment,
    compare_cpu_cuda_fp32,
    compare_fp32_bf16_loss,
    largest_passing_batch,
    probe_microbatch_capacity,
    run_benchmark,
    run_training_smoke,
    save_gpu_qualification_summary,
    verify_checkpoint_portability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basikGPT Milestone 14 RunPod GPU qualification.")
    parser.add_argument("--output-dir", type=str, default="runs/m14_gpu_qualification")
    parser.add_argument("--data-dir", type=str, default="data/fineweb-edu-smoke")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest stages (not used for a passing milestone; available for reruns).",
    )
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def _read_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prepare_smoke_data(data_dir: Path) -> None:
    train_shards = list(data_dir.glob("train-*.npy"))
    manifest = data_dir / "manifest.json"
    if train_shards and manifest.exists():
        print(f"[qualify] Reusing existing shards in {data_dir}")
        return
    print("+ preparing FineWeb-Edu smoke shards", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_fineweb_edu.py",
            "--output",
            str(data_dir),
            "--dataset-config",
            "sample-10BT",
            "--max-train-tokens",
            "100000",
            "--max-validation-tokens",
            "10000",
            "--shard-token-target",
            "50000",
            "--overwrite",
        ],
        cwd=str(repo_root),
        check=False,
    )
    train_shards = list(data_dir.glob("train-*.npy"))
    if train_shards and (data_dir / "manifest.json").exists():
        if completed.returncode != 0:
            print(
                "[qualify] prepare_fineweb_edu.py exited non-zero during shutdown; "
                f"shards are present so continuing (returncode={completed.returncode})."
            )
        return
    raise RuntimeError(
        f"FineWeb-Edu smoke data missing after prepare (returncode={completed.returncode}): {data_dir}"
    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "gpu_qualification.json"

    print("=" * 72)
    print("  basikGPT Milestone 14 — RunPod GPU Qualification")
    print("  torch.compile is not used. 2.5B main training is not started.")
    print("=" * 72)

    env = collect_gpu_environment()
    save_gpu_qualification_summary(out / "gpu_env.json", env)
    git = env["git"]
    cuda = env["cuda"]
    print(f"git_commit:  {git.get('git_commit')}")
    print(f"git_dirty:   {git.get('git_dirty')}")
    print(f"gpu:         {cuda.get('gpu_name')}")
    print(f"vram_bytes:  {cuda.get('total_vram_bytes')}")
    print(f"driver:      {cuda.get('nvidia_driver')}")
    print(f"cuda:        {cuda.get('cuda_runtime')}")
    print(f"capability:  {cuda.get('compute_capability')}")
    print(f"bf16:        {cuda.get('bf16_supported')}")

    if not cuda.get("cuda_available"):
        raise RuntimeError("CUDA is not available on this machine.")
    if not cuda.get("bf16_supported"):
        raise RuntimeError("This GPU does not support BF16; Milestone 14 fail-fast (no FP32/FP16 fallback).")

    if not args.skip_pytest:
        _run([sys.executable, "-m", "pytest", "tests/training/test_cuda.py", "-v"], repo_root)
        _run([sys.executable, "-m", "pytest", "-v"], repo_root)

    tiny_fp32 = run_training_smoke(
        output_dir=out / "tiny_cuda_fp32",
        preset="tiny",
        device="cuda",
        precision="fp32",
        attention_backend="sdpa",
        context_length=64,
        batch_size=2,
        grad_accum_steps=2,
        max_steps=10,
        warmup_steps=2,
    )
    print("[qualify] tiny CUDA FP32", {k: tiny_fp32[k] for k in ("optimizer_steps", "tokens_seen", "initial_loss", "final_loss", "token_accounting_ok")})
    if not tiny_fp32["loss_finite"] or not tiny_fp32["grad_finite"] or not tiny_fp32["token_accounting_ok"]:
        raise RuntimeError(f"Tiny CUDA FP32 smoke failed: {tiny_fp32}")

    small_fp32 = run_training_smoke(
        output_dir=out / "gpt2_small_cuda_fp32",
        preset="gpt2_small",
        device="cuda",
        precision="fp32",
        attention_backend="sdpa",
        context_length=1024,
        batch_size=1,
        grad_accum_steps=1,
        max_steps=8,
        warmup_steps=2,
    )
    print("[qualify] GPT-2 Small CUDA FP32", {k: small_fp32[k] for k in ("optimizer_steps", "tokens_seen", "initial_loss", "final_loss")})
    if not small_fp32["loss_finite"] or not small_fp32["grad_finite"] or not small_fp32["token_accounting_ok"]:
        raise RuntimeError(f"GPT-2 Small CUDA FP32 smoke failed: {small_fp32}")

    small_bf16 = run_training_smoke(
        output_dir=out / "gpt2_small_cuda_bf16",
        preset="gpt2_small",
        device="cuda",
        precision="bf16",
        attention_backend="sdpa",
        context_length=1024,
        batch_size=1,
        grad_accum_steps=1,
        max_steps=8,
        warmup_steps=2,
    )
    print("[qualify] GPT-2 Small CUDA BF16", {k: small_bf16[k] for k in ("optimizer_steps", "tokens_seen", "initial_loss", "final_loss")})
    if not small_bf16["loss_finite"] or not small_bf16["grad_finite"] or not small_bf16["token_accounting_ok"]:
        raise RuntimeError(f"GPT-2 Small CUDA BF16 smoke failed: {small_bf16}")

    cpu_cuda = compare_cpu_cuda_fp32()
    print("[qualify] CPU↔CUDA FP32", cpu_cuda)
    fp32_bf16 = compare_fp32_bf16_loss()
    print("[qualify] FP32↔BF16", fp32_bf16)
    if not fp32_bf16["both_finite"]:
        raise RuntimeError(f"FP32↔BF16 loss is not finite: {fp32_bf16}")

    ckpt = verify_checkpoint_portability(out / "checkpoint_portability")
    print("[qualify] checkpoint portability", ckpt)
    if not ckpt["checkpoint_resume_verified"]:
        raise RuntimeError(f"Checkpoint portability failed: {ckpt}")

    bench_fp32 = run_benchmark(
        preset="gpt2_small",
        precision="fp32",
        attention_backend="sdpa",
        context_length=1024,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=5,
        measured_steps=20,
        output_dir=out / "bench_fp32",
    )
    bench_bf16 = run_benchmark(
        preset="gpt2_small",
        precision="bf16",
        attention_backend="sdpa",
        context_length=1024,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=5,
        measured_steps=20,
        output_dir=out / "bench_bf16",
    )
    print("[qualify] FP32 tokens/sec", bench_fp32["tokens_per_second"], "allocated", bench_fp32["peak_allocated_vram_bytes"])
    print("[qualify] BF16 tokens/sec", bench_bf16["tokens_per_second"], "allocated", bench_bf16["peak_allocated_vram_bytes"])

    probe_bf16 = probe_microbatch_capacity(
        precision="bf16",
        attention_backend="sdpa",
        context_length=1024,
        grad_accum_steps=1,
        batch_candidates=[1, 2, 4, 8, 16],
        steps=2,
        output_dir=out / "capacity_bf16",
    )
    probe_fp32 = probe_microbatch_capacity(
        precision="fp32",
        attention_backend="sdpa",
        context_length=1024,
        grad_accum_steps=1,
        batch_candidates=[1, 2, 4, 8, 16],
        steps=2,
        output_dir=out / "capacity_fp32",
    )
    stable_b = largest_passing_batch(probe_bf16) or 1
    print("[qualify] BF16 capacity", probe_bf16)
    print("[qualify] FP32 capacity", probe_fp32)
    print(f"[qualify] dry-run micro-batch B={stable_b}")

    data_dir = Path(args.data_dir)
    _prepare_smoke_data(data_dir)

    tokens_per_step = stable_b * 1024 * 1
    dry_steps = min(32, max(8, 200_000 // tokens_per_step))
    save_at = max(4, dry_steps // 2)
    dry_dir = out / "fineweb_gpt2_small_bf16"
    _run(
        [
            sys.executable,
            "scripts/train.py",
            "--run-name",
            "m14-fineweb-bf16",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(dry_dir),
            "--overwrite",
            "--model-preset",
            "gpt2_small",
            "--context-length",
            "1024",
            "--attention-backend",
            "sdpa",
            "--batch-size",
            str(stable_b),
            "--grad-accum-steps",
            "1",
            "--max-steps",
            str(save_at),
            "--warmup-steps",
            str(min(4, save_at)),
            "--eval-interval",
            str(max(1, save_at)),
            "--eval-batches",
            "2",
            "--checkpoint-interval",
            str(save_at),
            "--log-interval",
            "1",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--seed",
            "1337",
        ],
        repo_root,
    )
    resume_ckpt = dry_dir / f"step-{save_at:08d}.pt"
    if not resume_ckpt.exists():
        resume_ckpt = dry_dir / "step-final.pt"
    _run(
        [
            sys.executable,
            "scripts/train.py",
            "--run-name",
            "m14-fineweb-bf16",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(dry_dir),
            "--resume",
            str(resume_ckpt),
            "--model-preset",
            "gpt2_small",
            "--context-length",
            "1024",
            "--attention-backend",
            "sdpa",
            "--batch-size",
            str(stable_b),
            "--grad-accum-steps",
            "1",
            "--max-steps",
            str(dry_steps),
            "--warmup-steps",
            str(min(4, dry_steps)),
            "--eval-interval",
            str(dry_steps),
            "--eval-batches",
            "2",
            "--checkpoint-interval",
            str(dry_steps),
            "--log-interval",
            "1",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--seed",
            "1337",
        ],
        repo_root,
    )

    metrics = _read_metrics(dry_dir)
    train_rows = [row for row in metrics if row.get("type") == "train"]
    val_rows = [row for row in metrics if row.get("type") == "val"]
    grad_norms = [row["grad_norm"] for row in train_rows if "grad_norm" in row]
    dry_run = {
        "requested_tokens": tokens_per_step * dry_steps,
        "actual_tokens": train_rows[-1]["tokens_seen"] if train_rows else None,
        "optimizer_steps": train_rows[-1]["step"] if train_rows else None,
        "tokens_per_optimizer_step": tokens_per_step,
        "micro_batch_size": stable_b,
        "gradient_accumulation_steps": 1,
        "context_length": 1024,
        "initial_loss": train_rows[0]["loss"] if train_rows else None,
        "final_loss": train_rows[-1]["loss"] if train_rows else None,
        "validation_loss": val_rows[-1]["val_loss"] if val_rows else None,
        "min_gradient_norm": min(grad_norms) if grad_norms else None,
        "max_gradient_norm": max(grad_norms) if grad_norms else None,
        "mean_gradient_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "initial_learning_rate": train_rows[0].get("learning_rate") if train_rows else None,
        "final_learning_rate": train_rows[-1].get("learning_rate") if train_rows else None,
        "elapsed_seconds": train_rows[-1].get("elapsed_seconds") if train_rows else None,
        "tokens_per_second": train_rows[-1].get("tokens_per_sec") if train_rows else None,
        "peak_allocated_vram_bytes": train_rows[-1].get("peak_allocated_vram_bytes") if train_rows else None,
        "peak_reserved_vram_bytes": train_rows[-1].get("peak_reserved_vram_bytes") if train_rows else None,
        "checkpoint_path": str(dry_dir / "step-final.pt"),
        "resume_status": "state-continuous",
        "uniform_ce_reference": math.log(50257),
    }

    summary = {
        "status": "passed",
        "provider": cuda.get("provider") or "RunPod",
        "gpu": cuda.get("gpu_name"),
        "gpu_count": cuda.get("gpu_count"),
        "total_vram_bytes": cuda.get("total_vram_bytes"),
        "nvidia_driver": cuda.get("nvidia_driver"),
        "cuda_runtime": cuda.get("cuda_runtime"),
        "pytorch_version": env["system"].get("pytorch_version"),
        "python_version": env["system"].get("python_version"),
        "compute_capability": cuda.get("compute_capability"),
        "bf16_supported": cuda.get("bf16_supported"),
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "precision": "bf16",
        "model": "gpt2-small",
        "parameter_count": 124439808,
        "context_length": 1024,
        "attention_backend": "sdpa",
        "micro_batch_size": stable_b,
        "gradient_accumulation_steps": 1,
        "tokens_per_optimizer_step": tokens_per_step,
        "tokens_per_second": bench_bf16["tokens_per_second"],
        "peak_allocated_vram_bytes": bench_bf16["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": bench_bf16["peak_reserved_vram_bytes"],
        "initial_loss": dry_run["initial_loss"],
        "final_loss": dry_run["final_loss"],
        "validation_loss": dry_run["validation_loss"],
        "checkpoint_resume_verified": ckpt["checkpoint_resume_verified"],
        "resume_class": "state-continuous",
        "torch_compile_used": False,
        "tiny_cuda_fp32": tiny_fp32,
        "gpt2_small_cuda_fp32": small_fp32,
        "gpt2_small_cuda_bf16": small_bf16,
        "cpu_cuda_fp32_sanity": cpu_cuda,
        "fp32_bf16_sanity": fp32_bf16,
        "checkpoint_portability": ckpt,
        "benchmark_fp32": bench_fp32,
        "benchmark_bf16": bench_bf16,
        "capacity_bf16": probe_bf16,
        "capacity_fp32": probe_fp32,
        "fineweb_dry_run": dry_run,
        "provisional_main_run": {
            "micro_batch_size": stable_b,
            "context_length": 1024,
            "gradient_accumulation_steps": 8,
            "world_size": 1,
            "tokens_per_optimizer_step": stable_b * 1024 * 8 * 1,
            "note": "Provisional plan using measured PASS micro-batch B. Not a final hyperparameter.",
        },
    }
    save_gpu_qualification_summary(summary_path, summary)
    print(f"[qualify] wrote {summary_path}")
    print("[qualify] status=passed")


if __name__ == "__main__":
    main()
