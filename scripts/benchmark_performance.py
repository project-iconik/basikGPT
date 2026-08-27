"""Milestone 15 performance orchestrator: compile, SDPA, batch, and G sweeps.

Preserves the Milestone 14 uncompiled path. Does not start DDP, FSDP, or 2.5B training.
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

from basikgpt.training.accounting import calculate_training_steps
from basikgpt.training.gpu_qualification import collect_gpu_environment
from basikgpt.training.performance import (
    append_jsonl,
    attach_speedup,
    compare_compiled_logits,
    probe_sdpa_backend,
    run_performance_benchmark,
    save_performance_summary,
    verify_compiled_checkpoint_roundtrip,
)
from basikgpt.training.sdpa import list_probe_sdpa_backends
from basikgpt.training.metadata import atomic_save_json


TARGET_TOKENS_2_5B = 2_500_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basikGPT Milestone 15 GPU performance benchmark.")
    parser.add_argument("--output-dir", type=str, default="runs/m15_performance")
    parser.add_argument("--data-dir", type=str, default="data/fineweb-edu-smoke")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip local pytest (not used for a passing milestone).",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip FineWeb-Edu short stability runs.",
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
        print(f"[m15] Reusing existing shards in {data_dir}")
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
        return
    raise RuntimeError(
        f"FineWeb-Edu smoke data missing after prepare (returncode={completed.returncode}): {data_dir}"
    )


def _record(jsonl: Path, row: dict, baseline_tps: float | None) -> dict:
    attach_speedup(row, baseline_tps)
    append_jsonl(jsonl, row)
    status = row.get("status")
    tps = row.get("steady_state_tokens_per_second")
    print(
        f"[m15] {row.get('experiment_name')} status={status} "
        f"tok/s={tps} compile_s={row.get('compile_seconds')} "
        f"peak_alloc={row.get('peak_allocated')}",
        flush=True,
    )
    return row


def _tokens_per_step(batch_size: int, grad_accum: int, context_length: int = 1024) -> int:
    return batch_size * context_length * grad_accum


def _plan_2_5b(batch_size: int, grad_accum: int) -> dict:
    plan = calculate_training_steps(
        target_tokens=TARGET_TOKENS_2_5B,
        micro_batch_size=batch_size,
        context_length=1024,
        grad_accum_steps=grad_accum,
        world_size=1,
    )
    return plan.to_dict()


def _pick_candidates(rows: list[dict]) -> list[dict]:
    passing = [
        row
        for row in rows
        if row.get("status") == "PASS"
        and row.get("steady_state_tokens_per_second")
        and row.get("attention_backend") == "sdpa"
        and (row.get("sdpa_kernel") in (None, "auto"))
    ]
    # tok/s per GiB is highest at tiny B; that is not a main-run candidate.
    relevant = [row for row in passing if int(row.get("B") or 0) >= 8]
    pool = relevant or passing
    if not pool:
        return []
    by_tps = max(pool, key=lambda row: row["steady_state_tokens_per_second"])
    with_eff = [row for row in pool if row.get("tokens_per_sec_per_gib_allocated")]
    by_eff = max(with_eff, key=lambda row: row["tokens_per_sec_per_gib_allocated"]) if with_eff else by_tps
    conservative = None
    for row in passing:
        if (
            not row.get("compiled")
            and row.get("B") == 8
            and row.get("G") == 8
            and row.get("sdpa_kernel") == "auto"
        ):
            conservative = row
            break
    if conservative is None:
        uncompiled = [row for row in pool if not row.get("compiled")]
        conservative = uncompiled[-1] if uncompiled else by_tps

    picked: list[dict] = []
    for label, row in (
        ("best_raw_throughput", by_tps),
        ("best_vram_efficient", by_eff),
        ("best_conservative_main_run", conservative),
    ):
        key = (
            row.get("compiled"),
            row.get("compile_mode"),
            row.get("B"),
            row.get("G"),
            row.get("sdpa_kernel"),
            row.get("attention_backend"),
        )
        existing = next((item for item in picked if item["key"] == key), None)
        if existing:
            existing["labels"].append(label)
            continue
        picked.append({"label": label, "labels": [label], "key": key, "row": row})
    return picked[:2]


def _stability_run(
    *,
    data_dir: Path,
    out_dir: Path,
    candidate: dict,
    target_tokens: int = 196_608,
) -> dict:
    row = candidate["row"]
    batch_size = int(row["B"])
    grad_accum = int(row.get("G") or 1)
    tokens_per_step = _tokens_per_step(batch_size, grad_accum)
    steps = max(math.ceil(target_tokens / tokens_per_step), 4)
    save_at = max(steps // 2, 1)
    run_dir = out_dir / f"stability_{candidate['labels'][0]}"
    cmd_base = [
        sys.executable,
        "scripts/train.py",
        "--data-dir",
        str(data_dir),
        "--model-preset",
        "gpt2_small",
        "--context-length",
        "1024",
        "--attention-backend",
        str(row.get("attention_backend") or "sdpa"),
        "--sdpa-kernel",
        str(row.get("sdpa_kernel") or "auto"),
        "--batch-size",
        str(batch_size),
        "--grad-accum-steps",
        str(grad_accum),
        "--warmup-steps",
        str(min(2, steps)),
        "--eval-interval",
        str(max(save_at, 1)),
        "--eval-batches",
        "2",
        "--log-interval",
        "1",
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--seed",
        "1337",
    ]
    if row.get("compiled"):
        cmd_base.extend(["--compile", "--compile-mode", str(row.get("compile_mode") or "default")])

    first = cmd_base + [
        "--run-name",
        f"m15-stability-{candidate['labels'][0]}",
        "--output-dir",
        str(run_dir),
        "--overwrite",
        "--max-steps",
        str(save_at),
        "--checkpoint-interval",
        str(save_at),
    ]
    _run(first, repo_root)
    resume_ckpt = run_dir / f"step-{save_at:08d}.pt"
    if not resume_ckpt.exists():
        resume_ckpt = run_dir / "step-final.pt"
    second = cmd_base + [
        "--run-name",
        f"m15-stability-{candidate['labels'][0]}",
        "--output-dir",
        str(run_dir),
        "--resume",
        str(resume_ckpt),
        "--max-steps",
        str(steps),
        "--checkpoint-interval",
        str(steps),
    ]
    _run(second, repo_root)
    metrics = _read_metrics(run_dir)
    train_rows = [item for item in metrics if item.get("type") == "train"]
    val_rows = [item for item in metrics if item.get("type") == "val"]
    losses = [item["loss"] for item in train_rows if math.isfinite(item.get("loss", float("nan")))]
    return {
        "labels": candidate["labels"],
        "config": {
            "compiled": bool(row.get("compiled")),
            "compile_mode": row.get("compile_mode"),
            "sdpa_kernel": row.get("sdpa_kernel"),
            "attention_backend": row.get("attention_backend"),
            "B": batch_size,
            "G": grad_accum,
            "T": 1024,
            "tokens_per_step": tokens_per_step,
        },
        "optimizer_steps": steps,
        "requested_tokens": tokens_per_step * steps,
        "actual_tokens": train_rows[-1]["tokens_seen"] if train_rows else None,
        "initial_loss": train_rows[0]["loss"] if train_rows else None,
        "final_loss": train_rows[-1]["loss"] if train_rows else None,
        "validation_loss": val_rows[-1]["val_loss"] if val_rows else None,
        "loss_finite": bool(losses) and len(losses) == len(train_rows),
        "resume_ok": resume_ckpt.exists() and len(train_rows) >= save_at,
        "tokens_per_second": train_rows[-1].get("tokens_per_sec") if train_rows else None,
        "peak_allocated_vram_bytes": train_rows[-1].get("peak_allocated_vram_bytes") if train_rows else None,
        "checkpoint_path": str(run_dir / "step-final.pt"),
        "main_run_2_5b": _plan_2_5b(batch_size, grad_accum),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "benchmarks.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    benches_dir = out / "experiments"
    benches_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("  basikGPT Milestone 15 — GPU Performance Engineering")
    print("  Uncompiled BF16+SDPA remains the canonical baseline.")
    print("  DDP / FSDP / 2.5B main training are not started.")
    print("=" * 72)

    env = collect_gpu_environment()
    save_performance_summary(out / "gpu_env.json", env)
    print(f"git_commit: {env['git'].get('git_commit')} dirty={env['git'].get('git_dirty')}")
    print(f"gpu:        {env['cuda'].get('gpu_name')}")
    print(f"pytorch:    {env['system'].get('pytorch_version')}")

    if not args.skip_pytest:
        _run([sys.executable, "-m", "pytest", "-q", "--tb=short"], repo_root)
        _run([sys.executable, "-m", "pytest", "tests/training/test_compile.py", "-q", "--tb=short"], repo_root)

    all_rows: list[dict] = []

    def track(row: dict, baseline_tps: float | None) -> dict:
        saved = _record(jsonl, row, baseline_tps)
        all_rows.append(saved)
        atomic_name = saved.get("experiment_name") or f"row_{len(all_rows)}"
        save_performance_summary(benches_dir / f"{atomic_name}.json", saved)
        return saved

    # 1. Uncompiled baseline re-measurement (M14 window: B=1 T=1024 G=1, 5+20)
    print("[m15] 1. Uncompiled BF16 baseline re-measurement", flush=True)
    baseline = track(
        run_performance_benchmark(
            output_dir=benches_dir / "baseline_bf16_B1",
            precision="bf16",
            compiled=False,
            batch_size=1,
            grad_accum_steps=1,
            warmup_steps=5,
            measured_steps=20,
            experiment_name="uncompiled_bf16_B1_G1",
        ),
        None,
    )
    baseline_tps = baseline.get("steady_state_tokens_per_second") if baseline.get("status") == "PASS" else None
    attach_speedup(baseline, baseline_tps)

    # 2. Compile correctness
    print("[m15] 2. Compile correctness", flush=True)
    logits = compare_compiled_logits(device="cuda")
    save_performance_summary(out / "compile_logits.json", logits)
    ckpt = verify_compiled_checkpoint_roundtrip(output_dir=benches_dir / "compile_ckpt", device="cuda", precision="bf16")
    save_performance_summary(out / "compile_checkpoint.json", ckpt)
    print(f"[m15] logits max_abs={logits.get('max_abs_diff')} ok={logits.get('ok')} ckpt_wrapper={ckpt.get('wrapper_keys_in_checkpoint')}")

    # 3-4. Compile default / reduce-overhead at B=8,16
    for mode in ("default", "reduce-overhead"):
        for batch_size in (8, 16):
            print(f"[m15] compile mode={mode} B={batch_size}", flush=True)
            track(
                run_performance_benchmark(
                    output_dir=benches_dir / f"compile_{mode}_B{batch_size}",
                    precision="bf16",
                    compiled=True,
                    compile_mode=mode,
                    batch_size=batch_size,
                    grad_accum_steps=1,
                    warmup_steps=5,
                    measured_steps=20,
                    experiment_name=f"compile_{mode}_B{batch_size}_G1",
                ),
                baseline_tps,
            )

    # 5. SDPA backend inspection
    print("[m15] 5. SDPA backend inspection", flush=True)
    present = list_probe_sdpa_backends()
    save_performance_summary(out / "sdpa_backends_present.json", {"present": present})
    print(f"[m15] SDPBackend present: {present}")

    # 6. SDPA controlled benchmark at representative B=8, plus eager reference
    print("[m15] 6. SDPA controlled benchmark B=8", flush=True)
    track(
        run_performance_benchmark(
            output_dir=benches_dir / "sdpa_auto_B8",
            precision="bf16",
            compiled=False,
            batch_size=8,
            grad_accum_steps=1,
            warmup_steps=3,
            measured_steps=10,
            experiment_name="uncompiled_bf16_B8_G1_sdpa_auto",
        ),
        baseline_tps,
    )
    for backend in present:
        print(f"[m15] forced SDPA {backend}", flush=True)
        track(
            probe_sdpa_backend(
                backend,
                output_dir=benches_dir / f"sdpa_{backend}_B8",
                batch_size=8,
                warmup_steps=3,
                measured_steps=10,
            ),
            baseline_tps,
        )
    track(
        run_performance_benchmark(
            output_dir=benches_dir / "eager_B8",
            precision="bf16",
            attention_backend="eager",
            compiled=False,
            batch_size=8,
            grad_accum_steps=1,
            warmup_steps=3,
            measured_steps=10,
            experiment_name="uncompiled_bf16_B8_G1_eager",
        ),
        baseline_tps,
    )

    # 7. Uncompiled B sweep (B=1 already measured as baseline)
    print("[m15] 7. Uncompiled BF16 B sweep", flush=True)
    for batch_size in (2, 4, 8, 16):
        name = f"uncompiled_bf16_B{batch_size}_G1"
        if any(row.get("experiment_name") == name or (row.get("B") == batch_size and not row.get("compiled") and row.get("G") == 1 and row.get("sdpa_kernel") == "auto" and row.get("attention_backend") == "sdpa" and row.get("measured_steps") == 20) for row in all_rows):
            # B=8 auto may already exist with a shorter window; still measure the full window.
            pass
        track(
            run_performance_benchmark(
                output_dir=benches_dir / f"sweep_B{batch_size}",
                precision="bf16",
                compiled=False,
                batch_size=batch_size,
                grad_accum_steps=1,
                warmup_steps=5,
                measured_steps=20,
                experiment_name=name if batch_size != 8 else "uncompiled_bf16_B8_G1_fullwindow",
            ),
            baseline_tps,
        )

    # 9. G / token-batch comparison (uncompiled)
    print("[m15] 9. Gradient accumulation / token-batch comparison", flush=True)
    for batch_size, grad_accum in ((8, 2), (8, 4), (8, 8), (16, 4), (16, 8)):
        track(
            run_performance_benchmark(
                output_dir=benches_dir / f"accum_B{batch_size}_G{grad_accum}",
                precision="bf16",
                compiled=False,
                batch_size=batch_size,
                grad_accum_steps=grad_accum,
                warmup_steps=3,
                measured_steps=8,
                experiment_name=f"uncompiled_bf16_B{batch_size}_G{grad_accum}",
            ),
            baseline_tps,
        )

    candidates = _pick_candidates(all_rows)
    stability: list[dict] = []
    if not args.skip_stability and candidates:
        print("[m15] 10. Short FineWeb-Edu stability runs", flush=True)
        from basikgpt.training.performance import empty_cuda

        empty_cuda()
        data_dir = Path(args.data_dir)
        _prepare_smoke_data(data_dir)
        for candidate in candidates:
            empty_cuda()
            try:
                stability.append(_stability_run(data_dir=data_dir, out_dir=out, candidate=candidate))
            except Exception as exc:
                stability.append(
                    {
                        "labels": candidate["labels"],
                        "config": {
                            "compiled": bool(candidate["row"].get("compiled")),
                            "compile_mode": candidate["row"].get("compile_mode"),
                            "B": candidate["row"].get("B"),
                            "G": candidate["row"].get("G"),
                        },
                        "error": str(exc),
                        "loss_finite": False,
                        "resume_ok": False,
                    }
                )
            empty_cuda()

    # Same-B compile speedups vs uncompiled full-window at that B
    uncompiled_by_b = {}
    for row in all_rows:
        if (
            row.get("status") == "PASS"
            and not row.get("compiled")
            and row.get("attention_backend") == "sdpa"
            and row.get("sdpa_kernel") == "auto"
            and row.get("G") == 1
            and row.get("measured_steps") == 20
        ):
            uncompiled_by_b[row["B"]] = row.get("steady_state_tokens_per_second")
    for row in all_rows:
        same_b = uncompiled_by_b.get(row.get("B"))
        if row.get("compiled") and same_b:
            attach_speedup(row, same_b)
            row["speedup_vs_uncompiled_same_B"] = row.get("speedup_vs_baseline")
            attach_speedup(row, baseline_tps)
            row["speedup_vs_uncompiled_same_B"] = (
                row["steady_state_tokens_per_second"] / same_b
                if row.get("status") == "PASS" and row.get("steady_state_tokens_per_second")
                else None
            )

    candidate_payloads = []
    for candidate in candidates:
        row = candidate["row"]
        candidate_payloads.append(
            {
                "labels": candidate["labels"],
                "compiled": row.get("compiled"),
                "compile_mode": row.get("compile_mode"),
                "sdpa_kernel": row.get("sdpa_kernel"),
                "attention_backend": row.get("attention_backend"),
                "B": row.get("B"),
                "G": row.get("G"),
                "tokens_per_step": row.get("tokens_per_step"),
                "tokens_per_second": row.get("steady_state_tokens_per_second"),
                "peak_allocated": row.get("peak_allocated"),
                "status": row.get("status"),
                "main_run_2_5b": _plan_2_5b(int(row["B"]), int(row.get("G") or 1)),
            }
        )

    summary = {
        "status": "complete",
        "provider": env["cuda"].get("provider") or "RunPod",
        "gpu": env["cuda"].get("gpu_name"),
        "pytorch_version": env["system"].get("pytorch_version"),
        "cuda_runtime": env["cuda"].get("cuda_runtime"),
        "git_commit": env["git"].get("git_commit"),
        "git_dirty": env["git"].get("git_dirty"),
        "note": (
            "Measured on this GPU/PyTorch build. Not an optimal or production configuration. "
            "Canonical baseline remains uncompiled BF16 + PyTorch SDPA."
        ),
        "baseline_remeasurement": baseline,
        "compile_correctness": {"logits": logits, "checkpoint": ckpt},
        "sdpa_backends_present": present,
        "benchmarks": all_rows,
        "candidates": candidate_payloads,
        "stability": stability,
        "torch_compile_default_training": False,
        "ddp_used": False,
        "fsdp_used": False,
        "main_2_5b_started": False,
    }
    save_performance_summary(out / "performance_summary.json", summary)
    print(f"[m15] wrote {out / 'performance_summary.json'}")
    print("[m15] status=complete")


if __name__ == "__main__":
    main()
