"""Milestone 16 1M/10M GPU config-pilot orchestrator.

Runs Candidate A/B as separate processes so checkpoint resume is process-level.
Does not start 2.5B main training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.accounting import calculate_compile_break_even_tokens
from basikgpt.training.config_pilot import (
    CANDIDATE_A,
    CANDIDATE_B,
    CANDIDATES,
    CANONICAL_SEED,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    EVAL_TOKENS_PER_EVAL,
    MAIN_TOKEN_BUDGET,
    MIN_LEARNING_RATE,
    PEAK_LEARNING_RATE,
    PILOT_10M_TOKENS,
    PILOT_1M_TOKENS,
    WORKSPACE_ROOT,
    assert_equal_token_batch,
    canonical_config_dict,
    checkpoint_steps_for_pilot,
    disk_gate_or_raise,
    eval_interval_for_pilot,
    plan_pilot_stage,
    save_canonical_config,
    summarize_run_metrics,
    workspace_usage_bytes,
)
from basikgpt.training.metadata import atomic_save_json, extract_dataset_provenance, load_json
from basikgpt.training.reproducibility import get_git_metadata, seed_everything

FIXED_GENERATION_PROMPT = "The history of artificial intelligence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basikGPT Milestone 16 configuration pilot")
    parser.add_argument(
        "--stage",
        choices=["preflight", "data", "init", "1m", "10m", "compare", "freeze", "all"],
        default="all",
    )
    parser.add_argument("--candidate", choices=["A", "B", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default="runs/m16_pilot")
    parser.add_argument("--data-dir", type=str, default="data/fineweb-edu-m16")
    parser.add_argument("--workspace-root", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _print_preflight(output_dir: Path, data_dir: Path) -> None:
    git = get_git_metadata()
    assert_equal_token_batch()
    plan_1m = plan_pilot_stage(CANDIDATE_A, PILOT_1M_TOKENS)
    plan_10m = plan_pilot_stage(CANDIDATE_A, PILOT_10M_TOKENS)
    used = workspace_usage_bytes()
    print("=" * 78)
    print("  Milestone 16 pre-report")
    print("=" * 78)
    print(f"  git_commit:     {git.get('git_commit')}")
    print(f"  git_dirty:      {git.get('git_dirty')}")
    print(f"  workspace_du:   {used / (1024**3):.2f} GiB")
    print(f"  output_dir:     {output_dir}")
    print(f"  data_dir:       {data_dir}")
    print()
    print("  Candidate A:    compile=false  B=8  G=8  T=1024  tokens/step=65536")
    print("  Candidate B:    compile=true   B=16 G=4  T=1024  tokens/step=65536")
    print()
    print("  Controlled:     FineWeb-Edu pinned revision, GPT-2 BPE, seed 1337,")
    print("                  shared from-scratch state_dict, peak/min LR, step scheduler,")
    print("                  identical eval tokens and cadence.")
    print()
    print(
        f"  1M:  steps={plan_1m['plan']['optimizer_steps']}  "
        f"actual={plan_1m['plan']['actual_token_budget']:,}  "
        f"overshoot={plan_1m['plan']['overshoot_tokens']:+,}  "
        f"warmup={plan_1m['warmup_steps']}  "
        f"eval_interval={eval_interval_for_pilot(plan_1m['plan']['optimizer_steps'])}  "
        f"ckpt={checkpoint_steps_for_pilot(plan_1m['plan']['optimizer_steps'])}"
    )
    print(
        f"  10M: steps={plan_10m['plan']['optimizer_steps']}  "
        f"actual={plan_10m['plan']['actual_token_budget']:,}  "
        f"overshoot={plan_10m['plan']['overshoot_tokens']:+,}  "
        f"warmup={plan_10m['warmup_steps']}  "
        f"eval_interval={eval_interval_for_pilot(plan_10m['plan']['optimizer_steps'])}  "
        f"ckpt={checkpoint_steps_for_pilot(plan_10m['plan']['optimizer_steps'])}"
    )
    print()
    print("  Scheduler: optimizer-step warmup+cosine. Same token batch ⇒ A/B comparable.")
    print("  1M warmup=2 (~12.5% of 16). 10M warmup=15 (~10% of 153).")
    print("  2.5B warmup remains recipe default 2000 (provisional, not copied from 10%).")
    print()
    print("  Resume: sequential data_sample_index (exact sample index).")
    print("  1M splits at middle via stop_at_step in a new process.")
    print("  Cost boundary: two candidates only. No 2.5B main run.")
    print("=" * 78)


def _train_cmd(
    *,
    candidate,
    output_dir: Path,
    data_dir: Path,
    init_weights: Path,
    target_tokens: int,
    device: str,
    resume: Path | None = None,
    stop_at_step: int | None = None,
    eval_at_start: bool = False,
) -> list[str]:
    stage = plan_pilot_stage(candidate, target_tokens)
    steps = stage["plan"]["optimizer_steps"]
    ckpt_steps = ",".join(str(step) for step in checkpoint_steps_for_pilot(steps))
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--run-name",
        output_dir.name,
        "--output-dir",
        str(output_dir),
        "--data-dir",
        str(data_dir),
        "--model-preset",
        "gpt2_small",
        "--context-length",
        str(candidate.context_length),
        "--attention-backend",
        candidate.attention_backend,
        "--sdpa-kernel",
        candidate.sdpa_kernel,
        "--precision",
        candidate.precision,
        "--device",
        device,
        "--batch-size",
        str(candidate.micro_batch_size),
        "--grad-accum-steps",
        str(candidate.grad_accum_steps),
        "--target-tokens",
        str(target_tokens),
        "--warmup-steps",
        str(stage["warmup_steps"]),
        "--lr",
        str(PEAK_LEARNING_RATE),
        "--min-lr",
        str(MIN_LEARNING_RATE),
        "--seed",
        str(CANONICAL_SEED),
        "--init-weights",
        str(init_weights),
        "--no-shuffle",
        "--track-data-index",
        "--no-save-step-final",
        "--eval-tokens",
        str(EVAL_TOKENS_PER_EVAL),
        "--eval-interval",
        str(eval_interval_for_pilot(steps)),
        "--checkpoint-steps",
        ckpt_steps,
        "--log-interval",
        "1",
    ]
    if candidate.compile:
        cmd.extend(["--compile", "--compile-mode", candidate.compile_mode])
    if eval_at_start:
        cmd.append("--eval-at-start")
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    if stop_at_step is not None:
        cmd.extend(["--stop-at-step", str(stop_at_step)])
    return cmd


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n[m16] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def prepare_data(data_dir: Path) -> dict[str, Any]:
    if (data_dir / "manifest.json").exists():
        print(f"[m16] Reusing existing dataset at {data_dir}")
        return load_json(data_dir / "manifest.json")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_fineweb_edu.py"),
        "--output",
        str(data_dir),
        "--dataset-repo",
        DATASET_REPO,
        "--dataset-config",
        DATASET_CONFIG,
        "--dataset-revision",
        DATASET_REVISION,
        "--max-train-tokens",
        "12000000",
        "--max-validation-tokens",
        "500000",
        "--shard-token-target",
        "1000000",
    ]
    _run(cmd, repo_root)
    return load_json(data_dir / "manifest.json")


def write_shared_init(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(CANONICAL_SEED)
    model = GPT(
        GPTConfig.gpt2_small(
            context_length=1024,
            attention_backend="sdpa",
            dropout=0.0,
        )
    )
    torch.save(model.state_dict(), path)
    print(f"[m16] Wrote shared init weights to {path} ({path.stat().st_size / (1024**2):.1f} MiB)")


def run_1m(candidate, output_dir: Path, data_dir: Path, init_weights: Path, device: str) -> dict[str, Any]:
    run_dir = output_dir / f"candidate_{candidate.name.lower()}_1m"
    stage = plan_pilot_stage(candidate, PILOT_1M_TOKENS)
    middle = checkpoint_steps_for_pilot(stage["plan"]["optimizer_steps"])[0]
    disk_gate_or_raise()
    _run(
        _train_cmd(
            candidate=candidate,
            output_dir=run_dir,
            data_dir=data_dir,
            init_weights=init_weights,
            target_tokens=PILOT_1M_TOKENS,
            device=device,
            stop_at_step=middle,
            eval_at_start=True,
        ),
        repo_root,
    )
    middle_ckpt = run_dir / f"step-{middle:08d}.pt"
    if not middle_ckpt.exists():
        raise FileNotFoundError(f"1M middle checkpoint missing: {middle_ckpt}")
    _run(
        _train_cmd(
            candidate=candidate,
            output_dir=run_dir,
            data_dir=data_dir,
            init_weights=init_weights,
            target_tokens=PILOT_1M_TOKENS,
            device=device,
            resume=middle_ckpt,
            eval_at_start=False,
        ),
        repo_root,
    )
    result = summarize_run_metrics(run_dir)
    result["resume_verified"] = (
        result.get("status") == "completed"
        and result.get("optimizer_steps") == stage["plan"]["optimizer_steps"]
        and result.get("actual_tokens") == stage["plan"]["actual_token_budget"]
        and (run_dir / f"step-{stage['plan']['optimizer_steps']:08d}.pt").exists()
    )
    result["process_level_resume"] = True
    return result


def _run_passed(summary: dict[str, Any]) -> tuple[bool, str]:
    if not summary.get("finite_train_loss"):
        return False, "non-finite train loss"
    if not summary.get("finite_grad_norm"):
        return False, "non-finite gradient norm"
    if not summary.get("finite_val_loss"):
        return False, "non-finite validation loss"
    if summary.get("status") not in {"completed", "paused"}:
        return False, f"status={summary.get('status')}"
    recompile = summary.get("compile_recompile_info") or {}
    if recompile.get("possible_repeated_recompile"):
        return False, f"possible repeated recompile at steps {recompile.get('later_step_time_spikes')}"
    if summary.get("resume_verified") is False:
        return False, "resume verification failed"
    return True, "ok"


def run_10m(candidate, output_dir: Path, data_dir: Path, init_weights: Path, device: str) -> dict[str, Any]:
    run_dir = output_dir / f"candidate_{candidate.name.lower()}_10m"
    stage = plan_pilot_stage(candidate, PILOT_10M_TOKENS)
    disk_gate_or_raise()
    _run(
        _train_cmd(
            candidate=candidate,
            output_dir=run_dir,
            data_dir=data_dir,
            init_weights=init_weights,
            target_tokens=PILOT_10M_TOKENS,
            device=device,
            eval_at_start=True,
        ),
        repo_root,
    )
    ckpt_steps = checkpoint_steps_for_pilot(stage["plan"]["optimizer_steps"])
    mid = ckpt_steps[1] if len(ckpt_steps) >= 2 else ckpt_steps[0]
    mid_ckpt = run_dir / f"step-{mid:08d}.pt"
    probe_dir = output_dir / f"candidate_{candidate.name.lower()}_10m_resume_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    _run(
        _train_cmd(
            candidate=candidate,
            output_dir=probe_dir,
            data_dir=data_dir,
            init_weights=init_weights,
            target_tokens=PILOT_10M_TOKENS,
            device=device,
            resume=mid_ckpt,
            stop_at_step=mid + 1,
        ),
        repo_root,
    )
    probe_summary = load_json(probe_dir / "summary.json") if (probe_dir / "summary.json").exists() else {}
    resume_ok = int(probe_summary.get("final_step", -1)) == mid + 1
    for pt_file in probe_dir.glob("*.pt"):
        pt_file.unlink()
    atomic_save_json(
        output_dir / f"candidate_{candidate.name.lower()}_10m_resume_probe.json",
        {
            "resume_ok": resume_ok,
            "resumed_from_step": mid,
            "probe_final_step": probe_summary.get("final_step"),
            "status": probe_summary.get("status"),
        },
    )
    result = summarize_run_metrics(run_dir)
    result["resume_verified"] = resume_ok
    result["process_level_resume"] = True
    result["expected_tokens"] = stage["plan"]["actual_token_budget"]
    result["token_accounting_ok"] = result.get("actual_tokens") == stage["plan"]["actual_token_budget"]
    if result["token_accounting_ok"] is False:
        result["resume_verified"] = False
    return result


def maybe_generate(run_dir: Path, output_dir: Path, candidate_name: str) -> None:
    ckpts = sorted(run_dir.glob("step-*.pt"))
    if not ckpts:
        return
    out = output_dir / f"candidate_{candidate_name.lower()}_10m_generation.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate.py"),
        "--checkpoint",
        str(ckpts[-1]),
        "--prompt",
        FIXED_GENERATION_PROMPT,
        "--max-new-tokens",
        "40",
        "--device",
        "cuda",
        "--output-json",
        str(out),
    ]
    try:
        _run(cmd, repo_root)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        atomic_save_json(out, {"error": str(exc), "prompt": FIXED_GENERATION_PROMPT})


def write_comparison(output_dir: Path, results: dict[str, Any]) -> Path:
    a_10 = results.get("candidate_a_10m") or {}
    b_10 = results.get("candidate_b_10m") or {}
    r0 = a_10.get("training_only_tokens_per_sec")
    r1 = b_10.get("training_only_tokens_per_sec")
    compile_s = b_10.get("cold_compile_seconds")
    break_even = None
    if r0 and r1 and compile_s is not None:
        break_even = calculate_compile_break_even_tokens(float(compile_s), float(r0), float(r1))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_metadata(),
        "candidates": {"A": CANDIDATE_A.to_dict(), "B": CANDIDATE_B.to_dict()},
        "results": results,
        "compile_break_even_tokens_10m": break_even,
        "disk_bytes": workspace_usage_bytes(),
    }
    path = output_dir / "comparison.json"
    atomic_save_json(path, payload)
    return path


def select_canonical(results: dict[str, Any]) -> tuple[Any, str]:
    """Pick a provisional canonical config. Throughput alone does not win."""
    a1 = results.get("candidate_a_1m") or {}
    b1 = results.get("candidate_b_1m") or {}
    a10 = results.get("candidate_a_10m") or {}
    b10 = results.get("candidate_b_10m") or {}
    a_ok = bool(results.get("candidate_a_10m_passed"))
    b_ok = bool(results.get("candidate_b_10m_passed"))
    reasons: list[str] = []
    if a_ok and not b_ok:
        reasons.append("Candidate B failed the 10M gate; A remained finite and resumable.")
        return CANDIDATE_A, " ".join(reasons)
    if b_ok and not a_ok:
        reasons.append("Candidate A failed the 10M gate; B remained finite and resumable.")
        return CANDIDATE_B, " ".join(reasons)
    if not a_ok and not b_ok:
        reasons.append("Neither 10M run passed; freezing A as the conservative fallback is withheld.")
        return CANDIDATE_A, "INCOMPLETE: " + " ".join(reasons)

    a_alloc = a10.get("peak_allocated_vram_bytes") or 0
    b_alloc = b10.get("peak_allocated_vram_bytes") or 0
    a_toks = a10.get("training_only_tokens_per_sec") or 0
    b_toks = b10.get("training_only_tokens_per_sec") or 0
    b_recompile = (b10.get("compile_recompile_info") or {}).get("possible_repeated_recompile")
    reasons.append(
        f"Both 10M runs were finite with process-level resume. "
        f"A training-only {a_toks:,.0f} tok/s allocated {a_alloc / (1024**3):.2f} GiB; "
        f"B {b_toks:,.0f} tok/s allocated {b_alloc / (1024**3):.2f} GiB."
    )
    if b_recompile:
        reasons.append("B showed possible repeated recompilation; preferring A operational simplicity.")
        return CANDIDATE_A, " ".join(reasons)
    # Prefer VRAM headroom and no compile dependency unless B is materially faster
    # without eating headroom. ~10% is not enough to pay compile complexity.
    if b_toks > a_toks * 1.15 and b_alloc < 20 * (1024**3):
        reasons.append(
            "B was more than 15% faster at acceptable VRAM; selected as canonical with compile complexity noted."
        )
        return CANDIDATE_B, " ".join(reasons)
    reasons.append(
        "Canonical prefers correctness, VRAM headroom, and operational simplicity "
        "(uncompiled B=8 G=8) over a modest compile speedup."
    )
    return CANDIDATE_A, " ".join(reasons)


def freeze_canonical(output_dir: Path, results: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    candidate, reason = select_canonical(results)
    provenance = extract_dataset_provenance(manifest)
    extra = {
        "selection_reason": reason,
        "results_available": sorted(results.keys()),
    }
    payload = canonical_config_dict(
        candidate,
        dataset_repository=provenance.get("repository") or DATASET_REPO,
        dataset_config=provenance.get("config") or DATASET_CONFIG,
        dataset_revision=provenance.get("revision") or DATASET_REVISION,
        extra=extra,
    )
    chosen_10m = results.get(f"candidate_{candidate.name.lower()}_10m") or {}
    tok_s = chosen_10m.get("training_only_tokens_per_sec")
    e2e = chosen_10m.get("end_to_end_tokens_per_sec")
    plan = candidate.plan(MAIN_TOKEN_BUDGET)
    payload["runtime_estimate"] = {
        "training_only_tokens_per_sec": tok_s,
        "end_to_end_tokens_per_sec": e2e,
        "rough_training_only_seconds": (MAIN_TOKEN_BUDGET / tok_s) if tok_s else None,
        "rough_end_to_end_seconds": (MAIN_TOKEN_BUDGET / e2e) if e2e else None,
        "note": "Rough estimate from 10M sustained throughput. Not an exact ETA.",
    }
    repo_cfg = repo_root / "configs" / "gpt2_small_fineweb_edu_single_gpu.json"
    repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    save_canonical_config(repo_cfg, payload)
    save_canonical_config(output_dir / "canonical_config.json", payload)
    return {"candidate": candidate.to_dict(), "reason": reason, "main_plan": plan.to_dict(), "path": str(repo_cfg)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["A", "B"] if args.candidate == "both" else [args.candidate]

    if args.stage in {"preflight", "all"}:
        _print_preflight(output_dir, data_dir)
        disk_gate_or_raise(args.workspace_root)

    manifest = None
    if args.stage in {"data", "all"} and not args.skip_data_prep:
        disk_gate_or_raise(args.workspace_root)
        manifest = prepare_data(data_dir)
        cache_bytes = workspace_usage_bytes(Path("/workspace/.cache")) if Path("/workspace/.cache").exists() else 0
        print(f"[m16] dataset ready. HF cache ≈ {cache_bytes / (1024**3):.2f} GiB")
        if cache_bytes > 8 * 1024**3:
            raise RuntimeError("HuggingFace cache exceeded 8 GiB; refusing to continue.")
    elif (data_dir / "manifest.json").exists():
        manifest = load_json(data_dir / "manifest.json")

    init_path = output_dir / "shared_init.pt"
    if args.stage in {"init", "all"}:
        write_shared_init(init_path)
    if not init_path.exists() and args.stage in {"1m", "10m", "all"}:
        write_shared_init(init_path)

    results: dict[str, Any] = {}
    comparison_path = output_dir / "comparison.json"
    if comparison_path.exists() and args.stage != "all":
        results = load_json(comparison_path).get("results") or {}

    if args.stage in {"1m", "all"}:
        for name in names:
            candidate = CANDIDATES[name]
            key = f"candidate_{name.lower()}_1m"
            print(f"\n[m16] === 1M Candidate {name} ===")
            summary = run_1m(candidate, output_dir, data_dir, init_path, args.device)
            ok, reason = _run_passed(summary)
            summary["passed"] = ok
            summary["gate_reason"] = reason
            results[key] = summary
            results[f"{key}_passed"] = ok
            write_comparison(output_dir, results)
            print(f"[m16] Candidate {name} 1M gate: {ok} ({reason})")
            disk_gate_or_raise(args.workspace_root)

    if args.stage in {"10m", "all"}:
        for name in names:
            if not results.get(f"candidate_{name.lower()}_1m_passed"):
                print(f"[m16] Skipping 10M Candidate {name}: 1M did not pass")
                results[f"candidate_{name.lower()}_10m_passed"] = False
                results[f"candidate_{name.lower()}_10m_skip_reason"] = "1M gate failed"
                continue
            candidate = CANDIDATES[name]
            print(f"\n[m16] === 10M Candidate {name} ===")
            summary = run_10m(candidate, output_dir, data_dir, init_path, args.device)
            ok, reason = _run_passed(summary)
            if summary.get("token_accounting_ok") is False:
                ok, reason = False, "incorrect token accounting"
            summary["passed"] = ok
            summary["gate_reason"] = reason
            results[f"candidate_{name.lower()}_10m"] = summary
            results[f"candidate_{name.lower()}_10m_passed"] = ok
            write_comparison(output_dir, results)
            if ok:
                maybe_generate(output_dir / f"candidate_{name.lower()}_10m", output_dir, name)
            print(f"[m16] Candidate {name} 10M gate: {ok} ({reason})")
            disk_gate_or_raise(args.workspace_root)

    if args.stage in {"compare", "freeze", "all"}:
        write_comparison(output_dir, results)

    if args.stage in {"freeze", "all"}:
        freeze = freeze_canonical(output_dir, results, manifest)
        atomic_save_json(output_dir / "canonical_selection.json", freeze)
        print(f"[m16] Canonical freeze: Candidate {freeze['candidate']['name']}")
        print(f"[m16] Reason: {freeze['reason']}")
        print(f"[m16] Wrote {freeze['path']}")


if __name__ == "__main__":
    main()
