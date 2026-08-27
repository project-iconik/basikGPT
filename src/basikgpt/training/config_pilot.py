"""Milestone 16 candidate specs, token planning, and canonical config freeze helpers.

Candidate A/B share the same global token batch so execution strategy can be
compared without changing optimizer-step count or the LR schedule horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from basikgpt.training.accounting import (
    TokenBudgetPlan,
    calculate_eval_batches,
    calculate_training_steps,
    calculate_warmup_steps,
)
from basikgpt.training.metadata import atomic_save_json, load_json

COMPARISON_TOKEN_BATCH = 65_536
PILOT_1M_TOKENS = 1_000_000
PILOT_10M_TOKENS = 10_000_000
MAIN_TOKEN_BUDGET = 2_500_000_000
EVAL_TOKENS_PER_EVAL = 131_072
WARMUP_FRACTION = 0.10
CANONICAL_SEED = 1337
CONTEXT_LENGTH = 1024
WORLD_SIZE = 1
PEAK_LEARNING_RATE = 6e-4
MIN_LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
MAIN_WARMUP_STEPS_PROVISIONAL = 2000
DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER_ENCODING = "gpt2"
DISK_GATE_BYTES = 55 * 1024**3
WORKSPACE_ROOT = Path("/workspace")


@dataclass(frozen=True, slots=True)
class ExecutionCandidate:
    """Single-GPU execution candidate with a fixed token batch."""

    name: str
    label: str
    precision: str
    attention_backend: str
    sdpa_kernel: str
    compile: bool
    compile_mode: str
    micro_batch_size: int
    context_length: int
    grad_accum_steps: int
    world_size: int

    @property
    def tokens_per_optimizer_step(self) -> int:
        return (
            self.micro_batch_size
            * self.context_length
            * self.grad_accum_steps
            * self.world_size
        )

    def plan(self, target_tokens: int) -> TokenBudgetPlan:
        return calculate_training_steps(
            target_tokens=target_tokens,
            micro_batch_size=self.micro_batch_size,
            context_length=self.context_length,
            grad_accum_steps=self.grad_accum_steps,
            world_size=self.world_size,
        )

    def eval_batches(self, eval_tokens: int = EVAL_TOKENS_PER_EVAL) -> int:
        return calculate_eval_batches(
            eval_tokens=eval_tokens,
            micro_batch_size=self.micro_batch_size,
            context_length=self.context_length,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tokens_per_optimizer_step"] = self.tokens_per_optimizer_step
        return payload


CANDIDATE_A = ExecutionCandidate(
    name="A",
    label="conservative_uncompiled",
    precision="bf16",
    attention_backend="sdpa",
    sdpa_kernel="auto",
    compile=False,
    compile_mode="default",
    micro_batch_size=8,
    context_length=CONTEXT_LENGTH,
    grad_accum_steps=8,
    world_size=WORLD_SIZE,
)

CANDIDATE_B = ExecutionCandidate(
    name="B",
    label="compiled_default",
    precision="bf16",
    attention_backend="sdpa",
    sdpa_kernel="auto",
    compile=True,
    compile_mode="default",
    micro_batch_size=16,
    context_length=CONTEXT_LENGTH,
    grad_accum_steps=4,
    world_size=WORLD_SIZE,
)

CANDIDATES: dict[str, ExecutionCandidate] = {
    "A": CANDIDATE_A,
    "B": CANDIDATE_B,
    "a": CANDIDATE_A,
    "b": CANDIDATE_B,
}


def assert_equal_token_batch(
    left: ExecutionCandidate = CANDIDATE_A,
    right: ExecutionCandidate = CANDIDATE_B,
    expected: int = COMPARISON_TOKEN_BATCH,
) -> None:
    """Fails if the comparison candidates do not share `expected` tokens/step."""
    if left.tokens_per_optimizer_step != right.tokens_per_optimizer_step:
        raise ValueError(
            f"Candidate token batches differ: {left.name}="
            f"{left.tokens_per_optimizer_step} vs {right.name}="
            f"{right.tokens_per_optimizer_step}"
        )
    if left.tokens_per_optimizer_step != expected:
        raise ValueError(
            f"Expected tokens/step {expected}, got {left.tokens_per_optimizer_step}"
        )


def plan_pilot_stage(
    candidate: ExecutionCandidate,
    target_tokens: int,
    warmup_fraction: float = WARMUP_FRACTION,
    eval_tokens: int = EVAL_TOKENS_PER_EVAL,
) -> dict[str, Any]:
    """Analytical 1M/10M/2.5B plan for one candidate. Does not hard-code step counts."""
    plan = candidate.plan(target_tokens)
    warmup = calculate_warmup_steps(plan.optimizer_steps, fraction=warmup_fraction)
    if warmup > plan.optimizer_steps:
        raise ValueError(
            f"warmup_steps ({warmup}) exceeds optimizer_steps ({plan.optimizer_steps})"
        )
    return {
        "candidate": candidate.to_dict(),
        "target_tokens": target_tokens,
        "plan": plan.to_dict(),
        "warmup_steps": warmup,
        "warmup_fraction": warmup_fraction,
        "eval_tokens": eval_tokens,
        "eval_batches": candidate.eval_batches(eval_tokens),
        "scheduler": "optimizer_step_linear_warmup_cosine_decay",
        "seed": CANONICAL_SEED,
        "peak_learning_rate": PEAK_LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE,
    }


def checkpoint_steps_for_pilot(optimizer_steps: int) -> list[int]:
    """Sparse checkpoint cadence: 1M → middle+final; longer runs → ~25/50/75/100%."""
    if optimizer_steps <= 0:
        raise ValueError(f"optimizer_steps must be positive, got {optimizer_steps}")
    if optimizer_steps <= 16:
        middle = max(1, optimizer_steps // 2)
        steps = [middle]
        if optimizer_steps != middle:
            steps.append(optimizer_steps)
        return steps
    fractions = (0.25, 0.50, 0.75, 1.00)
    steps: list[int] = []
    for frac in fractions:
        step = max(1, round(frac * optimizer_steps))
        if step not in steps:
            steps.append(step)
    if steps[-1] != optimizer_steps:
        steps.append(optimizer_steps)
    return steps


def eval_interval_for_pilot(optimizer_steps: int) -> int:
    """1M: middle/end. 10M: ~four interior points plus final (Trainer also evals max_steps)."""
    if optimizer_steps <= 16:
        return max(1, optimizer_steps // 2)
    return max(1, round(optimizer_steps / 4))


def canonical_config_dict(
    candidate: ExecutionCandidate,
    *,
    dataset_repository: str = DATASET_REPO,
    dataset_config: str = DATASET_CONFIG,
    dataset_revision: str = DATASET_REVISION,
    tokenizer_encoding: str = TOKENIZER_ENCODING,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provisional single-GPU freeze payload. Not claimed to be optimal."""
    plan = candidate.plan(MAIN_TOKEN_BUDGET)
    payload = {
        "schema": "basikgpt.canonical_single_gpu.v1",
        "status": "provisional_freeze",
        "model_preset": "gpt2_small",
        "precision": candidate.precision,
        "attention_backend": candidate.attention_backend,
        "sdpa_kernel": candidate.sdpa_kernel,
        "compile": candidate.compile,
        "compile_mode": candidate.compile_mode if candidate.compile else None,
        "micro_batch_size": candidate.micro_batch_size,
        "context_length": candidate.context_length,
        "grad_accum_steps": candidate.grad_accum_steps,
        "world_size": candidate.world_size,
        "tokens_per_optimizer_step": candidate.tokens_per_optimizer_step,
        "peak_learning_rate": PEAK_LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE,
        "warmup_steps_main_provisional": MAIN_WARMUP_STEPS_PROVISIONAL,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": CANONICAL_SEED,
        "dropout": 0.0,
        "dataset_repository": dataset_repository,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "tokenizer_encoding": tokenizer_encoding,
        "main_token_budget": MAIN_TOKEN_BUDGET,
        "main_plan": plan.to_dict(),
        "notes": (
            "Provisional single-GPU baseline from RTX PRO 4500 Blackwell 1M/10M "
            "FineWeb-Edu pilots. Not an optimal hyperparameter claim."
        ),
    }
    if extra:
        payload["extra"] = extra
    return payload


def save_canonical_config(path: Path | str, payload: dict[str, Any]) -> Path:
    return atomic_save_json(path, payload)


def load_canonical_config(path: Path | str) -> dict[str, Any]:
    payload = load_json(path)
    required = (
        "precision",
        "attention_backend",
        "compile",
        "micro_batch_size",
        "context_length",
        "grad_accum_steps",
        "tokens_per_optimizer_step",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"canonical config missing fields: {', '.join(missing)}")
    return payload


def workspace_usage_bytes(root: Path | str = WORKSPACE_ROOT) -> int:
    """Directory size via `du -sb`. Falls back to a file walk if du is unavailable."""
    root_path = Path(root)
    if not root_path.exists():
        return 0
    result = subprocess.run(
        ["du", "-sb", str(root_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return int(result.stdout.split()[0])
    total = 0
    for path in root_path.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def load_metrics_records(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def summarize_run_metrics(run_dir: Path | str) -> dict[str, Any]:
    """Extracts comparison-table fields from a completed (or paused) run directory."""
    run_dir = Path(run_dir)
    records = load_metrics_records(run_dir)
    train_rows = [row for row in records if row.get("type") == "train"]
    val_rows = [row for row in records if row.get("type") == "val"]
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    train_cfg = load_json(run_dir / "training_config.json") if (run_dir / "training_config.json").exists() else {}
    run_meta = load_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}

    losses = [float(row["loss"]) for row in train_rows if row.get("loss") is not None]
    grad_norms = [float(row["grad_norm"]) for row in train_rows if row.get("grad_norm") is not None]
    finite_losses = all(math.isfinite(value) for value in losses)
    finite_grads = all(math.isfinite(value) for value in grad_norms)
    finite_vals = all(math.isfinite(float(row["val_loss"])) for row in val_rows if row.get("val_loss") is not None)

    def _percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    last_train = train_rows[-1] if train_rows else {}
    return {
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "compile": bool(train_cfg.get("compile", False)),
        "compile_mode": train_cfg.get("compile_mode"),
        "batch_size": train_cfg.get("batch_size"),
        "grad_accum_steps": train_cfg.get("gradient_accumulation_steps"),
        "tokens_per_step": (
            int(train_cfg["batch_size"])
            * int(load_json(run_dir / "model_config.json")["context_length"])
            * int(train_cfg["gradient_accumulation_steps"])
            if (run_dir / "model_config.json").exists() and train_cfg
            else None
        ),
        "optimizer_steps": summary.get("final_step"),
        "actual_tokens": summary.get("tokens_seen"),
        "train_loss_initial": losses[0] if losses else None,
        "train_loss_final": losses[-1] if losses else None,
        "val_loss_final": summary.get("final_val_loss") or (val_rows[-1]["val_loss"] if val_rows else None),
        "val_curve": [{"step": row["step"], "tokens_seen": row.get("tokens_seen"), "val_loss": row.get("val_loss")} for row in val_rows],
        "grad_norm_min": min(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "grad_norm_median": _percentile(grad_norms, 0.5),
        "training_only_tokens_per_sec": summary.get("training_only_tokens_per_sec") or last_train.get("training_only_tokens_per_sec"),
        "end_to_end_tokens_per_sec": summary.get("end_to_end_tokens_per_sec") or last_train.get("end_to_end_tokens_per_sec"),
        "peak_allocated_vram_bytes": last_train.get("peak_allocated_vram_bytes"),
        "peak_reserved_vram_bytes": last_train.get("peak_reserved_vram_bytes"),
        "cold_compile_seconds": summary.get("cold_compile_seconds"),
        "time_to_first_optimizer_step": summary.get("time_to_first_optimizer_step"),
        "compile_recompile_info": summary.get("compile_recompile_info"),
        "resume_class": summary.get("resume_class"),
        "finite_train_loss": finite_losses,
        "finite_grad_norm": finite_grads,
        "finite_val_loss": finite_vals,
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "git": (run_meta.get("git") if isinstance(run_meta.get("git"), dict) else {}),
        "loss_curve": [
            {
                "step": row["step"],
                "tokens_seen": row.get("tokens_seen"),
                "train_loss": row.get("loss"),
                "learning_rate": row.get("learning_rate"),
                "grad_norm": row.get("grad_norm"),
            }
            for row in train_rows
        ],
    }


def disk_gate_or_raise(root: Path | str = WORKSPACE_ROOT, limit_bytes: int = DISK_GATE_BYTES) -> int:
    used = workspace_usage_bytes(root)
    if used > limit_bytes:
        raise RuntimeError(
            f"Disk gate: {root} uses {used / (1024**3):.1f} GiB, limit "
            f"{limit_bytes / (1024**3):.1f} GiB. Refusing to continue."
        )
    return used
