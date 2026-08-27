"""Local Pilot Pretraining Protocol definitions, execution presets, and structured summary reporting."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

from basikgpt.training.accounting import TokenBudgetPlan, calculate_training_steps

PilotStatus = Literal["passed", "failed", "not_executed"]


@dataclass(frozen=True, slots=True)
class PilotStageSpec:
    """Specification for a reproducible pilot pretraining stage."""

    stage_name: str
    description: str
    target_tokens: int
    model_preset: str
    context_length: int
    batch_size: int
    grad_accum_steps: int
    warmup_steps: int
    learning_rate: float
    min_learning_rate: float
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    log_interval: int

    def compute_plan(self, world_size: int = 1) -> TokenBudgetPlan:
        """Computes the exact TokenBudgetPlan for this pilot stage."""
        return calculate_training_steps(
            target_tokens=self.target_tokens,
            micro_batch_size=self.batch_size,
            context_length=self.context_length,
            grad_accum_steps=self.grad_accum_steps,
            world_size=world_size,
        )


# Canonical Pilot Stage Presets
PILOT_STAGES: dict[str, PilotStageSpec] = {
    "stage_a": PilotStageSpec(
        stage_name="stage_a",
        description="Stage A: Smoke Pilot (~10K tokens) - Verifies pipeline mechanics, finite loss, finite gradients, LR movement, checkpointing, and validation.",
        target_tokens=10_000,
        model_preset="tiny",
        context_length=64,
        batch_size=2,
        grad_accum_steps=2,
        warmup_steps=5,
        learning_rate=6e-4,
        min_learning_rate=6e-5,
        eval_interval=10,
        eval_batches=5,
        checkpoint_interval=20,
        log_interval=5,
    ),
    "stage_b": PilotStageSpec(
        stage_name="stage_b",
        description="Stage B: Short Pilot (~100K tokens) - Observes an overall downward loss trend (not per-step monotone), gradient norm stability, validation cadence, and state-continuous resume.",
        target_tokens=100_000,
        model_preset="tiny",
        context_length=128,
        batch_size=4,
        grad_accum_steps=2,
        warmup_steps=10,
        learning_rate=6e-4,
        min_learning_rate=6e-5,
        eval_interval=25,
        eval_batches=10,
        checkpoint_interval=50,
        log_interval=10,
    ),
    "stage_c": PilotStageSpec(
        stage_name="stage_c",
        description="Stage C: Extended Local Pilot (~1M tokens) - Validates configuration, schedule curve, and execution path for multi-thousand step regimens.",
        target_tokens=1_000_000,
        model_preset="tiny",
        context_length=256,
        batch_size=4,
        grad_accum_steps=4,
        warmup_steps=25,
        learning_rate=6e-4,
        min_learning_rate=6e-5,
        eval_interval=50,
        eval_batches=10,
        checkpoint_interval=100,
        log_interval=25,
    ),
}


@dataclass
class PilotSummary:
    """Structured summary record of a pilot pretraining execution."""

    pilot_stage: str
    status: PilotStatus
    requested_tokens: int
    actual_tokens: int
    tokens_per_step: int
    optimizer_steps: int
    initial_train_loss: float | None = None
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    best_validation_loss: float | None = None
    min_gradient_norm: float | None = None
    max_gradient_norm: float | None = None
    initial_learning_rate: float | None = None
    final_learning_rate: float | None = None
    elapsed_seconds: float = 0.0
    tokens_per_sec: float = 0.0
    checkpoint_path: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes summary to JSON-compatible dictionary."""
        return asdict(self)

    def save_json(self, output_path: Path | str) -> Path:
        """Saves structured summary to a JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    def format_human_readable(self) -> str:
        """Formats summary as a human-readable table string."""
        status_str = self.status.upper()
        lines = [
            "=" * 70,
            f"  basikGPT Pilot Protocol Summary [{self.pilot_stage.upper()}] - {status_str}",
            "=" * 70,
            f"  Status:                 {status_str}",
            f"  Requested Tokens:       {self.requested_tokens:,}",
            f"  Actual Tokens Processed:{self.actual_tokens:,} ({self.actual_tokens - self.requested_tokens:+,} overshoot)",
            f"  Tokens / Optimizer Step:{self.tokens_per_step:,}",
            f"  Total Optimizer Steps:  {self.optimizer_steps:,}",
        ]
        if self.initial_train_loss is not None:
            lines.append(f"  Initial Train Loss:     {self.initial_train_loss:.4f}")
        if self.final_train_loss is not None:
            lines.append(f"  Final Train Loss:       {self.final_train_loss:.4f}")
        if self.final_validation_loss is not None:
            lines.append(f"  Final Validation Loss:  {self.final_validation_loss:.4f}")
        if self.best_validation_loss is not None:
            lines.append(f"  Best Validation Loss:   {self.best_validation_loss:.4f}")
        if self.min_gradient_norm is not None and self.max_gradient_norm is not None:
            lines.append(f"  Gradient Norm (Min/Max):{self.min_gradient_norm:.3f} / {self.max_gradient_norm:.3f}")
        if self.initial_learning_rate is not None and self.final_learning_rate is not None:
            lines.append(f"  Learning Rate (Init/End):{self.initial_learning_rate:.2e} / {self.final_learning_rate:.2e}")
        lines.append(f"  Elapsed Time:           {self.elapsed_seconds:.2f}s")
        lines.append(f"  Throughput:             {self.tokens_per_sec:,.0f} tokens/sec")
        if self.checkpoint_path:
            lines.append(f"  Final Checkpoint:       {self.checkpoint_path}")
        if self.failure_reason:
            lines.append(f"  Failure Reason:         {self.failure_reason}")
        lines.append("=" * 70)
        return "\n".join(lines)
