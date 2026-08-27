"""CLI entrypoint for executing and validating reproducible Pilot Pretraining Stages in basikGPT."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Literal
import torch
from torch.utils.data import DataLoader

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from basikgpt.config import GPTConfig
from basikgpt.data.shard import ShardedTokenDataset
from basikgpt.model.gpt import GPT
from basikgpt.training.accounting import calculate_training_steps
from basikgpt.training.config import TrainingConfig
from basikgpt.training.metadata import load_json
from basikgpt.training.pilot import PILOT_STAGES, PilotStageSpec, PilotSummary
from basikgpt.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="basikGPT Pilot Pretraining Protocol Runner (Stage A: 10K, Stage B: 100K, Stage C: 1M)."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="stage_a",
        choices=["stage_a", "stage_b", "stage_c"],
        help="Pilot stage preset to execute (default: stage_a)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fineweb-edu-smoke",
        help="Directory containing tokenized .npy binary shards and manifest.json (default: data/fineweb-edu-smoke)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for run artifacts and checkpoints",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Custom path for structured pilot summary JSON (default: <output-dir>/pilot_summary.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cpu', 'cuda', 'auto') (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic initialization and data batching (default: 1337)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and validate configuration without executing training",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    return parser.parse_args()


def run_pilot(args: argparse.Namespace) -> PilotSummary:
    stage_spec: PilotStageSpec = PILOT_STAGES[args.stage]
    plan = stage_spec.compute_plan(world_size=1)

    data_path = Path(args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"pilot-{args.stage}-{timestamp}"
    json_path = Path(args.output_json) if args.output_json else out_dir / "pilot_summary.json"

    print(f"\n[Pilot] Initializing Pilot Protocol: {stage_spec.description}")
    print(f"  Target Budget:      {plan.requested_token_budget:,} requested tokens")
    print(f"  Batching Config:    B={plan.micro_batch_size}, T={plan.context_length}, G={plan.grad_accum_steps}, W={plan.world_size}")
    print(f"  Tokens / Step:      {plan.tokens_per_optimizer_step:,}")
    print(f"  Planned Steps:      {plan.optimizer_steps:,}")
    print(f"  Actual Tokens:      {plan.actual_token_budget:,} ({plan.overshoot_tokens:+,} overshoot)")

    if args.dry_run:
        summary = PilotSummary(
            pilot_stage=args.stage,
            status="not_executed",
            requested_tokens=plan.requested_token_budget,
            actual_tokens=plan.actual_token_budget,
            tokens_per_step=plan.tokens_per_optimizer_step,
            optimizer_steps=plan.optimizer_steps,
            failure_reason="Dry-run requested by user (execution skipped)",
        )
        summary.save_json(json_path)
        print("\n" + summary.format_human_readable())
        return summary

    # Configure Model
    model_cfg = GPTConfig(
        vocab_size=50257,
        context_length=stage_spec.context_length,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        attention_backend="sdpa",
        dropout=0.0,
    )

    train_shards = sorted(data_path.glob("train-*.npy"))
    val_shards = sorted(data_path.glob("validation-*.npy"))

    if not train_shards:
        raise FileNotFoundError(f"No train shards found in {data_path}")

    train_dataset = ShardedTokenDataset(train_shards, context_length=model_cfg.context_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=stage_spec.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = None
    if val_shards:
        val_dataset = ShardedTokenDataset(val_shards, context_length=model_cfg.context_length)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=stage_spec.batch_size, shuffle=False, drop_last=False)

    model = GPT(model_cfg)
    train_cfg = TrainingConfig(
        learning_rate=stage_spec.learning_rate,
        min_learning_rate=stage_spec.min_learning_rate,
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup_steps=stage_spec.warmup_steps,
        max_steps=plan.optimizer_steps,
        batch_size=stage_spec.batch_size,
        gradient_accumulation_steps=stage_spec.grad_accum_steps,
        eval_interval=stage_spec.eval_interval,
        eval_batches=stage_spec.eval_batches,
        checkpoint_interval=stage_spec.checkpoint_interval,
        log_interval=stage_spec.log_interval,
        device=args.device,
        precision="fp32",
        output_dir=str(out_dir),
        seed=args.seed,
    )

    manifest_path = data_path / "manifest.json"
    dataset_manifest = load_json(manifest_path) if manifest_path.exists() else None

    trainer = Trainer(
        model=model,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=f"pilot-{args.stage}",
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=manifest_path if manifest_path.exists() else None,
        overwrite=args.overwrite,
    )

    start_time = time.perf_counter()
    status: Literal["passed", "failed", "not_executed"] = "passed"
    failure_reason = None
    history = []

    try:
        history = trainer.train()
    except Exception as exc:
        status = "failed"
        failure_reason = str(exc)
        print(f"\n[Pilot] Execution failed with error: {exc}")

    elapsed = time.perf_counter() - start_time
    tok_per_sec = trainer.tokens_seen / max(1e-6, elapsed)

    # Extract metrics from history
    train_losses = [h["loss"] for h in history if "loss" in h]
    grad_norms = [h["grad_norm"] for h in history if "grad_norm" in h and h["grad_norm"] is not None]
    lrs = [h["learning_rate"] for h in history if "learning_rate" in h]

    summary = PilotSummary(
        pilot_stage=args.stage,
        status=status,
        requested_tokens=plan.requested_token_budget,
        actual_tokens=trainer.tokens_seen,
        tokens_per_step=plan.tokens_per_optimizer_step,
        optimizer_steps=trainer.global_step,
        initial_train_loss=train_losses[0] if train_losses else None,
        final_train_loss=train_losses[-1] if train_losses else None,
        final_validation_loss=trainer.last_val_loss,
        best_validation_loss=trainer.best_val_loss,
        min_gradient_norm=min(grad_norms) if grad_norms else None,
        max_gradient_norm=max(grad_norms) if grad_norms else None,
        initial_learning_rate=lrs[0] if lrs else None,
        final_learning_rate=lrs[-1] if lrs else None,
        elapsed_seconds=elapsed,
        tokens_per_sec=tok_per_sec,
        checkpoint_path=str(out_dir / "step-final.pt") if status == "passed" else None,
        failure_reason=failure_reason,
    )

    summary.save_json(json_path)
    print("\n" + summary.format_human_readable() + "\n")
    return summary


def main() -> None:
    args = parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
