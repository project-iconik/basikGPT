"""CLI training entrypoint for basikGPT baseline and mixed-precision pretraining."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.config import GPTConfig
from basikgpt.data.shard import ShardedTokenDataset
from basikgpt.model.gpt import GPT
from basikgpt.training.accounting import calculate_training_steps
from basikgpt.training.compatibility import validate_dataset_model_compatibility
from basikgpt.training.config import TrainingConfig
from basikgpt.training.metadata import extract_dataset_provenance, load_json
from basikgpt.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="basikGPT Pretraining Engine (CPU/CUDA, FP32/BF16/FP16)."
    )
    # Experiment Naming & Paths
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: auto-generated timestamped name)",
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
        help="Base or exact output directory for checkpoints and metrics (default: runs/<run-name>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target output directory if it already exists",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )

    # Model Configuration
    parser.add_argument(
        "--model-preset",
        type=str,
        default="gpt2_small",
        choices=["gpt2_small", "tiny"],
        help="Model preset architecture (default: gpt2_small)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Context length / sequence length T (default: 1024)",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa"],
        help="Attention implementation backend (default: sdpa)",
    )

    # Optimization Hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per forward pass (default: 4)")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max-steps", type=int, default=100, help="Total global optimizer steps (default: 100)")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Target nominal training token budget. If specified, max_steps is automatically derived via ceiling arithmetic.",
    )
    parser.add_argument("--warmup-steps", type=int, default=20, help="Linear warmup steps (default: 20)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate (default: 6e-4)")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate floor (default: 6e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for 2D weights (default: 0.1)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient clipping norm (default: 1.0)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility (default: 1337)")

    # Runtime Environment & Precision
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cpu', 'cuda', 'cuda:0', 'auto') (default: auto)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Execution precision ('fp32', 'bf16', 'fp16') (default: fp32)",
    )

    # Evaluation & Logging Intervals
    parser.add_argument("--eval-interval", type=int, default=50, help="Validation interval in steps (default: 50)")
    parser.add_argument("--eval-batches", type=int, default=10, help="Number of validation batches (default: 10)")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint interval in steps (default: 50)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps (default: 10)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)

    # 1. Determine Run Name & Output Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{timestamp}-{args.model_preset}"
    out_dir = Path(args.output_dir) if args.output_dir else Path("runs") / run_name

    # 2. Configure Model
    if args.model_preset == "tiny":
        tiny_context = min(args.context_length, 64)
        if args.context_length > 64:
            print(
                f"[train] Warning: tiny preset caps context_length at 64 "
                f"(requested {args.context_length}). Using context_length={tiny_context}."
            )
        model_cfg = GPTConfig(
            vocab_size=50257,
            context_length=tiny_context,
            n_layers=2,
            n_heads=4,
            d_model=64,
            d_ff=256,
            attention_backend=args.attention_backend,
            dropout=0.0,
        )
    else:
        model_cfg = GPTConfig.gpt2_small(
            context_length=args.context_length,
            attention_backend=args.attention_backend,
            dropout=0.0,
        )

    # 3. Load & Validate Dataset Manifest
    manifest_path = data_path / "manifest.json"
    dataset_manifest = None
    if manifest_path.exists():
        dataset_manifest = load_json(manifest_path)
        validate_dataset_model_compatibility(
            model_config=model_cfg,
            manifest=dataset_manifest,
            requested_context_length=model_cfg.context_length,
        )

    # 4. Configure Dataset & DataLoaders
    train_shards = sorted(data_path.glob("train-*.npy"))
    val_shards = sorted(data_path.glob("validation-*.npy"))

    if not train_shards:
        raise FileNotFoundError(f"No train shards found in {data_path}")

    train_dataset = ShardedTokenDataset(train_shards, context_length=model_cfg.context_length)
    if len(train_dataset) == 0:
        raise ValueError(f"Training dataset is empty in {data_path} for context_length={model_cfg.context_length}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_loader = None
    if val_shards:
        val_dataset = ShardedTokenDataset(val_shards, context_length=model_cfg.context_length)
        if len(val_dataset) == 0:
            raise ValueError(f"Validation shards exist in {data_path} but 0 samples could be extracted for context_length={model_cfg.context_length}")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 5. Calculate Training Steps and Budget
    plan = None
    if args.target_tokens is not None:
        plan = calculate_training_steps(
            target_tokens=args.target_tokens,
            micro_batch_size=args.batch_size,
            context_length=model_cfg.context_length,
            grad_accum_steps=args.grad_accum_steps,
            world_size=1,
        )
        max_steps = plan.optimizer_steps
    else:
        max_steps = args.max_steps

    # 6. Instantiate Model & Config
    model = GPT(model_cfg)
    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        precision=args.precision,
        output_dir=str(out_dir),
        seed=args.seed,
    )

    tokens_per_step = args.batch_size * model_cfg.context_length * args.grad_accum_steps

    print("=" * 75)
    print("  basikGPT Milestone 7: Pretraining Engine")
    print("=" * 75)
    print(f"  Run Name:           {run_name}")
    print(f"  Output Directory:   {out_dir}")
    print(f"  Model Architecture: {args.model_preset} ({model.num_parameters():,} parameters)")
    print(f"  Context Length (T): {model_cfg.context_length}")
    print(f"  Attention Backend:  {model_cfg.attention_backend}")
    print(f"  Device:             {args.device}")
    print(f"  Precision:          {args.precision.upper()}")
    print(f"  Random Seed:        {args.seed}")
    print(f"  Batch Size (B):     {args.batch_size}")
    print(f"  Grad Accum (G):     {args.grad_accum_steps}")
    print(f"  Tokens / Step:      {tokens_per_step:,} target tokens")
    if plan is not None:
        print(f"  Target Budget:      {plan.requested_token_budget:,} requested tokens")
        print(f"  Actual Budget:      {plan.actual_token_budget:,} tokens ({plan.overshoot_tokens:+,} overshoot)")
        print(f"  Optimizer Steps:    {plan.optimizer_steps:,} steps")
    else:
        print(f"  Optimizer Steps:    {max_steps:,} steps")
    print(f"  Train Dataset:      {len(train_shards)} shard(s), {train_dataset.total_tokens:,} tokens ({len(train_dataset):,} samples)")
    if val_loader:
        print(f"  Val Dataset:        {len(val_shards)} shard(s), {val_dataset.total_tokens:,} tokens ({len(val_dataset):,} samples)")
    if dataset_manifest:
        rev = extract_dataset_provenance(dataset_manifest).get("revision") or "unknown"
        rev_display = str(rev)
        suffix = "..." if len(rev_display) > 16 else ""
        print(f"  Dataset Revision:   {rev_display[:16]}{suffix}")
    print("=" * 75 + "\n")

    # 6. Run Training
    trainer = Trainer(
        model=model,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=manifest_path if manifest_path.exists() else None,
        resume_from=args.resume,
        overwrite=args.overwrite,
    )
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
