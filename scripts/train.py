"""CLI training entrypoint for basikGPT single-device baseline pretraining."""

import argparse
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
from basikgpt.training.config import TrainingConfig
from basikgpt.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="basikGPT Single-Device Baseline Pretraining Engine (FP32)."
    )
    # Data & Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fineweb-edu-smoke",
        help="Directory containing tokenized .npy binary shards (default: data/fineweb-edu-smoke)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/baseline",
        help="Output directory for checkpoints and metrics (default: runs/baseline)",
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
    parser.add_argument("--warmup-steps", type=int, default=20, help="Linear warmup steps (default: 20)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate (default: 6e-4)")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate floor (default: 6e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for 2D weights (default: 0.1)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient clipping norm (default: 1.0)")

    # Evaluation & Logging Intervals
    parser.add_argument("--eval-interval", type=int, default=50, help="Validation interval in steps (default: 50)")
    parser.add_argument("--eval-batches", type=int, default=10, help="Number of validation batches (default: 10)")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint interval in steps (default: 50)")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps (default: 10)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cpu', 'cuda', 'auto')")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)

    print("=" * 70)
    print("  basikGPT Milestone 7: Single-Device Baseline Pretraining Engine")
    print("=" * 70)

    # 1. Configure Model
    if args.model_preset == "tiny":
        model_cfg = GPTConfig(
            vocab_size=50257,
            context_length=min(args.context_length, 64),
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

    model = GPT(model_cfg)
    print(f"Model Architecture: {args.model_preset} ({model.num_parameters():,} parameters)")
    print(f"Attention Backend:  {model_cfg.attention_backend}")

    # 2. Configure Dataset & DataLoaders
    train_shards = sorted(data_path.glob("train-*.npy"))
    val_shards = sorted(data_path.glob("validation-*.npy"))

    if not train_shards:
        raise FileNotFoundError(f"No train shards found in {data_path}")

    train_dataset = ShardedTokenDataset(train_shards, context_length=model_cfg.context_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Train dataset:      {len(train_shards)} shard(s), {train_dataset.total_tokens:,} tokens ({len(train_dataset):,} samples)")

    val_loader = None
    if val_shards:
        val_dataset = ShardedTokenDataset(val_shards, context_length=model_cfg.context_length)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        print(f"Validation dataset: {len(val_shards)} shard(s), {val_dataset.total_tokens:,} tokens ({len(val_dataset):,} samples)")

    # 3. Configure Training Engine
    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        output_dir=args.output_dir,
    )

    tokens_per_step = args.batch_size * model_cfg.context_length * args.grad_accum_steps
    print(f"Tokens/step:        {tokens_per_step:,} target tokens")
    print("=" * 70 + "\n")

    # 4. Run Training
    trainer = Trainer(
        model=model,
        config=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
