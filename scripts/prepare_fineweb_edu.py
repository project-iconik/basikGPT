"""CLI script for streaming, tokenizing, and sharding FineWeb-Edu for basikGPT."""

import argparse
from pathlib import Path
import sys

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.data.pipeline import prepare_fineweb_edu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream, tokenize with GPT-2 BPE, and shard HuggingFace FineWeb-Edu."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fineweb-edu-gpt2",
        help="Target output directory for shards and manifest (default: data/fineweb-edu-gpt2)",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset repository (default: HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="sample-10BT",
        help="Dataset configuration name (default: sample-10BT, canonical main: default)",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        help="Pinned commit SHA for reproducibility (default: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9)",
    )
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=100_000,
        help="Maximum GPT-2 training tokens to extract (default: 100,000)",
    )
    parser.add_argument(
        "--max-validation-tokens",
        type=int,
        default=10_000,
        help="Maximum GPT-2 validation tokens to extract (default: 10,000)",
    )
    parser.add_argument(
        "--shard-token-target",
        type=int,
        default=50_000,
        help="Number of tokens per binary shard file (default: 50,000; production: 1,000,000)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.005,
        help="Fraction of document IDs routed to validation split (default: 0.005 = 0.5%)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists and is not empty",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Logging interval in documents seen (default: 500)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("  basikGPT Milestone 6: FineWeb-Edu Tokenization & Data Pipeline")
    print("=" * 70)
    print(f"Output directory:      {args.output}")
    print(f"Dataset repo:          {args.dataset_repo}")
    print(f"Dataset config:        {args.dataset_config}")
    print(f"Dataset revision:      {args.dataset_revision}")
    print(f"Max train tokens:      {args.max_train_tokens:,}")
    print(f"Max validation tokens: {args.max_validation_tokens:,}")
    print(f"Shard token target:    {args.shard_token_target:,}")
    print(f"Validation fraction:   {args.val_fraction * 100:.2f}%")
    print("=" * 70 + "\n")

    manifest = prepare_fineweb_edu(
        output_dir=args.output,
        dataset_repo=args.dataset_repo,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        max_train_tokens=args.max_train_tokens,
        max_validation_tokens=args.max_validation_tokens,
        shard_token_target=args.shard_token_target,
        val_fraction=args.val_fraction,
        overwrite=args.overwrite,
        log_interval=args.log_interval,
    )
    print("\nDataset preparation finished successfully!")


if __name__ == "__main__":
    main()
