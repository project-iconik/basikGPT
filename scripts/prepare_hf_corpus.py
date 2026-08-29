"""CLI: stream, tokenize, and shard a generic HuggingFace text corpus for basikGPT."""

import argparse
from pathlib import Path
import os
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.data.pipeline import prepare_hf_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a HuggingFace dataset, tokenize with GPT-2 BPE, and write uint16 shards."
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory for shards and manifest")
    parser.add_argument("--dataset-repo", type=str, required=True, help="HuggingFace dataset repository")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="",
        help="Dataset configuration name (empty = default config)",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default="main",
        help="Branch, tag, or commit SHA (resolved to a commit SHA and stored in the manifest)",
    )
    parser.add_argument("--text-field", type=str, default="text", help="Document field containing raw text")
    parser.add_argument("--max-train-tokens", type=int, required=True, help="Maximum GPT-2 training tokens")
    parser.add_argument(
        "--max-validation-tokens",
        type=int,
        default=0,
        help="Maximum GPT-2 validation tokens (default: 0)",
    )
    parser.add_argument(
        "--shard-token-target",
        type=int,
        default=1_000_000,
        help="Tokens per binary shard (default: 1,000,000)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of document IDs routed to validation (default: 0)",
    )
    parser.add_argument(
        "--dataset-license",
        type=str,
        default="ODC-By 1.0",
        help="License string stored in the manifest",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="HuggingFace streaming corpus",
        help="Provenance selection note stored in the manifest",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite a non-empty output directory")
    parser.add_argument("--log-interval", type=int, default=500, help="Progress log interval in documents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("  basikGPT HuggingFace corpus preparation")
    print("=" * 70)
    print(f"Output directory:      {args.output}")
    print(f"Dataset repo:          {args.dataset_repo}")
    print(f"Dataset config:        {args.dataset_config or '(default)'}")
    print(f"Dataset revision:      {args.dataset_revision}")
    print(f"Text field:            {args.text_field}")
    print(f"Max train tokens:      {args.max_train_tokens:,}")
    print(f"Max validation tokens: {args.max_validation_tokens:,}")
    print(f"Shard token target:    {args.shard_token_target:,}")
    print(f"Validation fraction:   {args.val_fraction * 100:.2f}%")
    print("=" * 70 + "\n")

    prepare_hf_corpus(
        output_dir=args.output,
        dataset_repo=args.dataset_repo,
        dataset_config=args.dataset_config or None,
        dataset_revision=args.dataset_revision,
        max_train_tokens=args.max_train_tokens,
        max_validation_tokens=args.max_validation_tokens,
        shard_token_target=args.shard_token_target,
        val_fraction=args.val_fraction,
        overwrite=args.overwrite,
        log_interval=args.log_interval,
        text_field=args.text_field,
        dataset_license=args.dataset_license,
        selection=args.selection,
    )
    print("\nDataset preparation finished successfully!")
    # HuggingFace datasets/pyarrow can abort during interpreter shutdown after a
    # successful streaming run (PyGILState_Release). Exit without running atexit.
    os._exit(0)


if __name__ == "__main__":
    main()
