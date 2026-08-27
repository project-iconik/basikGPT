"""CLI entrypoint for intrinsic language model evaluation and perplexity benchmarking."""

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.data.shard import ShardedTokenDataset
from basikgpt.evaluation.language_model import (
    evaluate_language_model,
    save_evaluation_result,
)
from basikgpt.training.checkpoint import load_model_from_checkpoint
from basikgpt.training.metadata import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basikGPT Language Model Evaluation CLI.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .pt checkpoint file to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/fineweb-edu-smoke",
        help="Directory containing validation .npy binary shards and manifest.json (default: data/fineweb-edu-smoke)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size (default: 4)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of validation batches to evaluate (default: None for full set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (default: cpu)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional destination path for evaluation JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)

    print("=" * 70)
    print("  basikGPT Language Model Evaluation & Perplexity")
    print("=" * 70)
    print(f"  Checkpoint:      {ckpt_path}")
    print(f"  Dataset Dir:     {data_path}")
    print(f"  Device:          {args.device}")

    # 1. Load Model
    model, meta = load_model_from_checkpoint(ckpt_path, device=args.device)
    model_cfg = meta["model_config"]
    print(f"  Model Params:    {model.num_parameters():,}")
    print(f"  Context Length:  {model_cfg.context_length}")
    print(f"  Checkpoint Step: {meta.get('global_step', 0):,}")

    # 2. Load Dataset Manifest & Validation Shards
    manifest_path = data_path / "manifest.json"
    dataset_manifest = load_json(manifest_path) if manifest_path.exists() else None

    val_shards = sorted(data_path.glob("validation-*.npy"))
    if not val_shards:
        raise FileNotFoundError(f"No validation shards found matching 'validation-*.npy' in {data_path}")

    val_dataset = ShardedTokenDataset(val_shards, context_length=model_cfg.context_length)
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty for context_length={model_cfg.context_length} in {data_path}"
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"  Val Shards:      {len(val_shards)} shard(s)")
    print(f"  Val Samples:     {len(val_dataset):,} samples ({val_dataset.total_tokens:,} total tokens)")
    print("=" * 70 + "\n")

    print("Evaluating language model on validation tokens ...")
    results = evaluate_language_model(
        model=model,
        dataloader=val_loader,
        device=args.device,
        max_batches=args.max_batches,
    )

    print("=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Validation Loss:   {results['validation_loss']:.4f}")
    print(f"  Perplexity (PPL):  {results['perplexity']:.2f}")
    print(f"  Evaluated Tokens:  {results['evaluated_tokens']:,}")
    print(f"  Batches Evaluated: {results['batches_evaluated']}")
    print("=" * 70)

    # 3. Save JSON Report
    out_json = Path(args.output_json) if args.output_json else ckpt_path.parent / "evaluation.json"
    save_evaluation_result(
        output_path=out_json,
        eval_metrics=results,
        checkpoint_path=ckpt_path,
        model_config=model_cfg,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=manifest_path if manifest_path.exists() else None,
        device=args.device,
    )
    print(f"\nEvaluation summary written to: {out_json}")


if __name__ == "__main__":
    main()
