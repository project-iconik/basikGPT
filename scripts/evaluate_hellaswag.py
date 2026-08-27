"""CLI entrypoint for zero-shot HellaSwag downstream multiple-choice benchmark evaluation."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import time
import torch

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
from basikgpt.data.tokenizer import GPT2Tokenizer
from basikgpt.evaluation.hellaswag import (
    evaluate_hellaswag,
    load_hellaswag_dataset,
)
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_model_from_checkpoint
from basikgpt.training.metadata import atomic_save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="basikGPT Zero-Shot HellaSwag Multiple-Choice Evaluation CLI."
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained basikGPT .pt checkpoint file to evaluate",
    )
    model_group.add_argument(
        "--hf-reference",
        action="store_true",
        help="Load official Hugging Face pretrained weights (openai-community/gpt2 124M)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="HellaSwag split to evaluate (default: validation)",
    )
    parser.add_argument(
        "--local-dataset",
        type=str,
        default=None,
        help="Optional local JSON/JSONL dataset file path instead of downloading from Hugging Face",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of validation examples to evaluate (default: None for full set)",
    )
    parser.add_argument(
        "--format-style",
        type=str,
        choices=["activity_ctx", "ctx_only"],
        default="activity_ctx",
        help="Context formatting style: 'activity_ctx' (lm-eval standard) or 'ctx_only' (nanoGPT standard)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (default: cpu)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Frequency of progress logging in examples (default: 10)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation summary JSON",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional path to save per-example prediction records JSONL",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("  basikGPT Zero-Shot HellaSwag Multiple-Choice Evaluation")
    print("=" * 70)

    # 1. Load Model
    if args.hf_reference:
        print("Loading official Hugging Face GPT-2 weights (openai-community/gpt2) ...")
        cfg = GPTConfig.gpt2_small(dropout=0.0)
        model = GPT(cfg)
        load_hf_gpt2_weights(model, "openai-community/gpt2")
        model.to(device)
        model.eval()
        source_desc = "openai-community/gpt2 (Reference Pretrained 124M)"
        cfg_dict = asdict(cfg) if is_dataclass(cfg) else (cfg.to_dict() if hasattr(cfg, "to_dict") else cfg)
        model_meta = {
            "source": "hf-reference",
            "model_name": "openai-community/gpt2",
            "parameters": model.num_parameters(),
            "config": cfg_dict,
        }
    else:
        print(f"Loading checkpoint from {args.checkpoint} ...")
        model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
        cfg = meta["model_config"]
        source_desc = f"Checkpoint: {args.checkpoint} (Step {meta.get('global_step', 0)})"
        cfg_dict = asdict(cfg) if is_dataclass(cfg) else (cfg.to_dict() if hasattr(cfg, "to_dict") else cfg)
        model_meta = {
            "source": "checkpoint",
            "checkpoint_path": args.checkpoint,
            "global_step": meta.get("global_step", 0),
            "parameters": model.num_parameters(),
            "config": cfg_dict,
        }

    print(f"  Model:           {source_desc}")
    print(f"  Parameters:      {model.num_parameters():,}")
    print(f"  Context Limit:   {cfg.context_length}")
    print(f"  Device:          {device}")
    print(f"  Split:           {args.split}")
    print(f"  Format Style:    {args.format_style}")
    print(f"  Max Examples:    {'All' if args.max_examples is None else args.max_examples}")
    print("-" * 70)

    # 2. Load Dataset
    print("Loading HellaSwag dataset ...")
    dataset_iter = load_hellaswag_dataset(
        split=args.split,
        streaming=True,
        local_path=args.local_dataset,
    )
    tokenizer = GPT2Tokenizer()

    # 3. Run Evaluation Loop
    print("\nStarting evaluation ...\n")
    summary, results = evaluate_hellaswag(
        model=model,
        dataset=dataset_iter,
        tokenizer=tokenizer,
        device=device,
        max_examples=args.max_examples,
        format_style=args.format_style,
        split_name=args.split,
        max_context_length=cfg.context_length,
        progress_interval=args.progress_interval,
        model_metadata=model_meta,
    )

    # 4. Report Final Results
    print("\n" + "=" * 70)
    print("  HELLASWAG BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Examples Evaluated:       {summary.num_examples:,}")
    print(f"  Raw Accuracy (Sum LL):    {summary.raw_accuracy * 100:.2f}% ({summary.raw_correct}/{summary.num_examples})")
    print(f"  Norm Accuracy (Mean LL):  {summary.norm_accuracy * 100:.2f}% ({summary.norm_correct}/{summary.num_examples})")
    print(f"  Elapsed Time:             {summary.elapsed_seconds:.2f}s")
    print(f"  Throughput:               {summary.examples_per_second:.2f} examples/s")
    print("=" * 70)

    # 5. Export JSON / JSONL
    if args.output_json:
        atomic_save_json(args.output_json, summary.to_dict())
        print(f"Summary JSON saved to: {args.output_json}")

    if args.output_jsonl:
        out_jsonl = Path(args.output_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"Per-example JSONL saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
