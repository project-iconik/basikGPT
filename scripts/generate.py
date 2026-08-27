"""CLI entrypoint for autoregressive text generation with basikGPT."""

import argparse
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
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.generate import generate
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_model_from_checkpoint
from basikgpt.training.metadata import atomic_save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basikGPT Autoregressive Text Generation CLI.")

    # Model Source
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained .pt checkpoint file",
    )
    group.add_argument(
        "--hf-reference",
        action="store_true",
        help="Load official pretrained weights from HuggingFace (openai-community/gpt2)",
    )

    # Generation Parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="The history of artificial intelligence",
        help="Input text prompt for continuation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum new tokens to generate (default: 30)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k filtering threshold (default: None)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus filtering threshold (default: None)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling reproducibility (default: 1337)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run generation on (default: cpu)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save generation metadata JSON",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("  basikGPT Autoregressive Text Generation")
    print("=" * 70)

    # 1. Load Model
    if args.hf_reference:
        print("Loading official Hugging Face GPT-2 weights (openai-community/gpt2) ...")
        cfg = GPTConfig.gpt2_small()
        model = GPT(cfg)
        load_hf_gpt2_weights(model, "openai-community/gpt2")
        model.to(device)
        model.eval()
        source_desc = "openai-community/gpt2 (Reference Pretrained 124M)"
    else:
        print(f"Loading checkpoint from {args.checkpoint} ...")
        model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
        cfg = meta["model_config"]
        source_desc = f"Checkpoint: {args.checkpoint} (Step {meta.get('global_step', 0)})"

    print(f"  Model:        {source_desc}")
    print(f"  Parameters:   {model.num_parameters():,}")
    print(f"  Context:      {cfg.context_length}")
    print(f"  Device:       {device}")
    print(f"  Decoding:     {'Sampling' if args.do_sample else 'Greedy Argmax'}")
    if args.do_sample:
        print(f"  Temperature:  {args.temperature}")
        print(f"  Top-k:        {args.top_k}")
        print(f"  Top-p:        {args.top_p}")
        print(f"  Seed:         {args.seed}")
    print("-" * 70)

    # 2. Tokenize Prompt
    tokenizer = GPT2Tokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        print("[Warning] Prompt was empty; starting with empty token sequence.")
        prompt_tokens = [tokenizer.eot_token_id]

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    print(f"Prompt ({len(prompt_tokens)} tokens):")
    print(f"  {repr(args.prompt)}\n")

    # 3. Configure & Execute Generation
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        seed=args.seed,
        stop_on_eot=True,
        eot_token_id=tokenizer.eot_token_id,
    )

    t0 = time.perf_counter()
    output_ids = generate(model, input_ids, config=gen_config)
    t1 = time.perf_counter()

    generated_tokens = output_ids[0].tolist()[len(prompt_tokens):]
    generated_text = tokenizer.decode(generated_tokens)
    full_text = tokenizer.decode(output_ids[0].tolist())
    elapsed = t1 - t0
    tok_per_sec = len(generated_tokens) / max(1e-6, elapsed)

    print("=" * 70)
    print("Generated Continuation:")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)
    print(f"Tokens Generated: {len(generated_tokens)} in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

    # 4. Optional JSON export
    if args.output_json:
        record = {
            "model_source": source_desc,
            "prompt": args.prompt,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generated_text": generated_text,
            "full_text": full_text,
            "generation_config": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "do_sample": args.do_sample,
                "seed": args.seed,
            },
            "elapsed_seconds": elapsed,
            "tokens_per_second": tok_per_sec,
        }
        atomic_save_json(args.output_json, record)
        print(f"Generation record written to {args.output_json}")


if __name__ == "__main__":
    main()
