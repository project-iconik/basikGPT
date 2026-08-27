"""GPT-2 Small micro-batch capacity probe. OOM is recorded; batch size is never auto-reduced."""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.training.gpu_qualification import (
    collect_gpu_environment,
    probe_microbatch_capacity,
    save_gpu_qualification_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe GPT-2 Small micro-batch capacity at T=1024 with PyTorch SDPA."
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--attention-backend", type=str, default="sdpa", choices=["eager", "sdpa"])
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--batch-candidates", type=str, default="1,2,4,8,16")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output-json", type=str, default="runs/microbatch_capacity.json")
    parser.add_argument("--output-dir", type=str, default="runs/capacity_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = collect_gpu_environment()
    if not env["cuda"]["cuda_available"]:
        raise RuntimeError("CUDA is not available; capacity probe requires a GPU.")
    if args.precision == "bf16" and not env["cuda"]["bf16_supported"]:
        raise RuntimeError("Requested precision 'bf16' but this GPU does not support bfloat16.")

    candidates = [int(part.strip()) for part in args.batch_candidates.split(",") if part.strip()]
    rows = probe_microbatch_capacity(
        precision=args.precision,
        attention_backend=args.attention_backend,
        context_length=args.context_length,
        grad_accum_steps=args.grad_accum_steps,
        batch_candidates=candidates,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    payload = {
        "environment": env,
        "attention_backend": args.attention_backend,
        "rows": rows,
    }
    path = save_gpu_qualification_summary(args.output_json, payload)
    print("Batch | Status | Peak allocated bytes | Peak reserved bytes | OOM stage")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['micro_batch_size']:<5} | {row['status']:<6} | "
            f"{row['peak_allocated_vram_bytes']} | {row['peak_reserved_vram_bytes']} | {row['oom_stage']}"
        )
    print(f"wrote: {path}")


if __name__ == "__main__":
    main()
