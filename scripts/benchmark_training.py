"""Throughput and VRAM benchmarking script for basikGPT training across devices and precisions."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to pythonpath
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.trainer import Trainer, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="basikGPT Training Throughput & VRAM Benchmark."
    )
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
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size (default: 4)")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Benchmark warmup steps before timing (default: 5)")
    parser.add_argument("--measured-steps", type=int, default=20, help="Benchmark measured steps for timing (default: 20)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write benchmark JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)

    print("=" * 70)
    print("  basikGPT Training Throughput & Memory Benchmark")
    print("=" * 70)

    # 1. Model Configuration
    if args.model_preset == "tiny":
        ctx_len = min(args.context_length, 64)
        model_cfg = GPTConfig(
            vocab_size=50257,
            context_length=ctx_len,
            n_layers=2,
            n_heads=4,
            d_model=64,
            d_ff=256,
            attention_backend=args.attention_backend,
            dropout=0.0,
        )
    else:
        ctx_len = args.context_length
        model_cfg = GPTConfig.gpt2_small(
            context_length=ctx_len,
            attention_backend=args.attention_backend,
            dropout=0.0,
        )

    model = GPT(model_cfg)
    param_count = model.num_parameters()

    print(f"Model Preset:        {args.model_preset} ({param_count:,} parameters)")
    print(f"Context Length:      {ctx_len}")
    print(f"Attention Backend:   {model_cfg.attention_backend}")
    print(f"Device:              {resolved_device}")
    print(f"Precision:           {args.precision.upper()}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Grad Accum Steps:    {args.grad_accum_steps}")
    tokens_per_step = args.batch_size * ctx_len * args.grad_accum_steps
    print(f"Tokens / Step:       {tokens_per_step:,} target tokens")
    print(f"Warmup / Measured:   {args.warmup_steps} / {args.measured_steps} steps")

    # 2. Synthetic In-Memory Dataset (isolated from disk I/O)
    total_needed_micro_batches = (args.warmup_steps + args.measured_steps) * args.grad_accum_steps + 10
    torch.manual_seed(1337)
    raw_tokens = torch.randint(
        0,
        model_cfg.vocab_size,
        (total_needed_micro_batches * args.batch_size, ctx_len + 1),
        dtype=torch.long,
    )
    dataset = TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    train_cfg = TrainingConfig(
        learning_rate=6e-4,
        warmup_steps=0,
        max_steps=args.warmup_steps + args.measured_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        device=str(resolved_device),
        precision=args.precision,
        output_dir="runs/benchmark_temp",
    )

    trainer = Trainer(model=model, config=train_cfg, train_loader=loader)
    data_iter = trainer._infinite_loader(loader)

    # 3. Benchmark Warmup (Kernel compilation & memory allocation)
    print("\nWarming up execution pipeline ...")
    for _ in range(args.warmup_steps):
        trainer.train_step(data_iter)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.cuda.synchronize(resolved_device)

    # 4. Measured Timing Iterations
    print(f"Measuring {args.measured_steps} optimizer steps ...")
    start_time = time.perf_counter()

    losses: list[float] = []
    for _ in range(args.measured_steps):
        step_res = trainer.train_step(data_iter)
        losses.append(step_res["loss"])

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)

    elapsed_time = time.perf_counter() - start_time

    # 5. Metrics Computation
    total_measured_tokens = args.measured_steps * tokens_per_step
    tokens_per_sec = total_measured_tokens / max(1e-6, elapsed_time)
    ms_per_step = (elapsed_time / args.measured_steps) * 1000.0
    mean_loss = sum(losses) / len(losses)

    peak_vram_mb = None
    if resolved_device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024)

    # 6. Report Results
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Device:             {resolved_device}")
    if resolved_device.type == "cuda":
        print(f"  GPU Model:          {torch.cuda.get_device_name(resolved_device)}")
        cap = torch.cuda.get_device_capability(resolved_device)
        print(f"  Compute Capability: {cap[0]}.{cap[1]}")
    print(f"  Precision:          {args.precision.upper()}")
    print(f"  Throughput:         {tokens_per_sec:,.1f} tokens/sec")
    print(f"  Step Time:          {ms_per_step:.2f} ms/step")
    print(f"  Total Tokens:       {total_measured_tokens:,} tokens in {elapsed_time:.2f}s")
    if peak_vram_mb is not None:
        print(f"  Peak VRAM:          {peak_vram_mb:.1f} MB")
    print(f"  Mean Loss:          {mean_loss:.4f}")
    print("=" * 70 + "\n")

    # 7. Optional JSON Output
    if args.output_json:
        out_data = {
            "model_preset": args.model_preset,
            "parameter_count": param_count,
            "context_length": ctx_len,
            "attention_backend": model_cfg.attention_backend,
            "device": str(resolved_device),
            "precision": args.precision,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "total_measured_tokens": total_measured_tokens,
            "elapsed_seconds": elapsed_time,
            "tokens_per_second": tokens_per_sec,
            "ms_per_step": ms_per_step,
            "mean_loss": mean_loss,
            "peak_vram_mb": peak_vram_mb,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if resolved_device.type == "cuda":
            out_data["gpu_name"] = torch.cuda.get_device_name(resolved_device)
            out_data["cuda_version"] = torch.version.cuda
            cap = torch.cuda.get_device_capability(resolved_device)
            out_data["compute_capability"] = f"{cap[0]}.{cap[1]}"

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Benchmark data written to {out_path}")


if __name__ == "__main__":
    main()
