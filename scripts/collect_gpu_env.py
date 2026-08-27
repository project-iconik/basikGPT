"""Collect RunPod / CUDA environment metadata for basikGPT GPU qualification."""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.training.gpu_qualification import collect_gpu_environment, save_gpu_qualification_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CUDA/GPU environment metadata (no secrets).")
    parser.add_argument("--output-json", type=str, default="runs/gpu_env.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = collect_gpu_environment()
    path = save_gpu_qualification_summary(args.output_json, payload)
    cuda = payload["cuda"]
    git = payload["git"]
    print(f"git_commit:     {git.get('git_commit')}")
    print(f"git_dirty:      {git.get('git_dirty')}")
    print(f"provider:       {cuda.get('provider')}")
    print(f"cuda_available: {cuda.get('cuda_available')}")
    print(f"gpu_name:       {cuda.get('gpu_name')}")
    print(f"gpu_count:      {cuda.get('gpu_count')}")
    print(f"total_vram:     {cuda.get('total_vram_bytes')}")
    print(f"driver:         {cuda.get('nvidia_driver')}")
    print(f"cuda_runtime:   {cuda.get('cuda_runtime')}")
    print(f"capability:     {cuda.get('compute_capability')}")
    print(f"bf16:           {cuda.get('bf16_supported')}")
    print(f"os:             {cuda.get('os')}")
    print(f"wrote:          {path}")


if __name__ == "__main__":
    main()
