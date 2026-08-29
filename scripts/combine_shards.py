"""CLI: hard-link math + FineWeb train shards into one 9:1 sequential mix directory."""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.data.combine import combine_shard_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interleave OpenWebMath and FineWeb train shards (math first, FineWeb tail)."
    )
    parser.add_argument("--output", type=str, required=True, help="Mixed shard output directory")
    parser.add_argument("--fineweb-dir", type=str, required=True, help="Directory of FineWeb train-*.npy shards")
    parser.add_argument("--math-dir", type=str, required=True, help="Directory of OpenWebMath train-*.npy shards")
    parser.add_argument(
        "--val-dir",
        type=str,
        default="",
        help="Directory of FineWeb-Edu validation-*.npy shards to symlink (optional)",
    )
    parser.add_argument("--math-per-cycle", type=int, default=1, help="Math shards at the start of each cycle")
    parser.add_argument("--fineweb-per-cycle", type=int, default=9, help="FineWeb shards after math in each cycle")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite a non-empty output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_shard_directories(
        output_dir=Path(args.output),
        fineweb_dir=Path(args.fineweb_dir),
        math_dir=Path(args.math_dir),
        val_dir=Path(args.val_dir) if args.val_dir else None,
        math_per_cycle=args.math_per_cycle,
        fineweb_per_cycle=args.fineweb_per_cycle,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
