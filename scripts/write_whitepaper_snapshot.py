"""CLI to assemble copy-ready whitepaper tables from a finished training run directory."""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.training.whitepaper import write_whitepaper_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write whitepaper_snapshot.json and WHITEPAPER.md from a basikGPT run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training output directory containing run.json, summary.json, and metrics.jsonl",
    )
    parser.add_argument(
        "--evaluation-json",
        type=str,
        default=None,
        help="Optional path to scripts/evaluate.py JSON (default: <run-dir>/evaluation.json)",
    )
    parser.add_argument(
        "--hellaswag-json",
        type=str,
        default=None,
        help="Optional path to HellaSwag JSON (default: <run-dir>/hellaswag.json)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Destination JSON path (default: <run-dir>/whitepaper_snapshot.json)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Destination markdown path (default: <run-dir>/WHITEPAPER.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    written = write_whitepaper_snapshot(
        run_dir,
        evaluation_json=args.evaluation_json,
        hellaswag_json=args.hellaswag_json,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(f"Whitepaper JSON written to: {written['json']}")
    print(f"Whitepaper markdown written to: {written['markdown']}")


if __name__ == "__main__":
    main()
