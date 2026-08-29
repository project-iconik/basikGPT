"""Plot english-lm-suite-v1 primary scores from benchmarks/summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = REPO_ROOT / "benchmarks" / "summary.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "whitepaper" / "figures"

MODEL_ORDER = (
    "basikgpt-2p5b",
    "basikgpt-5b",
    "gpt2",
    "SmolLM2-135M",
    "SmolLM2-360M",
    "pythia-160m",
    "pythia-410m",
    "Qwen2.5-0.5B",
)

DISPLAY_NAMES = {
    "basikgpt-2p5b": "basikGPT-1 v1.0",
    "basikgpt-5b": "basikGPT-1 v1.1",
    "gpt2": "gpt2",
    "SmolLM2-135M": "SmolLM2-135M",
    "SmolLM2-360M": "SmolLM2-360M",
    "pythia-160m": "Pythia-160M",
    "pythia-410m": "Pythia-410M",
    "Qwen2.5-0.5B": "Qwen2.5-0.5B",
}

TASKS = (
    ("hellaswag", "HellaSwag"),
    ("lambada_openai", "LAMBADA"),
    ("piqa", "PIQA"),
    ("winogrande", "WinoGrande"),
    ("arc_easy", "ARC-Easy"),
)

# Colorblind-friendly; first two highlight the basikGPT checkpoints.
COLORS = (
    "#0072B2",
    "#56B4E9",
    "#000000",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#CC79A7",
    "#999999",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write grouped and HellaSwag-vs-size figures from english-lm-suite-v1."
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY),
        help="Path to benchmarks/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for grouped.png and hellaswag_vs_size.png",
    )
    return parser.parse_args()


def load_models(summary_path: Path) -> dict[str, dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    models = payload["models"]
    missing = [model_id for model_id in MODEL_ORDER if model_id not in models]
    if missing:
        raise KeyError(f"summary.json is missing protocol models: {missing}")
    return models


def primary_score(task: dict) -> float:
    return float(task["score"]) * 100.0


def plot_grouped(models: dict[str, dict], output_path: Path) -> None:
    n_models = len(MODEL_ORDER)
    n_tasks = len(TASKS)
    x = np.arange(n_tasks)
    width = 0.10
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    for i, model_id in enumerate(MODEL_ORDER):
        scores = [primary_score(models[model_id]["tasks"][task_id]) for task_id, _ in TASKS]
        ax.bar(
            x + offsets[i],
            scores,
            width=width,
            color=COLORS[i],
            label=DISPLAY_NAMES[model_id],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in TASKS])
    ax.set_ylabel("Primary accuracy (%)")
    ax.set_ylim(0, 80)
    ax.set_title("english-lm-suite-v1 (zero-shot, shared protocol)")
    ax.axhline(25, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.axhline(50, color="#bbbbbb", linewidth=0.8, linestyle=":")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hellaswag_vs_size(models: dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rows = []
    for model_id in MODEL_ORDER:
        model = models[model_id]
        rows.append(
            (
                model["parameters"] / 1e6,
                primary_score(model["tasks"]["hellaswag"]),
                DISPLAY_NAMES[model_id],
                COLORS[MODEL_ORDER.index(model_id)],
            )
        )
    rows.sort(key=lambda row: row[0])

    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    colors = [row[3] for row in rows]
    ax.scatter(xs, ys, c=colors, s=48, zorder=3)
    for x, y, label, _color in rows:
        dy = 1.4 if "v1.1" in label else 1.2
        if label == "basikGPT-1 v1.0":
            dy = -2.2
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, dy), fontsize=8)

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("HellaSwag acc_norm (%)")
    ax.set_title("HellaSwag acc_norm vs parameter count")
    ax.set_ylim(24, 60)
    ax.axhline(25, color="#bbbbbb", linewidth=0.8, linestyle="--", label="chance 25%")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = load_models(summary_path)
    grouped_path = output_dir / "grouped.png"
    size_path = output_dir / "hellaswag_vs_size.png"
    plot_grouped(models, grouped_path)
    plot_hellaswag_vs_size(models, size_path)
    print(f"wrote {grouped_path}")
    print(f"wrote {size_path}")


if __name__ == "__main__":
    main()
