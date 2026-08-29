"""Zero-shot English task loaders: PIQA, WinoGrande, ARC-Easy, plus shared JSON helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from basikgpt.evaluation.hellaswag import load_hellaswag_dataset
from basikgpt.evaluation.lambada import load_lambada_dataset
from basikgpt.evaluation.multiple_choice import evaluate_multiple_choice


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yields dict records from a JSON array or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset file not found: {p}")
    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and "data" in data:
        yield from data["data"]
    else:
        raise ValueError(f"Unrecognized JSON structure in {p}")


def _load_hf_or_local(
    local_path: str | Path | None,
    hf_path: str,
    split: str,
    config_name: str | None = None,
) -> Iterable[dict[str, Any]]:
    if local_path is not None:
        yield from iter_json_records(local_path)
        return
    from datasets import load_dataset

    if config_name is None:
        hf_ds = load_dataset(hf_path, split=split)
    else:
        hf_ds = load_dataset(hf_path, config_name, split=split)
    yield from hf_ds


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------

def parse_piqa_example(doc: dict[str, Any], index: int = 0) -> tuple[Any, str, list[str], int]:
    """goal → context; sol1/sol2 → choices; label in {0, 1}."""
    example_id = doc.get("id", index)
    context = str(doc["goal"]).strip()
    choices = [str(doc["sol1"]).strip(), str(doc["sol2"]).strip()]
    gold = int(doc["label"])
    if gold not in (0, 1):
        raise ValueError(f"PIQA example {example_id} has invalid label {gold}.")
    return example_id, context, choices, gold


def load_piqa_dataset(
    split: str = "validation",
    local_path: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Loads PIQA from local JSON/JSONL or Hugging Face `baber/piqa` (parquet mirror of ybisk/piqa)."""
    yield from _load_hf_or_local(local_path, "baber/piqa", split)


def iter_piqa_scored(dataset: Iterable[dict[str, Any]]) -> Iterable[tuple[Any, str, list[str], int]]:
    for i, doc in enumerate(dataset):
        yield parse_piqa_example(doc, i)


# ---------------------------------------------------------------------------
# WinoGrande
# ---------------------------------------------------------------------------

def parse_winogrande_example(doc: dict[str, Any], index: int = 0) -> tuple[Any, str, list[str], int]:
    """Blank `_` → left context; each choice is option + right remainder.

    Gold answer is `"1"` / `"2"` (1-indexed) mapped to 0 / 1.
    Primary metric is acc_raw (shared suffix; option tokens are scored).
    """
    example_id = doc.get("qID", doc.get("id", index))
    sentence = str(doc["sentence"])
    if "_" not in sentence:
        raise ValueError(f"WinoGrande example {example_id} has no '_' blank.")
    left, right = sentence.split("_", 1)
    context = left.rstrip()
    option1 = str(doc["option1"]).strip()
    option2 = str(doc["option2"]).strip()
    remainder = right.strip()
    if remainder:
        choices = [f"{option1} {remainder}", f"{option2} {remainder}"]
    else:
        choices = [option1, option2]
    raw_answer = str(doc["answer"]).strip()
    if raw_answer not in ("1", "2"):
        raise ValueError(f"WinoGrande example {example_id} has invalid answer {raw_answer!r}.")
    gold = int(raw_answer) - 1
    return example_id, context, choices, gold


def load_winogrande_dataset(
    split: str = "validation",
    local_path: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Loads WinoGrande from local JSON/JSONL or `allenai/winogrande` config `winogrande_xl`."""
    yield from _load_hf_or_local(
        local_path, "allenai/winogrande", split, config_name="winogrande_xl"
    )


def iter_winogrande_scored(
    dataset: Iterable[dict[str, Any]],
) -> Iterable[tuple[Any, str, list[str], int]]:
    for i, doc in enumerate(dataset):
        yield parse_winogrande_example(doc, i)


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------

def parse_arc_example(doc: dict[str, Any], index: int = 0) -> tuple[Any, str, list[str], int]:
    """Question as context; choice texts as completions; gold from answerKey vs labels."""
    example_id = doc.get("id", index)
    question = str(doc["question"]).strip()
    raw_choices = doc["choices"]
    texts = [str(t).strip() for t in raw_choices["text"]]
    labels = [str(lab).strip() for lab in raw_choices["label"]]
    if not texts:
        raise ValueError(f"ARC example {example_id} has no choices.")
    key = str(doc["answerKey"]).strip()
    if key in labels:
        gold = labels.index(key)
    else:
        raise ValueError(
            f"ARC example {example_id} answerKey {key!r} not in labels {labels}."
        )
    context = f"Question: {question}\nAnswer:"
    return example_id, context, texts, gold


def load_arc_easy_dataset(
    split: str = "test",
    local_path: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Loads ARC-Easy from local JSON/JSONL or Hugging Face `allenai/ai2_arc` config `ARC-Easy`."""
    yield from _load_hf_or_local(local_path, "allenai/ai2_arc", split, config_name="ARC-Easy")


def iter_arc_easy_scored(
    dataset: Iterable[dict[str, Any]],
) -> Iterable[tuple[Any, str, list[str], int]]:
    for i, doc in enumerate(dataset):
        yield parse_arc_example(doc, i)


# ---------------------------------------------------------------------------
# Task registry (protocol)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One zero-shot suite task: split, primary metric, chance rate, loader."""

    name: str
    split: str
    primary_metric: str
    chance: float | None
    hf_path: str
    hf_config: str | None = None


PROTOCOL_TASKS: dict[str, TaskSpec] = {
    "hellaswag": TaskSpec(
        name="hellaswag",
        split="validation",
        primary_metric="acc_norm",
        chance=0.25,
        hf_path="Rowan/hellaswag",
    ),
    "lambada_openai": TaskSpec(
        name="lambada_openai",
        split="test",
        primary_metric="accuracy",
        chance=None,
        hf_path="EleutherAI/lambada_openai",
    ),
    "piqa": TaskSpec(
        name="piqa",
        split="validation",
        primary_metric="acc_norm",
        chance=0.50,
        hf_path="baber/piqa",
    ),
    "winogrande": TaskSpec(
        name="winogrande",
        split="validation",
        primary_metric="acc_raw",
        chance=0.50,
        hf_path="allenai/winogrande",
        hf_config="winogrande_xl",
    ),
    "arc_easy": TaskSpec(
        name="arc_easy",
        split="test",
        primary_metric="acc_norm",
        chance=None,
        hf_path="allenai/ai2_arc",
        hf_config="ARC-Easy",
    ),
}

DEFAULT_SUITE_TASKS = ("hellaswag", "lambada_openai", "piqa", "winogrande", "arc_easy")

# Re-export evaluate_multiple_choice for callers that score PIQA / WG / ARC.
__all__ = [
    "PROTOCOL_TASKS",
    "DEFAULT_SUITE_TASKS",
    "TaskSpec",
    "evaluate_multiple_choice",
    "iter_json_records",
    "load_hellaswag_dataset",
    "load_lambada_dataset",
    "load_piqa_dataset",
    "load_winogrande_dataset",
    "load_arc_easy_dataset",
    "parse_piqa_example",
    "parse_winogrande_example",
    "parse_arc_example",
    "iter_piqa_scored",
    "iter_winogrande_scored",
    "iter_arc_easy_scored",
]
