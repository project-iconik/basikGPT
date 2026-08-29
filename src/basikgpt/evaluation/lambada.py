"""LAMBADA OpenAI last-word accuracy (greedy match of the continuation)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
import torch.nn as nn

from basikgpt.evaluation.multiple_choice import (
    EncodeTokenizer,
    fallback_token_id,
    resolve_bos_token_id,
    score_completion,
)


def split_last_word(text: str) -> tuple[str, str]:
    """Splits a LAMBADA passage into prefix and the final word (OpenAI / lm-eval convention)."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("LAMBADA text is empty.")
    if " " not in stripped:
        raise ValueError(f"LAMBADA text has no whitespace to split a last word: {stripped!r}")
    prefix, last_word = stripped.rsplit(" ", 1)
    last_word = last_word.strip()
    if not last_word:
        raise ValueError("LAMBADA last word is empty after split.")
    return prefix, last_word


@dataclass(frozen=True, slots=True)
class LambadaResult:
    """One LAMBADA last-word scoring record."""

    example_id: Any
    last_word: str
    token_count: int
    greedy_match: bool
    total_log_likelihood: float
    mean_log_likelihood: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "last_word": self.last_word,
            "token_count": self.token_count,
            "greedy_match": self.greedy_match,
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
        }


@dataclass(frozen=True, slots=True)
class LambadaSummary:
    """Aggregated last-word accuracy."""

    task: str
    split: str
    num_examples: int
    accuracy: float
    correct: int
    elapsed_seconds: float
    examples_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "split": self.split,
            "num_examples": self.num_examples,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "elapsed_seconds": self.elapsed_seconds,
            "examples_per_second": self.examples_per_second,
            "metric_primary": "accuracy",
            "score": self.accuracy,
        }


def evaluate_lambada_example(
    model: nn.Module,
    text: str,
    tokenizer: EncodeTokenizer,
    device: torch.device = torch.device("cpu"),
    max_context_length: int = 1024,
    example_id: Any = None,
) -> LambadaResult:
    """Scores greedy last-word match: prefix = text without last word, target = ' ' + last_word."""
    prefix, last_word = split_last_word(text)
    pad_id = fallback_token_id(tokenizer)
    context_tokens = tokenizer.encode(prefix)
    if not context_tokens:
        context_tokens = [pad_id]
    completion_tokens = tokenizer.encode(" " + last_word)
    if not completion_tokens:
        completion_tokens = [pad_id]
    score = score_completion(
        model=model,
        context_tokens=context_tokens,
        completion_tokens=completion_tokens,
        device=device,
        max_context_length=max_context_length,
        bos_token_id=resolve_bos_token_id(tokenizer),
    )
    return LambadaResult(
        example_id=example_id,
        last_word=last_word,
        token_count=score.token_count,
        greedy_match=score.greedy_match,
        total_log_likelihood=score.total_log_likelihood,
        mean_log_likelihood=score.mean_log_likelihood,
    )


def evaluate_lambada(
    model: nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: EncodeTokenizer,
    device: torch.device = torch.device("cpu"),
    max_examples: int | None = None,
    max_context_length: int = 1024,
    split_name: str = "test",
    progress_interval: int = 50,
) -> tuple[LambadaSummary, list[LambadaResult]]:
    """Evaluates LAMBADA OpenAI last-word accuracy over a stream of `{text: ...}` records."""
    results: list[LambadaResult] = []
    correct = 0
    t0 = time.perf_counter()
    for idx, raw in enumerate(dataset):
        if max_examples is not None and idx >= max_examples:
            break
        text = str(raw.get("text", ""))
        example_id = raw.get("id", idx)
        res = evaluate_lambada_example(
            model=model,
            text=text,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
            example_id=example_id,
        )
        results.append(res)
        if res.greedy_match:
            correct += 1
        processed = idx + 1
        if progress_interval > 0 and processed % progress_interval == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[{processed:>5} examples] "
                f"Accuracy: {correct / processed * 100:>5.2f}% ({correct}/{processed}) | "
                f"Speed: {processed / max(1e-6, elapsed):.1f} ex/s"
            )

    elapsed = time.perf_counter() - t0
    num_ex = len(results)
    summary = LambadaSummary(
        task="lambada_openai",
        split=split_name,
        num_examples=num_ex,
        accuracy=correct / max(1, num_ex),
        correct=correct,
        elapsed_seconds=elapsed,
        examples_per_second=num_ex / max(1e-6, elapsed),
    )
    return summary, results


def load_lambada_dataset(
    split: str = "test",
    local_path: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Loads LAMBADA OpenAI from a local JSON/JSONL file or Hugging Face `EleutherAI/lambada_openai`."""
    if local_path is not None:
        yield from _iter_json_records(Path(local_path))
        return
    from datasets import load_dataset

    hf_ds = load_dataset("EleutherAI/lambada_openai", split=split)
    yield from hf_ds


def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and "data" in data:
        yield from data["data"]
    else:
        raise ValueError(f"Unrecognized JSON structure in {path}")
