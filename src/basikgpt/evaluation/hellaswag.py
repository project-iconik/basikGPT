"""Zero-shot multiple-choice HellaSwag evaluation engine for basikGPT."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn

from basikgpt.data.tokenizer import GPT2Tokenizer
from basikgpt.evaluation.multiple_choice import (
    CandidateScore,
    EncodeTokenizer,
    fallback_token_id,
    resolve_bos_token_id,
    score_completion,
)
from basikgpt.training.reproducibility import get_git_metadata, get_system_metadata


# =====================================================================
# 1. Data Structures
# =====================================================================

@dataclass(frozen=True, slots=True)
class HellaSwagExample:
    """Parsed HellaSwag benchmark sample."""
    ind: int | str
    activity_label: str
    ctx_a: str
    ctx_b: str
    ctx: str
    endings: list[str]
    label: int


@dataclass(frozen=True, slots=True)
class HellaSwagResult:
    """Evaluation result and predicted choices for a single HellaSwag example."""
    example_id: int | str
    activity_label: str
    gold_label: int
    pred_raw: int
    pred_norm: int
    raw_scores: list[float]
    norm_scores: list[float]
    token_counts: list[int]

    @property
    def is_raw_correct(self) -> bool:
        return self.pred_raw == self.gold_label

    @property
    def is_norm_correct(self) -> bool:
        return self.pred_norm == self.gold_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "activity_label": self.activity_label,
            "gold_label": self.gold_label,
            "pred_raw": self.pred_raw,
            "pred_norm": self.pred_norm,
            "is_raw_correct": self.is_raw_correct,
            "is_norm_correct": self.is_norm_correct,
            "raw_scores": self.raw_scores,
            "norm_scores": self.norm_scores,
            "token_counts": self.token_counts,
        }


@dataclass(frozen=True, slots=True)
class HellaSwagSummary:
    """Aggregated benchmark evaluation summary."""
    benchmark: str
    split: str
    format_style: str
    num_examples: int
    raw_accuracy: float
    norm_accuracy: float
    raw_correct: int
    norm_correct: int
    elapsed_seconds: float
    examples_per_second: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. Context Formatting & BPE Boundary Handling
# =====================================================================

def format_hellaswag_context(
    example: dict[str, Any] | HellaSwagExample,
    format_style: Literal["activity_ctx", "ctx_only"] = "activity_ctx",
) -> str:
    """Formats the context string for a HellaSwag example according to benchmark conventions.

    Styles:
    - 'activity_ctx': Official lm-eval / paper standard:
        f"{activity_label}: {ctx_a} {ctx_b.capitalize()}" (or f"{activity_label}: {ctx_a}" if ctx_b is empty).
    - 'ctx_only': Raw context string 'ctx' directly without activity prefix (nanoGPT convention).
    """
    if isinstance(example, HellaSwagExample):
        activity = example.activity_label.strip()
        ctx_a = example.ctx_a.strip()
        ctx_b = example.ctx_b.strip()
        ctx_raw = example.ctx.strip()
    else:
        activity = str(example.get("activity_label", "")).strip()
        ctx_a = str(example.get("ctx_a", "")).strip()
        ctx_b = str(example.get("ctx_b", "")).strip()
        ctx_raw = str(example.get("ctx", "")).strip()

    if format_style == "ctx_only":
        if ctx_raw:
            return ctx_raw
        return f"{ctx_a} {ctx_b}".strip()

    # Default 'activity_ctx' (lm-evaluation-harness standard)
    if ctx_b:
        ctx_body = f"{ctx_a} {ctx_b.capitalize()}".strip()
    elif ctx_a:
        ctx_body = ctx_a
    else:
        ctx_body = ctx_raw

    if activity:
        return f"{activity}: {ctx_body}"
    return ctx_body


# =====================================================================
# 4. Single Example Evaluation
# =====================================================================

def evaluate_hellaswag_example(
    model: nn.Module,
    example: dict[str, Any] | HellaSwagExample,
    tokenizer: EncodeTokenizer,
    device: torch.device = torch.device("cpu"),
    format_style: Literal["activity_ctx", "ctx_only"] = "activity_ctx",
    max_context_length: int = 1024,
) -> HellaSwagResult:
    """Evaluates a 4-choice HellaSwag multiple choice example."""
    # 1. Parse and validate gold label
    if isinstance(example, HellaSwagExample):
        ex_id = example.ind
        activity_label = example.activity_label
        gold_label = example.label
        endings = example.endings
    else:
        ex_id = example.get("ind", example.get("id", "unknown"))
        activity_label = str(example.get("activity_label", ""))
        raw_label = example.get("label", None)
        if raw_label is None or raw_label == "":
            raise ValueError(f"Example {ex_id} has missing or empty gold label (test split cannot be scored).")
        try:
            gold_label = int(raw_label)
        except ValueError as err:
            raise ValueError(f"Example {ex_id} has invalid gold label '{raw_label}': must be int in 0..3") from err
        endings = example.get("endings", [])

    if gold_label not in (0, 1, 2, 3):
        raise ValueError(f"Gold label {gold_label} is out of expected range [0, 1, 2, 3].")

    if len(endings) != 4:
        raise ValueError(f"Example {ex_id} has {len(endings)} candidate endings (expected exactly 4).")

    # 2. Format and tokenize context
    context_str = format_hellaswag_context(example, format_style=format_style)
    context_tokens = tokenizer.encode(context_str)
    pad_id = fallback_token_id(tokenizer)
    if not context_tokens:
        context_tokens = [pad_id]
    bos_id = resolve_bos_token_id(tokenizer)

    # 3. Score each of the 4 candidate completions
    raw_scores: list[float] = []
    norm_scores: list[float] = []
    token_counts: list[int] = []

    for ending in endings:
        # Prepend space to match GPT-2 BPE boundary merging invariant
        completion_tokens = tokenizer.encode(" " + ending)
        if not completion_tokens:
            completion_tokens = [pad_id]

        score = score_completion(
            model=model,
            context_tokens=context_tokens,
            completion_tokens=completion_tokens,
            device=device,
            max_context_length=max_context_length,
            bos_token_id=bos_id,
        )
        raw_scores.append(score.total_log_likelihood)
        norm_scores.append(score.mean_log_likelihood)
        token_counts.append(score.token_count)

    # 4. Argmax predictions
    pred_raw = int(torch.tensor(raw_scores).argmax().item())
    pred_norm = int(torch.tensor(norm_scores).argmax().item())

    return HellaSwagResult(
        example_id=ex_id,
        activity_label=activity_label,
        gold_label=gold_label,
        pred_raw=pred_raw,
        pred_norm=pred_norm,
        raw_scores=raw_scores,
        norm_scores=norm_scores,
        token_counts=token_counts,
    )


# =====================================================================
# 5. Full Dataset Evaluation Engine
# =====================================================================

def evaluate_hellaswag(
    model: nn.Module,
    dataset: Iterable[dict[str, Any]],
    tokenizer: EncodeTokenizer | None = None,
    device: torch.device = torch.device("cpu"),
    max_examples: int | None = None,
    format_style: Literal["activity_ctx", "ctx_only"] = "activity_ctx",
    split_name: str = "validation",
    max_context_length: int = 1024,
    progress_interval: int = 10,
    model_metadata: dict[str, Any] | None = None,
) -> tuple[HellaSwagSummary, list[HellaSwagResult]]:
    """Evaluates a language model over a stream/list of HellaSwag multiple-choice examples."""
    if tokenizer is None:
        tokenizer = GPT2Tokenizer()

    results: list[HellaSwagResult] = []
    raw_correct = 0
    norm_correct = 0

    t0 = time.perf_counter()
    for idx, raw_example in enumerate(dataset):
        if max_examples is not None and idx >= max_examples:
            break

        res = evaluate_hellaswag_example(
            model=model,
            example=raw_example,
            tokenizer=tokenizer,
            device=device,
            format_style=format_style,
            max_context_length=max_context_length,
        )
        results.append(res)
        if res.is_raw_correct:
            raw_correct += 1
        if res.is_norm_correct:
            norm_correct += 1

        processed = idx + 1
        if progress_interval > 0 and (processed % progress_interval == 0 or (max_examples and processed == max_examples)):
            curr_elapsed = time.perf_counter() - t0
            eps = processed / max(1e-6, curr_elapsed)
            raw_acc = raw_correct / processed
            norm_acc = norm_correct / processed
            print(
                f"[{processed:>5} examples] "
                f"Raw Acc: {raw_acc * 100:>5.2f}% ({raw_correct}/{processed}) | "
                f"Norm Acc: {norm_acc * 100:>5.2f}% ({norm_correct}/{processed}) | "
                f"Speed: {eps:.1f} ex/s"
            )

    t1 = time.perf_counter()
    elapsed = t1 - t0
    num_ex = len(results)
    raw_acc = raw_correct / max(1, num_ex)
    norm_acc = norm_correct / max(1, num_ex)
    eps = num_ex / max(1e-6, elapsed)

    provenance_meta = {
        "git": get_git_metadata(),
        "system": get_system_metadata(),
        "device": str(device),
        "tokenizer": "GPT2Tokenizer (tiktoken gpt2)",
        "max_context_length": max_context_length,
    }
    if model_metadata:
        provenance_meta["model"] = model_metadata

    summary = HellaSwagSummary(
        benchmark="hellaswag",
        split=split_name,
        format_style=format_style,
        num_examples=num_ex,
        raw_accuracy=raw_acc,
        norm_accuracy=norm_acc,
        raw_correct=raw_correct,
        norm_correct=norm_correct,
        elapsed_seconds=elapsed,
        examples_per_second=eps,
        metadata=provenance_meta,
    )

    return summary, results


# =====================================================================
# 6. Dataset Loader Helper
# =====================================================================

def load_hellaswag_dataset(
    split: str = "validation",
    streaming: bool = False,
    local_path: str | Path | None = None,
) -> Iterable[dict[str, Any]]:
    """Loads HellaSwag dataset from local JSON/JSONL file or Hugging Face 'Rowan/hellaswag'."""
    if local_path is not None:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"Local HellaSwag dataset file not found: {p}")
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    yield from data
                elif isinstance(data, dict) and "data" in data:
                    yield from data["data"]
                else:
                    raise ValueError(f"Unrecognized JSON structure in {p}")
        return

    # Hugging Face datasets fallback
    from datasets import load_dataset
    hf_ds = load_dataset("Rowan/hellaswag", split=split, streaming=streaming)
    yield from hf_ds
