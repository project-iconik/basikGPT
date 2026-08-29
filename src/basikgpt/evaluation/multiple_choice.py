"""Shared completion log-likelihood scoring for zero-shot multiple-choice tasks.

Works with both basikGPT (`forward` returns a logits tensor) and Hugging Face
causal LMs (`forward` returns an object with `.logits`).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodeTokenizer(Protocol):
    """Minimal tokenizer used by the English LM suite."""

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Scoring metrics for a single candidate completion."""

    total_log_likelihood: float
    mean_log_likelihood: float
    token_count: int
    greedy_match: bool = False


def extract_logits(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Returns next-token logits of shape (batch, seq, vocab) from either model family."""
    try:
        output = model(
            input_ids=tokens,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    except TypeError:
        output = model(tokens)
    if torch.is_tensor(output):
        return output
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if logits is None:
        raise TypeError(f"Model output has no logits: {type(output)!r}")
    return logits.detach()


def fallback_token_id(tokenizer: EncodeTokenizer) -> int:
    """EOS / EOT id used when a context or choice encodes to an empty token list."""
    token_id = getattr(tokenizer, "eot_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", 0)
    return int(token_id if token_id is not None else 0)


def resolve_bos_token_id(tokenizer: EncodeTokenizer) -> int | None:
    """Optional BOS id prepended to context only (Llama-style tokenizers)."""
    bos = getattr(tokenizer, "bos_token_id", None)
    return int(bos) if bos is not None else None


def score_completion(
    model: nn.Module,
    context_tokens: list[int],
    completion_tokens: list[int],
    device: torch.device = torch.device("cpu"),
    max_context_length: int = 1024,
    bos_token_id: int | None = None,
) -> CandidateScore:
    """Computes conditional completion-only log-likelihood given context tokens.

    Autoregressive shift alignment:
      tokens: [x_0, x_1, ..., x_{P-1}, x_P, ..., x_{T-1}]
      Logits at position k-1 parameterize P(x_k | x_{<k}).
      shift_logits = logits[:, P-1 : T-1, :]
      shift_targets = tokens[:, P : T]
    """
    if bos_token_id is not None:
        if not context_tokens or context_tokens[0] != bos_token_id:
            context_tokens = [bos_token_id] + list(context_tokens)

    if not context_tokens:
        raise ValueError("Context tokens list must contain at least 1 token.")
    if not completion_tokens:
        raise ValueError("Completion tokens list must not be empty.")

    M = len(completion_tokens)
    if M >= max_context_length:
        raise ValueError(
            f"Completion token length {M} exceeds maximum context length {max_context_length}."
        )

    total_len = len(context_tokens) + M
    if total_len > max_context_length:
        keep_context_len = max(1, max_context_length - M)
        context_tokens = context_tokens[-keep_context_len:]

    full_sequence = context_tokens + completion_tokens
    P = len(context_tokens)
    T = len(full_sequence)
    tokens = torch.tensor([full_sequence], dtype=torch.long, device=device)

    model.eval()
    with torch.inference_mode():
        logits = extract_logits(model, tokens).float()
        shift_logits = logits[:, P - 1 : T - 1, :].contiguous()
        shift_targets = tokens[:, P:T].contiguous()
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        target_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_targets.unsqueeze(-1)
        ).squeeze(-1)
        total_ll = float(target_log_probs.sum().item())
        mean_ll = float(target_log_probs.mean().item())
        greedy = log_probs.argmax(dim=-1)
        greedy_match = bool((greedy == shift_targets).all().item())

    return CandidateScore(
        total_log_likelihood=total_ll,
        mean_log_likelihood=mean_ll,
        token_count=M,
        greedy_match=greedy_match,
    )


def encode_choice_completion(tokenizer: EncodeTokenizer, ending: str) -> list[int]:
    """Encodes a multiple-choice ending with a leading space (GPT-2 BPE boundary)."""
    tokens = tokenizer.encode(" " + ending)
    return tokens


def score_choices(
    model: nn.Module,
    context: str,
    choices: list[str],
    tokenizer: EncodeTokenizer,
    device: torch.device = torch.device("cpu"),
    max_context_length: int = 1024,
    pad_token_id: int | None = None,
) -> tuple[list[float], list[float], list[int], int, int]:
    """Scores each choice; returns raw scores, norm scores, token counts, pred_raw, pred_norm."""
    pad_id = fallback_token_id(tokenizer) if pad_token_id is None else pad_token_id
    context_tokens = tokenizer.encode(context)
    if not context_tokens:
        context_tokens = [pad_id]
    bos_id = resolve_bos_token_id(tokenizer)

    raw_scores: list[float] = []
    norm_scores: list[float] = []
    token_counts: list[int] = []
    for ending in choices:
        completion_tokens = encode_choice_completion(tokenizer, ending)
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

    pred_raw = int(torch.tensor(raw_scores).argmax().item())
    pred_norm = int(torch.tensor(norm_scores).argmax().item())
    return raw_scores, norm_scores, token_counts, pred_raw, pred_norm


@dataclass(frozen=True, slots=True)
class MultipleChoiceResult:
    """One scored multiple-choice example."""

    example_id: Any
    gold_label: int
    pred_raw: int
    pred_norm: int
    raw_scores: list[float]
    norm_scores: list[float]
    token_counts: list[int]
    num_choices: int

    @property
    def is_raw_correct(self) -> bool:
        return self.pred_raw == self.gold_label

    @property
    def is_norm_correct(self) -> bool:
        return self.pred_norm == self.gold_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "gold_label": self.gold_label,
            "pred_raw": self.pred_raw,
            "pred_norm": self.pred_norm,
            "is_raw_correct": self.is_raw_correct,
            "is_norm_correct": self.is_norm_correct,
            "raw_scores": self.raw_scores,
            "norm_scores": self.norm_scores,
            "token_counts": self.token_counts,
            "num_choices": self.num_choices,
        }


@dataclass(frozen=True, slots=True)
class MultipleChoiceSummary:
    """Aggregated accuracy for a multiple-choice task."""

    task: str
    split: str
    num_examples: int
    raw_accuracy: float
    norm_accuracy: float
    raw_correct: int
    norm_correct: int
    elapsed_seconds: float
    examples_per_second: float
    primary_metric: str
    chance: float | None = None

    def primary_score(self) -> float:
        if self.primary_metric == "acc_raw":
            return self.raw_accuracy
        return self.norm_accuracy

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task": self.task,
            "split": self.split,
            "num_examples": self.num_examples,
            "acc_raw": self.raw_accuracy,
            "acc_norm": self.norm_accuracy,
            "raw_correct": self.raw_correct,
            "norm_correct": self.norm_correct,
            "elapsed_seconds": self.elapsed_seconds,
            "examples_per_second": self.examples_per_second,
            "metric_primary": self.primary_metric,
            "score": self.primary_score(),
        }
        if self.chance is not None:
            payload["chance"] = self.chance
        return payload


def evaluate_multiple_choice(
    model: nn.Module,
    examples: Iterable[tuple[Any, str, list[str], int]],
    tokenizer: EncodeTokenizer,
    device: torch.device = torch.device("cpu"),
    max_context_length: int = 1024,
    max_examples: int | None = None,
    task: str = "multiple_choice",
    split: str = "validation",
    primary_metric: str = "acc_norm",
    chance: float | None = None,
    progress_interval: int = 50,
    progress_fn: Callable[[int, int, int, float], None] | None = None,
) -> tuple[MultipleChoiceSummary, list[MultipleChoiceResult]]:
    """Scores an iterable of (example_id, context, choices, gold_index) records."""
    results: list[MultipleChoiceResult] = []
    raw_correct = 0
    norm_correct = 0
    t0 = time.perf_counter()
    for idx, (example_id, context, choices, gold_label) in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break
        raw_scores, norm_scores, token_counts, pred_raw, pred_norm = score_choices(
            model=model,
            context=context,
            choices=choices,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
        )
        result = MultipleChoiceResult(
            example_id=example_id,
            gold_label=gold_label,
            pred_raw=pred_raw,
            pred_norm=pred_norm,
            raw_scores=raw_scores,
            norm_scores=norm_scores,
            token_counts=token_counts,
            num_choices=len(choices),
        )
        results.append(result)
        if result.is_raw_correct:
            raw_correct += 1
        if result.is_norm_correct:
            norm_correct += 1
        processed = idx + 1
        if progress_interval > 0 and processed % progress_interval == 0:
            elapsed = time.perf_counter() - t0
            if progress_fn is not None:
                progress_fn(processed, raw_correct, norm_correct, elapsed)
            else:
                print(
                    f"[{processed:>5} examples] "
                    f"Raw Acc: {raw_correct / processed * 100:>5.2f}% | "
                    f"Norm Acc: {norm_correct / processed * 100:>5.2f}% | "
                    f"Speed: {processed / max(1e-6, elapsed):.1f} ex/s"
                )

    elapsed = time.perf_counter() - t0
    num_ex = len(results)
    summary = MultipleChoiceSummary(
        task=task,
        split=split,
        num_examples=num_ex,
        raw_accuracy=raw_correct / max(1, num_ex),
        norm_accuracy=norm_correct / max(1, num_ex),
        raw_correct=raw_correct,
        norm_correct=norm_correct,
        elapsed_seconds=elapsed,
        examples_per_second=num_ex / max(1e-6, elapsed),
        primary_metric=primary_metric,
        chance=chance,
    )
    return summary, results

