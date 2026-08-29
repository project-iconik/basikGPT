"""Unit tests for shared completion log-likelihood scoring (GPT-2 tensor and HF .logits)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from basikgpt.evaluation.multiple_choice import (
    CandidateScore,
    evaluate_multiple_choice,
    extract_logits,
    score_choices,
    score_completion,
)


class MockLogitsModel(nn.Module):
    def __init__(self, logits_map: dict[tuple[int, ...], torch.Tensor], vocab_size: int = 10):
        super().__init__()
        self.logits_map = logits_map
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq = tuple(input_ids[0].tolist())
        if seq in self.logits_map:
            return self.logits_map[seq].unsqueeze(0)
        bsz, tlen = input_ids.shape
        return torch.zeros(bsz, tlen, self.vocab_size)


class MockHFWrapper(nn.Module):
    """Returns an object with `.logits` like AutoModelForCausalLM."""

    def __init__(self, inner: MockLogitsModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor):
        return SimpleNamespace(logits=self.inner(input_ids))


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        mapping = {
            "ctx": [0, 1],
            " a": [2],
            " bb": [3, 3],
        }
        if text in mapping:
            return list(mapping[text])
        return [min(ord(ch) % 8, 7) for ch in text] or [0]

    @property
    def eot_token_id(self) -> int:
        return 0


def test_extract_logits_tensor_and_hf_object() -> None:
    context = [0, 1]
    completion = [2]
    full = tuple(context + completion)
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    tensor_model = MockLogitsModel({full: logits}, vocab_size=4)
    hf_model = MockHFWrapper(tensor_model)
    tokens = torch.tensor([list(full)])
    assert torch.equal(extract_logits(tensor_model, tokens), logits.unsqueeze(0))
    assert torch.equal(extract_logits(hf_model, tokens), logits.unsqueeze(0))


def test_greedy_match_true_and_false() -> None:
    context = [0]
    completion = [1]
    full = tuple(context + completion)
    # Position 0 predicts token 1 with a large logit → greedy match.
    match_logits = torch.tensor([[0.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    miss_logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    match_score = score_completion(
        MockLogitsModel({full: match_logits}, vocab_size=3), context, completion
    )
    miss_score = score_completion(
        MockLogitsModel({full: miss_logits}, vocab_size=3), context, completion
    )
    assert match_score.greedy_match is True
    assert miss_score.greedy_match is False
    assert match_score.token_count == 1


def test_hf_wrapper_matches_tensor_nll() -> None:
    context = [0, 1]
    completion = [2, 3]
    full = tuple(context + completion)
    logits = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    tensor_model = MockLogitsModel({full: logits}, vocab_size=4)
    hf_model = MockHFWrapper(tensor_model)
    a = score_completion(tensor_model, context, completion)
    b = score_completion(hf_model, context, completion)
    assert a.total_log_likelihood == pytest.approx(b.total_log_likelihood)
    assert a.mean_log_likelihood == pytest.approx(b.mean_log_likelihood)
    assert a.greedy_match == b.greedy_match


def test_score_choices_prefers_higher_mean_for_acc_norm() -> None:
    # Choice " a" → 1 token; " bb" → 2 tokens. Mock always emits zeros → uniform NLL.
    class Uniform(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            bsz, tlen = input_ids.shape
            return torch.zeros(bsz, tlen, 8)

    raw, norm, counts, pred_raw, pred_norm = score_choices(
        Uniform(),
        context="ctx",
        choices=["a", "bb"],
        tokenizer=DummyTokenizer(),
    )
    assert counts == [1, 2]
    assert len(raw) == 2
    assert pred_raw in (0, 1)
    assert pred_norm in (0, 1)
    # Uniform logits: mean LL is identical per token, so acc_norm ties → argmax picks first.
    assert pred_norm == 0
    # Sum LL of the longer sequence is more negative → acc_raw picks the short choice.
    assert pred_raw == 0
    assert raw[1] < raw[0]


def test_bos_is_prepended_to_context_only() -> None:
    captured: list[list[int]] = []

    class Capture(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            captured.append(input_ids[0].tolist())
            bsz, tlen = input_ids.shape
            return torch.zeros(bsz, tlen, 8)

    score_completion(
        Capture(),
        context_tokens=[4, 5],
        completion_tokens=[6],
        bos_token_id=9,
    )
    assert captured[0][0] == 9
    assert captured[0][-1] == 6
    assert captured[0][1:-1] == [4, 5]


def test_evaluate_multiple_choice_aggregation() -> None:
    class Uniform(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            bsz, tlen = input_ids.shape
            return torch.zeros(bsz, tlen, 8)

    examples = [
        (0, "ctx", ["a", "bb"], 0),
        (1, "ctx", ["a", "bb"], 1),
    ]
    summary, results = evaluate_multiple_choice(
        Uniform(),
        examples,
        DummyTokenizer(),
        task="piqa",
        split="validation",
        primary_metric="acc_norm",
        chance=0.5,
        progress_interval=0,
    )
    assert summary.num_examples == 2
    assert len(results) == 2
    assert summary.raw_correct + (2 - summary.raw_correct) == 2
    payload = summary.to_dict()
    assert payload["metric_primary"] == "acc_norm"
    assert payload["chance"] == 0.5
    assert "score" in payload


def test_candidate_score_default_greedy_match() -> None:
    score = CandidateScore(total_log_likelihood=-1.0, mean_log_likelihood=-1.0, token_count=1)
    assert score.greedy_match is False
