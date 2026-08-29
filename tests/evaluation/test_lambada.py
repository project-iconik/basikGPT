"""Unit tests for LAMBADA last-word split and greedy accuracy (offline fixtures)."""

from pathlib import Path

import torch
import torch.nn as nn

from basikgpt.evaluation.lambada import (
    evaluate_lambada,
    evaluate_lambada_example,
    split_last_word,
)


class DummyTok:
    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [min((ord(ch) % 7) + 1, 7) for ch in text]

    @property
    def eot_token_id(self) -> int:
        return 0


class ScriptedGreedy(nn.Module):
    """If `force_match`, every position's argmax equals the next token; else never."""

    def __init__(self, force_match: bool, vocab_size: int = 8) -> None:
        super().__init__()
        self.force_match = force_match
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, tlen = input_ids.shape
        logits = torch.full((bsz, tlen, self.vocab_size), -20.0)
        if self.force_match:
            for t in range(tlen - 1):
                target = int(input_ids[0, t + 1].item())
                logits[0, t, target] = 20.0
        else:
            logits[:, :, 0] = 20.0
        return logits


def test_split_last_word_openai_convention() -> None:
    prefix, last = split_last_word("Once upon a time there was a cat")
    assert prefix == "Once upon a time there was a"
    assert last == "cat"


def test_split_last_word_strips_and_rejects_empty() -> None:
    prefix, last = split_last_word("  hello world  \n")
    assert prefix == "hello"
    assert last == "world"


def test_lambada_greedy_match_accuracy() -> None:
    text = "the last word is apple"
    hit = evaluate_lambada_example(ScriptedGreedy(True), text, DummyTok())
    miss = evaluate_lambada_example(ScriptedGreedy(False), text, DummyTok())
    assert hit.last_word == "apple"
    assert hit.greedy_match is True
    assert miss.greedy_match is False
    assert hit.token_count >= 1


def test_evaluate_lambada_aggregates_offline_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "lambada_openai.jsonl"
    path.write_text(
        '{"text": "she opened the door"}\n{"text": "he closed the window"}\n',
        encoding="utf-8",
    )
    from basikgpt.evaluation.lambada import load_lambada_dataset

    dataset = list(load_lambada_dataset(local_path=path))
    summary, results = evaluate_lambada(
        ScriptedGreedy(True),
        dataset,
        DummyTok(),
        progress_interval=0,
    )
    assert summary.num_examples == 2
    assert summary.correct == 2
    assert summary.accuracy == 1.0
    assert summary.task == "lambada_openai"
    assert len(results) == 2
    assert summary.to_dict()["metric_primary"] == "accuracy"
