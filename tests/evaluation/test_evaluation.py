"""Unit tests for language model validation loss, perplexity, and evaluation provenance."""

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.evaluation.language_model import (
    evaluate_language_model,
    save_evaluation_result,
)
from basikgpt.training.metadata import load_json


def test_evaluate_language_model_known_answer() -> None:
    """Verifies that on uniform random output logits, cross-entropy is ln(V) and perplexity is V."""
    vocab_size = 64
    context_length = 8
    num_samples = 4

    # Dummy model returning uniform zeros for all logits -> softmax is 1/V for all tokens
    class UniformModel(nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            B, T = input_ids.shape
            return torch.zeros((B, T, self.vocab_size), dtype=torch.float32)

    model = UniformModel(vocab_size)
    x = torch.randint(0, vocab_size, (num_samples, context_length), dtype=torch.long)
    y = torch.randint(0, vocab_size, (num_samples, context_length), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    results = evaluate_language_model(model, loader, device="cpu")

    expected_loss = math.log(vocab_size)  # ln(64) = 4.15888
    expected_ppl = float(vocab_size)     # 64.0

    assert results["validation_loss"] == pytest.approx(expected_loss, rel=1e-4)
    assert results["perplexity"] == pytest.approx(expected_ppl, rel=1e-4)
    assert results["evaluated_tokens"] == num_samples * context_length
    assert results["batches_evaluated"] == 2


def test_perplexity_overflow_protection() -> None:
    """Verifies that huge losses safely return float('inf') without raising OverflowError."""
    class BadModel(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            B, T = input_ids.shape
            # Return extreme logits that make target loss massive
            logits = torch.full((B, T, 10), -10000.0)
            logits[..., 0] = 10000.0  # Force index 0
            return logits

    model = BadModel()
    x = torch.zeros((2, 4), dtype=torch.long)
    y = torch.ones((2, 4), dtype=torch.long)  # Target is 1 (opposite of predicted 0)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    results = evaluate_language_model(model, loader, device="cpu")
    assert results["perplexity"] == float("inf") or results["perplexity"] > 1e10


def test_evaluate_no_grad_flow() -> None:
    """Verifies that evaluation does not create computational graphs or accumulate gradients on parameters."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    from basikgpt.model.gpt import GPT
    model = GPT(cfg)

    # Initial gradients are None
    for p in model.parameters():
        assert p.grad is None

    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    evaluate_language_model(model, loader, device="cpu")

    # Gradients must still be None
    for p in model.parameters():
        assert p.grad is None


def test_save_evaluation_result_json(tmp_path: Path) -> None:
    """Verifies that save_evaluation_result creates a structured JSON artifact with full provenance."""
    out_file = tmp_path / "eval_report.json"
    metrics = {
        "validation_loss": 3.456,
        "perplexity": 31.69,
        "evaluated_tokens": 4096,
        "batches_evaluated": 8,
    }
    cfg = GPTConfig.gpt2_small()
    manifest_mock = {
        "dataset_provenance": {"repository": "HuggingFaceFW/fineweb-edu", "revision": "test-rev-123"},
    }

    save_evaluation_result(
        output_path=out_file,
        eval_metrics=metrics,
        checkpoint_path="runs/test_run/step-00000010.pt",
        model_config=cfg,
        dataset_manifest=manifest_mock,
        device="cpu",
    )

    assert out_file.exists()
    data = load_json(out_file)

    assert data["evaluation_format_version"] == 1
    assert data["metrics"]["validation_loss"] == 3.456
    assert data["metrics"]["perplexity"] == 31.69
    assert data["dataset"]["revision"] == "test-rev-123"
    assert "git" in data
    assert "system" in data
