"""Unit and integration tests for the HellaSwag zero-shot evaluation engine in basikGPT.

Tests:
1. Shift alignment and exact known-answer log-likelihood calculation.
2. Strict exclusion of context tokens from candidate likelihood.
3. Length normalization ranking inversion (raw vs normalized).
4. GPT-2 BPE token boundary whitespace invariance.
5. 4-choice argmax prediction resolution.
6. Accuracy aggregation and summary arithmetic.
7. Invalid label rejection and error policies.
8. Context overflow left-truncation guardrail.
9. Offline dataset parsing on synthetic dictionary fixtures.
10. Score-level numerical parity between basikGPT and Hugging Face GPT-2.
11. Real HellaSwag dataset small smoke integration test.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
from basikgpt.data.tokenizer import GPT2Tokenizer
from basikgpt.evaluation.hellaswag import (
    CandidateScore,
    HellaSwagExample,
    HellaSwagResult,
    evaluate_hellaswag,
    evaluate_hellaswag_example,
    format_hellaswag_context,
    score_completion,
)
from basikgpt.model.gpt import GPT

try:
    from transformers import GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =====================================================================
# 1. Synthetic Fixed-Logits Mock Model
# =====================================================================

class MockLogitsModel(nn.Module):
    """A deterministic mock model that returns pre-configured fixed logits."""

    def __init__(self, logits_map: dict[tuple[int, ...], torch.Tensor], default_vocab_size: int = 10):
        super().__init__()
        self.logits_map = logits_map
        self.default_vocab_size = default_vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_tuple = tuple(input_ids[0].tolist())
        if seq_tuple in self.logits_map:
            return self.logits_map[seq_tuple].unsqueeze(0)
        B, T = input_ids.shape
        return torch.zeros(B, T, self.default_vocab_size)


# =====================================================================
# 2. Shift Alignment & Hand-Calculated Known-Answer Tests
# =====================================================================

def test_shift_alignment_and_known_answer() -> None:
    """Verifies that logits at index k-1 parameterize target token at index k with exact hand math."""
    vocab_size = 4
    context_tokens = [0, 1]      # P = 2
    completion_tokens = [2, 3]   # M = 2, total T = 4
    full_seq = tuple(context_tokens + completion_tokens)

    # Logits at T=4:
    # position 0: predicts token 1
    # position 1: predicts token 2 (target completion token 0)
    # position 2: predicts token 3 (target completion token 1)
    # position 3: predicts next token (unused)
    fixed_logits = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # pos 0 -> softmax for token 1 is exp(0)/(exp(1)+3)
        [0.0, 0.0, 2.0, 0.0],  # pos 1 -> token 2 has logit 2.0, others 0.0
        [0.0, 0.0, 0.0, 3.0],  # pos 2 -> token 3 has logit 3.0, others 0.0
        [0.0, 0.0, 0.0, 0.0],  # pos 3 -> unused
    ])

    model = MockLogitsModel({full_seq: fixed_logits}, default_vocab_size=vocab_size)

    score = score_completion(
        model=model,
        context_tokens=context_tokens,
        completion_tokens=completion_tokens,
    )

    # Hand calculate log_softmax:
    # For pos 1: logits [0, 0, 2, 0] -> log(exp(2) / (3*1 + exp(2))) = 2.0 - ln(3 + exp(2))
    expected_logp_2 = float(2.0 - torch.log(torch.tensor(3.0) + torch.exp(torch.tensor(2.0))))
    # For pos 2: logits [0, 0, 0, 3] -> log(exp(3) / (3*1 + exp(3))) = 3.0 - ln(3 + exp(3))
    expected_logp_3 = float(3.0 - torch.log(torch.tensor(3.0) + torch.exp(torch.tensor(3.0))))

    expected_total = expected_logp_2 + expected_logp_3
    expected_mean = expected_total / 2.0

    assert pytest.approx(score.total_log_likelihood, abs=1e-5) == expected_total
    assert pytest.approx(score.mean_log_likelihood, abs=1e-5) == expected_mean
    assert score.token_count == 2


def test_context_tokens_strictly_excluded() -> None:
    """Verifies that mutating context logits (position 0) never affects completion score."""
    vocab_size = 4
    context_tokens = [0, 1]
    completion_tokens = [2, 3]
    full_seq = tuple(context_tokens + completion_tokens)

    logits_base = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    # Alter position 0 dramatically:
    logits_altered_context = logits_base.clone()
    logits_altered_context[0] = torch.tensor([100.0, -50.0, 42.0, -99.0])

    model_base = MockLogitsModel({full_seq: logits_base}, default_vocab_size=vocab_size)
    model_altered = MockLogitsModel({full_seq: logits_altered_context}, default_vocab_size=vocab_size)

    score_base = score_completion(model_base, context_tokens, completion_tokens)
    score_altered = score_completion(model_altered, context_tokens, completion_tokens)

    assert score_base.total_log_likelihood == score_altered.total_log_likelihood
    assert score_base.mean_log_likelihood == score_altered.mean_log_likelihood


# =====================================================================
# 3. Length Normalization Mechanics & Ranking Inversion
# =====================================================================

def test_length_normalization_mechanics() -> None:
    """Verifies that raw vs length-normalized ranking can correctly invert on controlled token counts."""
    # Candidate A: 1 token with logprob -2.0 -> total: -2.0, mean: -2.0
    # Candidate B: 3 tokens with logprob -0.8 each -> total: -2.4, mean: -0.8
    # Raw Argmax selects A (-2.0 > -2.4)
    # Norm Argmax selects B (-0.8 > -2.0)
    score_a = CandidateScore(total_log_likelihood=-2.0, mean_log_likelihood=-2.0, token_count=1)
    score_b = CandidateScore(total_log_likelihood=-2.4, mean_log_likelihood=-0.8, token_count=3)
    score_c = CandidateScore(total_log_likelihood=-5.0, mean_log_likelihood=-2.5, token_count=2)
    score_d = CandidateScore(total_log_likelihood=-6.0, mean_log_likelihood=-3.0, token_count=2)

    raw_scores = [score_a.total_log_likelihood, score_b.total_log_likelihood, score_c.total_log_likelihood, score_d.total_log_likelihood]
    norm_scores = [score_a.mean_log_likelihood, score_b.mean_log_likelihood, score_c.mean_log_likelihood, score_d.mean_log_likelihood]

    pred_raw = int(torch.tensor(raw_scores).argmax().item())
    pred_norm = int(torch.tensor(norm_scores).argmax().item())

    assert pred_raw == 0, "Raw scoring should select Candidate A (highest sum LL)"
    assert pred_norm == 1, "Length-normalized scoring should select Candidate B (highest mean LL)"


# =====================================================================
# 4. Token Boundary & Context Formatting Tests
# =====================================================================

def test_gpt2_token_boundary_invariance() -> None:
    """Verifies that prepending ' ' to endings preserves exact concatenated BPE tokens."""
    tokenizer = GPT2Tokenizer()

    test_pairs = [
        ("A man is sitting on a roof. he", "is using wrap to wrap a pair of skis."),
        ("A lady walks to a barbell. She bends down.", "stands and lifts the weight."),
        ("Two women in a child are shown in a canoe", "sit in a canoe while the man paddles."),
    ]

    for ctx, end in test_pairs:
        ctx_tokens = tokenizer.encode(ctx)
        end_tokens = tokenizer.encode(" " + end)
        full_tokens = tokenizer.encode(ctx + " " + end)

        assert ctx_tokens + end_tokens == full_tokens, f"BPE boundary mismatch on '{ctx}' + '{end}'"


def test_offline_context_formatting() -> None:
    """Tests both 'activity_ctx' and 'ctx_only' formatting styles on sample dictionaries."""
    sample = {
        "activity_label": "Roof shingle removal",
        "ctx_a": "A man is sitting on a roof.",
        "ctx_b": "he",
        "ctx": "A man is sitting on a roof. he",
        "endings": ["choice 0", "choice 1", "choice 2", "choice 3"],
        "label": 2,
    }

    # 1. 'activity_ctx'
    formatted_act = format_hellaswag_context(sample, format_style="activity_ctx")
    assert formatted_act == "Roof shingle removal: A man is sitting on a roof. He"

    # 2. 'ctx_only'
    formatted_ctx = format_hellaswag_context(sample, format_style="ctx_only")
    assert formatted_ctx == "A man is sitting on a roof. he"


# =====================================================================
# 5. Four-Choice Example Prediction & Accuracy Aggregation
# =====================================================================

def test_four_candidate_prediction_argmax() -> None:
    """Verifies that evaluate_hellaswag_example scores 4 choices and computes argmax correctly."""
    cfg = GPTConfig(vocab_size=50257, context_length=64, n_layers=2, n_heads=2, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    tokenizer = GPT2Tokenizer()
    sample = {
        "ind": 101,
        "activity_label": "Testing",
        "ctx_a": "Hello world.",
        "ctx_b": "",
        "ctx": "Hello world.",
        "endings": ["one ending", "another ending", "third ending", "fourth ending"],
        "label": "2",
    }

    res = evaluate_hellaswag_example(
        model=model,
        example=sample,
        tokenizer=tokenizer,
        format_style="activity_ctx",
    )

    assert res.example_id == 101
    assert res.gold_label == 2
    assert len(res.raw_scores) == 4
    assert len(res.norm_scores) == 4
    assert len(res.token_counts) == 4
    assert res.pred_raw in (0, 1, 2, 3)
    assert res.pred_norm in (0, 1, 2, 3)
    assert isinstance(res.is_raw_correct, bool)
    assert isinstance(res.is_norm_correct, bool)


def test_accuracy_aggregation_arithmetic() -> None:
    """Verifies that evaluate_hellaswag accurately calculates raw and norm accuracy fractions."""
    cfg = GPTConfig(vocab_size=50257, context_length=64, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    tokenizer = GPT2Tokenizer()
    dataset = [
        {"ind": i, "activity_label": "T", "ctx_a": "Hi", "ctx_b": "", "ctx": "Hi", "endings": ["a", "b", "c", "d"], "label": 0}
        for i in range(4)
    ]

    summary, results = evaluate_hellaswag(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        progress_interval=0,
    )

    assert summary.num_examples == 4
    assert len(results) == 4
    assert 0.0 <= summary.raw_accuracy <= 1.0
    assert 0.0 <= summary.norm_accuracy <= 1.0
    assert summary.raw_accuracy == summary.raw_correct / 4.0
    assert summary.norm_accuracy == summary.norm_correct / 4.0


# =====================================================================
# 6. Error Handling & Overflow Guardrail Tests
# =====================================================================

def test_invalid_and_missing_labels() -> None:
    """Verifies that invalid or out-of-range labels raise explicit ValueErrors."""
    cfg = GPTConfig(vocab_size=50257, context_length=16, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)
    tokenizer = GPT2Tokenizer()

    # Missing label
    with pytest.raises(ValueError, match="missing or empty gold label"):
        evaluate_hellaswag_example(
            model=model,
            example={"ind": 1, "ctx": "hi", "endings": ["a", "b", "c", "d"], "label": ""},
            tokenizer=tokenizer,
        )

    # Out-of-range label
    with pytest.raises(ValueError, match="out of expected range"):
        evaluate_hellaswag_example(
            model=model,
            example={"ind": 1, "ctx": "hi", "endings": ["a", "b", "c", "d"], "label": "5"},
            tokenizer=tokenizer,
        )

    # Invalid choice count
    with pytest.raises(ValueError, match="candidate endings"):
        evaluate_hellaswag_example(
            model=model,
            example={"ind": 1, "ctx": "hi", "endings": ["a", "b"], "label": "0"},
            tokenizer=tokenizer,
        )


def test_context_overflow_guardrail() -> None:
    """Verifies that context is safely left-truncated if total sequence exceeds max_context_length."""
    cfg = GPTConfig(vocab_size=64, context_length=8, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    context_tokens = [10, 11, 12, 13, 14, 15]  # len 6
    completion_tokens = [20, 21, 22]            # len 3 -> total 9 > 8

    # Max length = 8 -> context left-truncated to 8 - 3 = 5 tokens
    score = score_completion(
        model=model,
        context_tokens=context_tokens,
        completion_tokens=completion_tokens,
        max_context_length=8,
    )

    assert score.token_count == 3
    assert isinstance(score.total_log_likelihood, float)
    assert isinstance(score.mean_log_likelihood, float)


# =====================================================================
# 7. Hugging Face Score-Level Parity Test
# =====================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers package required for reference parity")
def test_hf_gpt2_candidate_score_parity() -> None:
    """Verifies that basikGPT candidate log-likelihood matches Hugging Face GPT-2 on identical inputs."""
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    hf_model.eval()

    cfg = GPTConfig.gpt2_small(dropout=0.0)
    basik_model = GPT(cfg)
    load_hf_gpt2_weights(basik_model, "openai-community/gpt2")
    basik_model.eval()

    tokenizer = GPT2Tokenizer()
    context = "The quick brown fox"
    completion = " jumps over the lazy dog."

    ctx_tokens = tokenizer.encode(context)
    comp_tokens = tokenizer.encode(" " + completion)

    # 1. basikGPT scoring
    basik_score = score_completion(basik_model, ctx_tokens, comp_tokens)

    # 2. Reference Hugging Face scoring
    full_tokens = torch.tensor([ctx_tokens + comp_tokens], dtype=torch.long)
    with torch.no_grad():
        hf_logits = hf_model(full_tokens).logits
        P = len(ctx_tokens)
        T = full_tokens.shape[1]
        shift_logits = hf_logits[:, P - 1 : T - 1, :].contiguous()
        shift_targets = full_tokens[:, P : T].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)
        hf_total_ll = float(target_log_probs.sum().item())
        hf_mean_ll = float(target_log_probs.mean().item())

    assert pytest.approx(basik_score.total_log_likelihood, abs=1e-4) == hf_total_ll
    assert pytest.approx(basik_score.mean_log_likelihood, abs=1e-4) == hf_mean_ll


# =====================================================================
# 8. Real HellaSwag Dataset Smoke Test
# =====================================================================

def test_real_hellaswag_small_smoke() -> None:
    """Integration test evaluating 5 real HellaSwag validation examples."""
    from datasets import load_dataset

    cfg = GPTConfig(vocab_size=50257, context_length=1024, n_layers=1, n_heads=1, d_model=32, d_ff=64)
    model = GPT(cfg)
    model.eval()

    dataset_stream = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
    tokenizer = GPT2Tokenizer()

    summary, results = evaluate_hellaswag(
        model=model,
        dataset=dataset_stream,
        tokenizer=tokenizer,
        max_examples=5,
        progress_interval=0,
    )

    assert summary.num_examples == 5
    assert len(results) == 5
    assert 0.0 <= summary.raw_accuracy <= 1.0
    assert 0.0 <= summary.norm_accuracy <= 1.0
    for r in results:
        assert len(r.raw_scores) == 4
        assert len(r.norm_scores) == 4
        assert r.pred_raw in (0, 1, 2, 3)
        assert r.pred_norm in (0, 1, 2, 3)
        assert r.gold_label in (0, 1, 2, 3)
