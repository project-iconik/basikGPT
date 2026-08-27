"""Unit tests for GenerationConfig, sampling filters (temperature, top-k, top-p), and autoregressive generation."""

import pytest
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.generate import generate
from basikgpt.generation.sampling import sample_next_token
from basikgpt.model.gpt import GPT


# =====================================================================
# 1. GenerationConfig Validation
# =====================================================================

def test_generation_config_validation() -> None:
    """Verifies that GenerationConfig enforces valid hyperparameter bounds."""
    # Valid defaults
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 50
    assert cfg.temperature == 1.0
    assert not cfg.do_sample

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        GenerationConfig(max_new_tokens=0)

    with pytest.raises(ValueError, match="temperature must be strictly positive"):
        GenerationConfig(temperature=0.0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        GenerationConfig(top_k=-5)

    with pytest.raises(ValueError, match="top_p must be in the range"):
        GenerationConfig(top_p=0.0)

    with pytest.raises(ValueError, match="top_p must be in the range"):
        GenerationConfig(top_p=1.5)

    with pytest.raises(ValueError, match="eot_token_id must be non-negative"):
        GenerationConfig(eot_token_id=-1)


# =====================================================================
# 2. Sampling Filter Unit Tests
# =====================================================================

def test_sample_next_token_greedy_argmax() -> None:
    """Verifies that greedy decoding (do_sample=False) strictly selects the maximum logit token."""
    # Batch size 2, Vocab size 5
    logits = torch.tensor([
        [1.0, 5.0, 2.0, 0.0, 3.0],   # max is index 1
        [-2.0, -1.0, 4.5, 4.4, 0.0], # max is index 2
    ])
    cfg = GenerationConfig(do_sample=False)
    tokens = sample_next_token(logits, cfg)

    assert tokens.shape == (2, 1)
    assert tokens[0, 0].item() == 1
    assert tokens[1, 0].item() == 2


def test_sample_next_token_temperature_sharpness() -> None:
    """Verifies that low temperature concentrates probability mass on the maximum token."""
    logits = torch.tensor([[10.0, 12.0, 8.0]])
    gen = torch.Generator().manual_seed(42)

    # Very low temperature -> nearly deterministic on max index (1)
    cfg = GenerationConfig(do_sample=True, temperature=0.01, seed=42)
    selected = [sample_next_token(logits, cfg, generator=gen).item() for _ in range(20)]
    assert all(idx == 1 for idx in selected), "Low temperature must sample the peak logit token almost 100% of the time"


def test_sample_next_token_top_k() -> None:
    """Verifies that top_k restricts sampling exclusively to the top k tokens."""
    # Vocab size 5: top 2 tokens are indices 3 and 4
    logits = torch.tensor([[1.0, 2.0, 3.0, 10.0, 10.5]])
    cfg = GenerationConfig(do_sample=True, top_k=2, temperature=1.0)
    gen = torch.Generator().manual_seed(123)

    sampled_indices = {sample_next_token(logits, cfg, generator=gen).item() for _ in range(50)}
    assert sampled_indices.issubset({3, 4}), f"Sampled indices {sampled_indices} contained tokens outside top_k=2"


def test_sample_next_token_top_p_nucleus() -> None:
    """Verifies that nucleus sampling keeps only the smallest set with cumulative probability <= p."""
    # Distinct probabilities: index 0 (0.7), index 1 (0.2), index 2 (0.08), index 3 (0.02)
    logits = torch.tensor([[10.0, 8.75, 7.83, 6.44]])
    cfg = GenerationConfig(do_sample=True, top_p=0.8, temperature=1.0)
    gen = torch.Generator().manual_seed(42)

    sampled = {sample_next_token(logits, cfg, generator=gen).item() for _ in range(50)}
    # With top_p = 0.8, index 0 (~0.7) and index 1 (~0.2) reach > 0.8 cumsum, so only {0, 1} are eligible
    assert sampled.issubset({0, 1}), f"Sampled indices {sampled} exceeded top_p nucleus"


def test_sampling_seed_reproducibility() -> None:
    """Verifies that specifying a random seed yields identical sampling sequences."""
    logits = torch.randn((1, 50))
    cfg = GenerationConfig(do_sample=True, temperature=1.2, top_k=20, top_p=0.9, seed=999)

    gen1 = torch.Generator().manual_seed(999)
    seq1 = [sample_next_token(logits, cfg, generator=gen1).item() for _ in range(10)]

    gen2 = torch.Generator().manual_seed(999)
    seq2 = [sample_next_token(logits, cfg, generator=gen2).item() for _ in range(10)]

    assert seq1 == seq2


# =====================================================================
# 3. Autoregressive Generation Loop Tests
# =====================================================================

class MockPredictableGPT(nn.Module):
    """Mock model where next token prediction is always current_last_token + 1 (mod vocab_size)."""

    def __init__(self, vocab_size: int = 32, context_length: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = GPTConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            n_layers=1,
            n_heads=1,
            d_model=16,
            d_ff=32,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        logits = torch.zeros((B, T, self.vocab_size), dtype=torch.float32)
        # Give highest logit to (last_token + 1) % vocab_size
        for b in range(B):
            next_idx = (input_ids[b, -1].item() + 1) % self.vocab_size
            logits[b, -1, next_idx] = 100.0
        return logits


def test_generate_greedy_end_to_end() -> None:
    """Verifies that generate accurately appends predictable greedy argmax tokens."""
    model = MockPredictableGPT(vocab_size=32, context_length=16)
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)

    cfg = GenerationConfig(max_new_tokens=4, do_sample=False)
    out = generate(model, input_ids, cfg)

    # Initial [10, 11] -> should append 12, 13, 14, 15
    expected = [10, 11, 12, 13, 14, 15]
    assert out.squeeze(0).tolist() == expected


def test_generate_stop_on_eot() -> None:
    """Verifies that generation halts immediately when the canonical EOT token is emitted."""
    eot_id = 99
    # Mock model that emits EOT on the 2nd step
    class MockEOTModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = GPTConfig(vocab_size=100, context_length=16, n_layers=1, n_heads=1, d_model=16, d_ff=32)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            B, T = input_ids.shape
            logits = torch.zeros((B, T, 100))
            if T == 2:
                logits[:, -1, 5] = 10.0
            else:
                logits[:, -1, eot_id] = 10.0  # Emits EOT on step 2
            return logits

    model = MockEOTModel()
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=10, stop_on_eot=True, eot_token_id=eot_id, do_sample=False)

    out = generate(model, input_ids, cfg)
    # Prompt is 2 tokens, step 1 appends 5 (length 3), step 2 appends 99 (length 4) and stops
    assert out.shape == (1, 4)
    assert out[0, -1].item() == eot_id


def test_generate_context_length_boundary_and_overflow() -> None:
    """Verifies context length guardrails and boundary clipping behavior."""
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)

    # 1. Overflow prompt (length 9 > context 8) raises ValueError
    overflow_prompt = torch.randint(0, 32, (1, 9))
    with pytest.raises(ValueError, match="exceeds model maximum context_length"):
        generate(model, overflow_prompt)

    # 2. Prompt equal to context length (length 8 == context 8) returns prompt directly without error
    full_prompt = torch.randint(0, 32, (1, 8))
    out = generate(model, full_prompt, GenerationConfig(max_new_tokens=5))
    assert out.shape == (1, 8)
    assert torch.equal(out, full_prompt)

    # 3. Prompt of length 6 with max_new_tokens=5 generates up to context limit (2 new tokens -> length 8)
    prompt_6 = torch.randint(0, 32, (1, 6))
    out_6 = generate(model, prompt_6, GenerationConfig(max_new_tokens=5))
    assert out_6.shape == (1, 8)
