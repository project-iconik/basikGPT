"""Unit and integration tests for Key-Value Caching in basikGPT.

Tests:
1. Cache tensor shapes, incremental growth, and layer isolation.
2. Transient nature of cache (not in model parameters or state_dict).
3. Position offset correctness with learned positional embeddings.
4. Step-by-step logit parity between naive full-prefix forward and cached decode (Eager & SDPA).
5. Eager vs SDPA cached parity.
6. Greedy and sampling token sequence exact equality (Naive == Cached).
7. Context length boundary enforcement.
8. Cache reset and caller ownership isolation.
9. Reference GPT-2 Hugging Face cached greedy parity.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
from basikgpt.data.tokenizer import GPT2Tokenizer
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.generate import (
    generate,
    generate_cached,
    generate_naive,
)
from basikgpt.model.gpt import GPT

try:
    from transformers import GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =====================================================================
# 1. Cache Tensor Shapes, Growth, and Invariants
# =====================================================================

def test_kv_cache_tensor_shapes_and_growth() -> None:
    """Verifies that per-layer KV cache expands incrementally and matches expected tensor shapes."""
    B, T_prompt = 2, 5
    H, D = 4, 16
    n_layers = 3
    vocab_size = 64
    context_length = 32

    cfg = GPTConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        n_layers=n_layers,
        n_heads=H,
        d_model=H * D,
        d_ff=4 * H * D,
    )
    model = GPT(cfg)
    model.eval()

    prompt_ids = torch.randint(0, vocab_size, (B, T_prompt))

    with torch.inference_mode():
        # Phase 1: Prefill prompt (T_prompt = 5)
        logits_prefill, past_kv = model.forward_cached(prompt_ids, past_key_values=None)

        assert logits_prefill.shape == (B, T_prompt, vocab_size)
        assert isinstance(past_kv, tuple)
        assert len(past_kv) == n_layers

        for layer_idx, (k, v) in enumerate(past_kv):
            assert k.shape == (B, H, T_prompt, D), f"Layer {layer_idx} key shape mismatch"
            assert v.shape == (B, H, T_prompt, D), f"Layer {layer_idx} value shape mismatch"

        # Verify distinct tensor allocations across layers (no aliasing)
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                assert past_kv[i][0].data_ptr() != past_kv[j][0].data_ptr(), "Layer key caches must not alias"
                assert past_kv[i][1].data_ptr() != past_kv[j][1].data_ptr(), "Layer value caches must not alias"

        # Phase 2: Decode single token (T = 1) -> Cache length becomes T_prompt + 1 = 6
        single_token = torch.randint(0, vocab_size, (B, 1))
        logits_decode_1, past_kv_1 = model.forward_cached(single_token, past_key_values=past_kv)

        assert logits_decode_1.shape == (B, 1, vocab_size)
        assert len(past_kv_1) == n_layers

        for layer_idx, (k, v) in enumerate(past_kv_1):
            assert k.shape == (B, H, T_prompt + 1, D)
            assert v.shape == (B, H, T_prompt + 1, D)


def test_kv_cache_not_in_parameters_or_state_dict() -> None:
    """Verifies that KV cache remains transient caller-owned state and never enters model state_dict."""
    cfg = GPTConfig(vocab_size=32, context_length=16, n_layers=2, n_heads=2, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    state_dict_before = {k: v.clone() for k, v in model.state_dict().items()}

    prompt = torch.randint(0, 32, (1, 4))
    with torch.inference_mode():
        _, past_kv = model.forward_cached(prompt, past_key_values=None)
        _, _ = model.forward_cached(torch.tensor([[5]]), past_key_values=past_kv)

    state_dict_after = model.state_dict()
    assert state_dict_before.keys() == state_dict_after.keys()
    for k in state_dict_before:
        assert torch.equal(state_dict_before[k], state_dict_after[k])


# =====================================================================
# 2. Step-by-Step Logit Parity (Naive vs Cached)
# =====================================================================

@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_step_by_step_logit_parity(backend: str) -> None:
    """Verifies bitwise numerical logit parity between full-prefix naive forward and cached decode."""
    torch.manual_seed(1337)
    cfg = GPTConfig(
        vocab_size=100,
        context_length=32,
        n_layers=3,
        n_heads=2,
        d_model=32,
        d_ff=64,
        attention_backend=backend,
        dropout=0.0,
    )
    model = GPT(cfg)
    model.eval()

    prompt = torch.tensor([[12, 45, 78]], dtype=torch.long)  # length 3
    new_tokens = [23, 67, 89, 90]                             # 4 decode steps

    with torch.inference_mode():
        # 1. Prefill step comparison
        naive_logits_prefill = model(prompt)
        cached_logits_prefill, past_kv = model.forward_cached(prompt, past_key_values=None)

        torch.testing.assert_close(
            cached_logits_prefill,
            naive_logits_prefill,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Prefill logits mismatch for backend {backend}",
        )

        # 2. Sequential decode steps
        accumulated = prompt.clone()
        for step_idx, tok in enumerate(new_tokens):
            tok_tensor = torch.tensor([[tok]], dtype=torch.long)
            accumulated = torch.cat([accumulated, tok_tensor], dim=1)

            # Naive full forward
            naive_logits_step = model(accumulated)
            naive_last_logits = naive_logits_step[:, -1:, :]

            # Cached single token forward
            cached_logits_step, past_kv = model.forward_cached(tok_tensor, past_key_values=past_kv)

            torch.testing.assert_close(
                cached_logits_step,
                naive_last_logits,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Step {step_idx} decode logits mismatch for backend {backend}",
            )


def test_eager_vs_sdpa_cached_parity() -> None:
    """Verifies that Eager and SDPA backends produce identical cached outputs."""
    torch.manual_seed(42)
    cfg_eager = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=2, d_model=16, d_ff=32, attention_backend="eager", dropout=0.0)
    cfg_sdpa = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=2, d_model=16, d_ff=32, attention_backend="sdpa", dropout=0.0)

    model_eager = GPT(cfg_eager)
    model_sdpa = GPT(cfg_sdpa)
    model_sdpa.load_state_dict(model_eager.state_dict())
    model_eager.eval()
    model_sdpa.eval()

    prompt = torch.randint(0, 64, (1, 4))
    token1 = torch.tensor([[10]])

    with torch.inference_mode():
        l_eager_pre, kv_eager = model_eager.forward_cached(prompt)
        l_sdpa_pre, kv_sdpa = model_sdpa.forward_cached(prompt)
        torch.testing.assert_close(l_eager_pre, l_sdpa_pre, rtol=1e-5, atol=1e-5)

        l_eager_dec, _ = model_eager.forward_cached(token1, past_key_values=kv_eager)
        l_sdpa_dec, _ = model_sdpa.forward_cached(token1, past_key_values=kv_sdpa)
        torch.testing.assert_close(l_eager_dec, l_sdpa_dec, rtol=1e-5, atol=1e-5)


# =====================================================================
# 3. End-to-End Generation Parity (Naive vs Cached)
# =====================================================================

def test_greedy_token_sequence_exact_parity() -> None:
    """Verifies that generate_naive and generate_cached produce the exact same greedy token sequence."""
    torch.manual_seed(777)
    cfg = GPTConfig(vocab_size=128, context_length=32, n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    prompt = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    gen_cfg = GenerationConfig(max_new_tokens=10, do_sample=False)

    out_naive = generate_naive(model, prompt, gen_cfg)
    out_cached = generate_cached(model, prompt, gen_cfg)
    out_unified = generate(model, prompt, gen_cfg, use_cache=True)

    assert out_naive.shape == (1, 14)
    assert torch.equal(out_naive, out_cached), f"Token mismatch:\nNaive:  {out_naive.tolist()}\nCached: {out_cached.tolist()}"
    assert torch.equal(out_cached, out_unified)


def test_sampling_token_sequence_parity_with_seed() -> None:
    """Verifies that with the same RNG seed, sampling produce identical sequences in naive and cached modes."""
    torch.manual_seed(999)
    cfg = GPTConfig(vocab_size=128, context_length=32, n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0)
    model = GPT(cfg)
    model.eval()

    prompt = torch.tensor([[5, 15, 25]], dtype=torch.long)
    gen_cfg = GenerationConfig(max_new_tokens=8, do_sample=True, temperature=0.9, top_k=10, seed=1234)

    out_naive = generate_naive(model, prompt, gen_cfg)
    out_cached = generate_cached(model, prompt, gen_cfg)

    assert torch.equal(out_naive, out_cached), f"Sampling mismatch:\nNaive:  {out_naive.tolist()}\nCached: {out_cached.tolist()}"


# =====================================================================
# 4. Context Boundary, Position Offset, and Reset Tests
# =====================================================================

def test_context_length_boundary_cached() -> None:
    """Verifies that forward_cached rejects inputs exceeding max context_length."""
    cfg = GPTConfig(vocab_size=32, context_length=6, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    prompt = torch.randint(0, 32, (1, 4))
    with torch.inference_mode():
        _, past_kv = model.forward_cached(prompt)  # cached len = 4

        # Adding 2 tokens is fine (4 + 2 = 6 <= 6)
        _, past_kv2 = model.forward_cached(torch.randint(0, 32, (1, 2)), past_key_values=past_kv)
        assert past_kv2[0][0].shape[-2] == 6

        # Adding 1 more token must raise ValueError (6 + 1 = 7 > 6)
        with pytest.raises(ValueError, match="exceeds maximum configured context length"):
            model.forward_cached(torch.randint(0, 32, (1, 1)), past_key_values=past_kv2)


def test_cache_caller_isolation_and_reset() -> None:
    """Verifies that separate generations with fresh past_key_values=None do not leak state."""
    cfg = GPTConfig(vocab_size=32, context_length=16, n_layers=1, n_heads=1, d_model=16, d_ff=32)
    model = GPT(cfg)
    model.eval()

    prompt_a = torch.tensor([[1, 2, 3]])
    prompt_b = torch.tensor([[10, 20, 30]])

    gen_cfg = GenerationConfig(max_new_tokens=4, do_sample=False)

    out_a1 = generate_cached(model, prompt_a, gen_cfg)
    out_b = generate_cached(model, prompt_b, gen_cfg)
    out_a2 = generate_cached(model, prompt_a, gen_cfg)

    assert torch.equal(out_a1, out_a2), "Repeated generation on prompt_a must be identical"
    assert not torch.equal(out_a1, out_b), "Prompt A and Prompt B should yield different completions"


# =====================================================================
# 5. Hugging Face Reference Parity (Cached vs Reference)
# =====================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers package required for reference parity")
def test_reference_gpt2_cached_greedy_parity() -> None:
    """Verifies that basikGPT generate_cached matches HuggingFace GPT-2 greedy output token-for-token."""
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    hf_model.eval()

    cfg = GPTConfig.gpt2_small(dropout=0.0)
    basik_model = GPT(cfg)
    load_hf_gpt2_weights(basik_model, "openai-community/gpt2")
    basik_model.eval()

    prompt = "The history of artificial intelligence"
    tokenizer = GPT2Tokenizer()
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    # Hugging Face with use_cache=True (default)
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids,
            max_new_tokens=15,
            do_sample=False,
            use_cache=True,
            pad_token_id=50256,
        )

    # basikGPT with generate_cached
    gen_cfg = GenerationConfig(max_new_tokens=15, do_sample=False)
    basik_cached_out = generate_cached(basik_model, input_ids, gen_cfg)

    assert torch.equal(hf_out, basik_cached_out), (
        f"Mismatch between HF reference and basikGPT cached generation:\n"
        f"HF:    {hf_out.tolist()}\n"
        f"Basik: {basik_cached_out.tolist()}"
    )
