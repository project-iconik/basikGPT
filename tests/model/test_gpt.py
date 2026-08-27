"""Unit and integration tests for the full GPT model in basikGPT."""

from typing import Literal
import pytest
import torch

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.model.gpt import GPT

BACKENDS: list[AttentionBackend] = ["eager", "sdpa"]


def make_config(
    backend: AttentionBackend = "eager",
    dropout: float = 0.0,
    n_layers: int = 2,
    bias: bool = True,
) -> GPTConfig:
    """Helper to create a small test GPTConfig."""
    return GPTConfig(
        vocab_size=128,
        context_length=32,
        n_layers=n_layers,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=dropout,
        bias=bias,
        attention_backend=backend,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpt_forward_output_shape(backend: AttentionBackend) -> None:
    """Verifies that full GPT forward produces logits of shape (B, T, V)."""
    config = make_config(backend=backend)
    model = GPT(config)

    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))

    logits = model(input_ids)

    assert logits.shape == (B, T, config.vocab_size), (
        f"Expected logits shape {(B, T, config.vocab_size)}, got {logits.shape}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seq_len", [1, 4, 16, 32])
def test_gpt_different_sequence_lengths(backend: AttentionBackend, seq_len: int) -> None:
    """Verifies full GPT model forward across various sequence lengths T <= context_length."""
    config = make_config(backend=backend)
    model = GPT(config)

    B = 2
    input_ids = torch.randint(0, config.vocab_size, (B, seq_len))

    logits = model(input_ids)

    assert logits.shape == (B, seq_len, config.vocab_size)


def test_gpt_input_validation() -> None:
    """Verifies input rank validation and sequence length overflow errors."""
    config = make_config()
    model = GPT(config)

    # Test 1: Non-2D inputs (1D or 3D)
    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        model(torch.tensor([1, 2, 3]))

    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        model(torch.randint(0, config.vocab_size, (2, 4, 1)))

    # Test 2: Sequence length exceeds context_length
    overflow_T = config.context_length + 1
    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        model(torch.randint(0, config.vocab_size, (2, overflow_T)))


def test_gpt_component_parameter_shapes() -> None:
    """Verifies weight matrix shapes of wte, wpe, lm_head and block count."""
    config = make_config(n_layers=3)
    model = GPT(config)

    V = config.vocab_size
    T_max = config.context_length
    C = config.d_model

    # Token Embedding: (V, C)
    assert model.wte.weight.shape == (V, C)

    # Positional Embedding: (context_length, C)
    assert model.wpe.weight.shape == (T_max, C)

    # LM Head: (V, C) with bias=False
    assert model.lm_head.weight.shape == (V, C)
    assert model.lm_head.bias is None

    # Transformer Blocks count
    assert len(model.blocks) == 3


def test_gpt_independent_block_instances() -> None:
    """Verifies that each TransformerBlock has distinct, independent parameters (no weight sharing)."""
    config = make_config(n_layers=2)
    model = GPT(config)

    block0 = model.blocks[0]
    block1 = model.blocks[1]

    assert block0 is not block1
    assert block0.attn.qkv_proj.weight is not block1.attn.qkv_proj.weight
    assert block0.mlp.fc_in.weight is not block1.mlp.fc_in.weight


def test_gpt_weight_tying_identity() -> None:
    """Verifies that lm_head.weight and wte.weight share the exact same Parameter object and memory."""
    config = make_config()
    model = GPT(config)

    # 1. Exact Python object identity
    assert model.lm_head.weight is model.wte.weight

    # 2. Underlying storage pointer identity
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()


def test_gpt_weight_tying_gradient_accumulation() -> None:
    """Verifies that backpropagation accumulates gradients into the shared tied weight."""
    config = make_config(dropout=0.0)
    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # The tied weight must have accumulated gradients
    assert model.wte.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert model.wte.weight.grad is model.lm_head.weight.grad
    assert model.wte.weight.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpt_causal_leakage_end_to_end(backend: AttentionBackend) -> None:
    """Verifies end-to-end causality: future token ID changes strictly do NOT alter past token logits."""
    config = make_config(backend=backend, dropout=0.0)
    model = GPT(config)
    model.eval()

    # Sequence A: [10, 20, 30, 40]
    input_a = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

    # Sequence B: [10, 20, 99, 88] (same prefix at t=0, 1; different future at t=2, 3)
    input_b = torch.tensor([[10, 20, 99, 88]], dtype=torch.long)

    with torch.no_grad():
        logits_a = model(input_a)
        logits_b = model(input_b)

    # 1. Past logits (t = 0, 1) MUST be identical
    torch.testing.assert_close(
        logits_a[:, :2, :],
        logits_b[:, :2, :],
        rtol=1e-5,
        atol=1e-5,
        msg=f"End-to-end causal leakage detected in backend '{backend}'.",
    )

    # 2. Future logits (t = 2, 3) MUST differ
    diff = (logits_a[:, 2:, :] - logits_b[:, 2:, :]).abs().max().item()
    assert diff > 1e-3, "Future logits should differ when future token IDs are altered."


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpt_gradient_flow_end_to_end(backend: AttentionBackend) -> None:
    """Verifies complete gradient propagation across the entire model."""
    config = make_config(backend=backend, dropout=0.0, n_layers=2)
    model = GPT(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Embedding gradients
    assert model.wte.weight.grad is not None and not torch.isnan(model.wte.weight.grad).any()
    assert model.wpe.weight.grad is not None and not torch.isnan(model.wpe.weight.grad).any()

    # First and last block gradients
    assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert model.blocks[-1].mlp.fc_out.weight.grad is not None

    # Final LayerNorm gradients
    assert model.ln_f.weight.grad is not None
    assert model.ln_f.bias.grad is not None


def test_gpt_eval_mode_determinism() -> None:
    """Verifies that full GPT model in eval mode is completely deterministic."""
    config = make_config(dropout=0.5)
    model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)

    torch.testing.assert_close(logits1, logits2)


def test_gpt2_small_exact_parameter_count() -> None:
    """Verifies the exact unique parameter count of full GPT-2 Small (124,439,808 parameters)."""
    config = GPTConfig.gpt2_small()
    model = GPT(config)

    # 1. Total unique parameters in model instance (with weight tying)
    actual_unique = model.num_parameters()
    expected_unique = 124_439_808
    assert actual_unique == expected_unique
    assert actual_unique == config.num_total_parameters(tied_weights=True)

    # 2. Non-embedding parameters (85,056,000)
    actual_non_emb = model.num_parameters(non_embedding=True)
    expected_non_emb = 85_056_000
    assert actual_non_emb == expected_non_emb
    assert actual_non_emb == config.num_transformer_parameters()

    # 3. Analytical breakdown verification
    assert config.num_embedding_parameters() == 39_383_808
    assert config.num_total_parameters(tied_weights=False) == 163_037_184


def test_gpt_eager_sdpa_full_model_parity() -> None:
    """Verifies full model logits parity between Eager and SDPA backends."""
    cfg_eager = make_config(backend="eager", dropout=0.0, n_layers=2)
    cfg_sdpa = make_config(backend="sdpa", dropout=0.0, n_layers=2)

    model_eager = GPT(cfg_eager)
    model_sdpa = GPT(cfg_sdpa)

    # Load identical state dict
    model_sdpa.load_state_dict(model_eager.state_dict())

    model_eager.eval()
    model_sdpa.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg_eager.vocab_size, (2, 8))

    with torch.no_grad():
        logits_eager = model_eager(input_ids)
        logits_sdpa = model_sdpa(input_ids)

    torch.testing.assert_close(
        logits_eager,
        logits_sdpa,
        rtol=1e-5,
        atol=1e-5,
        msg="Full model logits diverged between Eager and SDPA backends.",
    )
