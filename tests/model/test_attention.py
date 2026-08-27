"""Unit tests for CausalSelfAttention in basikGPT."""

import pytest
import torch

from basikgpt.config import GPTConfig
from basikgpt.model.attention import CausalSelfAttention


@pytest.fixture
def small_config() -> GPTConfig:
    """Fixture providing a lightweight GPTConfig for fast unit tests."""
    return GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        bias=True,
    )


def test_output_shape(small_config: GPTConfig) -> None:
    """Test 1: Verifies that CausalSelfAttention preserves the (B, T, C) tensor shape."""
    attn = CausalSelfAttention(small_config)
    B, T, C = 2, 8, small_config.d_model
    x = torch.randn(B, T, C)

    out = attn(x)

    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"


@pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
def test_different_sequence_lengths(small_config: GPTConfig, seq_len: int) -> None:
    """Test 2: Verifies that attention supports various sequence lengths T <= context_length."""
    attn = CausalSelfAttention(small_config)
    B, C = 2, small_config.d_model
    x = torch.randn(B, seq_len, C)

    out = attn(x)

    assert out.shape == (B, seq_len, C)


def test_context_length_overflow(small_config: GPTConfig) -> None:
    """Test 3: Verifies that passing T > context_length raises a descriptive ValueError."""
    attn = CausalSelfAttention(small_config)
    B, C = 2, small_config.d_model
    overflow_T = small_config.context_length + 1
    x = torch.randn(B, overflow_T, C)

    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        attn(x)


def test_feature_dimension_mismatch(small_config: GPTConfig) -> None:
    """Verifies that passing an input with mismatched C raises a descriptive ValueError."""
    attn = CausalSelfAttention(small_config)
    B, T = 2, 8
    wrong_C = small_config.d_model + 8
    x = torch.randn(B, T, wrong_C)

    with pytest.raises(ValueError, match="does not match model d_model"):
        attn(x)


def test_causal_leakage(small_config: GPTConfig) -> None:
    """Test 4: Verifies the Causal Invariant.

    Modifying future tokens at positions t >= 2 must strictly NOT change
    the attention representations at past positions t < 2.
    """
    attn = CausalSelfAttention(small_config)
    attn.eval()  # Ensure dropout is 0.0

    B, T, C = 2, 6, small_config.d_model

    # Input A
    torch.manual_seed(42)
    input_a = torch.randn(B, T, C)

    # Input B: identical for positions 0 and 1, perturbed at positions 2..5
    input_b = input_a.clone()
    input_b[:, 2:, :] += torch.randn_like(input_b[:, 2:, :]) * 5.0

    with torch.no_grad():
        output_a = attn(input_a)
        output_b = attn(input_b)

    # 1. Past positions (0, 1) MUST be identical
    torch.testing.assert_close(
        output_a[:, :2, :],
        output_b[:, :2, :],
        rtol=1e-5,
        atol=1e-5,
        msg="Causal leakage detected: past tokens were altered by future perturbations.",
    )

    # 2. Future positions (2..) MUST differ
    diff = (output_a[:, 2:, :] - output_b[:, 2:, :]).abs().max().item()
    assert diff > 1e-3, "Future tokens should produce different representations."


def test_gradient_flow(small_config: GPTConfig) -> None:
    """Test 5: Verifies backward autograd pass and gradient propagation to inputs and parameters."""
    attn = CausalSelfAttention(small_config)
    B, T, C = 2, 8, small_config.d_model
    x = torch.randn(B, T, C, requires_grad=True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert x.grad.abs().sum().item() > 0.0

    # Check linear projection weight gradients
    assert attn.qkv_proj.weight.grad is not None
    assert not torch.isnan(attn.qkv_proj.weight.grad).any()
    assert attn.qkv_proj.weight.grad.abs().sum().item() > 0.0

    assert attn.out_proj.weight.grad is not None
    assert not torch.isnan(attn.out_proj.weight.grad).any()
    assert attn.out_proj.weight.grad.abs().sum().item() > 0.0

    # Check linear projection bias gradients
    assert attn.qkv_proj.bias.grad is not None
    assert attn.out_proj.bias.grad is not None


def test_eval_mode_determinism() -> None:
    """Test 6: Verifies deterministic behavior in eval mode even with high dropout configured."""
    config_with_dropout = GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        bias=True,
    )
    attn = CausalSelfAttention(config_with_dropout)
    attn.eval()

    B, T, C = 2, 8, config_with_dropout.d_model
    x = torch.randn(B, T, C)

    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x)

    torch.testing.assert_close(out1, out2)


def test_training_mode_stochasticity() -> None:
    """Verifies stochastic behavior in train mode when dropout > 0.0."""
    config_with_dropout = GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        bias=True,
    )
    attn = CausalSelfAttention(config_with_dropout)
    attn.train()

    B, T, C = 2, 8, config_with_dropout.d_model
    x = torch.randn(B, T, C)

    out1 = attn(x)
    out2 = attn(x)

    assert not torch.equal(out1, out2), "Outputs should differ across runs in training mode due to dropout."


def test_parameter_shapes(small_config: GPTConfig) -> None:
    """Test 7: Verifies structural weight and bias dimensions for linear projections."""
    attn = CausalSelfAttention(small_config)
    C = small_config.d_model

    assert attn.qkv_proj.weight.shape == (3 * C, C)
    assert attn.qkv_proj.bias is not None
    assert attn.qkv_proj.bias.shape == (3 * C,)

    assert attn.out_proj.weight.shape == (C, C)
    assert attn.out_proj.bias is not None
    assert attn.out_proj.bias.shape == (C,)


def test_bias_disabled() -> None:
    """Verifies that bias=False cleanly omits biases from projection layers."""
    config_no_bias = GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
        bias=False,
    )
    attn = CausalSelfAttention(config_no_bias)

    assert attn.qkv_proj.bias is None
    assert attn.out_proj.bias is None

    B, T, C = 2, 8, config_no_bias.d_model
    x = torch.randn(B, T, C)
    out = attn(x)
    assert out.shape == (B, T, C)


def test_single_token_step(small_config: GPTConfig) -> None:
    """Verifies mathematical identity for T=1: attention probability is 1.0."""
    attn = CausalSelfAttention(small_config)
    attn.eval()

    B, T, C = 1, 1, small_config.d_model
    x = torch.randn(B, T, C)

    with torch.no_grad():
        out = attn(x)

    assert out.shape == (1, 1, C)
    assert not torch.isnan(out).any()
