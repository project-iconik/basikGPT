"""Unit tests for CausalSelfAttention across Eager and SDPA backends in basikGPT."""

from typing import Literal
import pytest
import torch

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.model.attention import CausalSelfAttention

BACKENDS: list[AttentionBackend] = ["eager", "sdpa"]


def make_config(
    backend: AttentionBackend = "eager",
    dropout: float = 0.0,
    bias: bool = True,
) -> GPTConfig:
    """Helper to create a small GPTConfig for unit tests."""
    return GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=dropout,
        bias=bias,
        attention_backend=backend,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_shape(backend: AttentionBackend) -> None:
    """Verifies that CausalSelfAttention preserves the (B, T, C) tensor shape across all backends."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)

    out = attn(x)

    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
def test_different_sequence_lengths(backend: AttentionBackend, seq_len: int) -> None:
    """Verifies that attention supports various sequence lengths T <= context_length across backends."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    B, C = 2, config.d_model
    x = torch.randn(B, seq_len, C)

    out = attn(x)

    assert out.shape == (B, seq_len, C)


@pytest.mark.parametrize("backend", BACKENDS)
def test_context_length_overflow(backend: AttentionBackend) -> None:
    """Verifies that passing T > context_length raises a descriptive ValueError across backends."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    B, C = 2, config.d_model
    overflow_T = config.context_length + 1
    x = torch.randn(B, overflow_T, C)

    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        attn(x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_feature_dimension_mismatch(backend: AttentionBackend) -> None:
    """Verifies that passing an input with mismatched C raises a descriptive ValueError."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    B, T = 2, 8
    wrong_C = config.d_model + 8
    x = torch.randn(B, T, wrong_C)

    with pytest.raises(ValueError, match="does not match model d_model"):
        attn(x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_causal_leakage(backend: AttentionBackend) -> None:
    """Verifies the Causal Invariant for both Eager and SDPA.

    Modifying future tokens at positions t >= 2 must strictly NOT change
    the attention representations at past positions t < 2.
    """
    config = make_config(backend=backend, dropout=0.0)
    attn = CausalSelfAttention(config)
    attn.eval()  # Ensure dropout is 0.0

    B, T, C = 2, 6, config.d_model

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
        msg=f"Causal leakage detected in '{backend}' backend.",
    )

    # 2. Future positions (2..) MUST differ
    diff = (output_a[:, 2:, :] - output_b[:, 2:, :]).abs().max().item()
    assert diff > 1e-3, f"Future tokens should produce different representations in '{backend}'."


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_flow(backend: AttentionBackend) -> None:
    """Verifies backward autograd pass and gradient propagation across backends."""
    config = make_config(backend=backend, dropout=0.0)
    attn = CausalSelfAttention(config)
    B, T, C = 2, 8, config.d_model
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_mode_determinism(backend: AttentionBackend) -> None:
    """Verifies deterministic behavior in eval mode (dropout disabled) across backends."""
    config = make_config(backend=backend, dropout=0.5)
    attn = CausalSelfAttention(config)
    attn.eval()

    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)

    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x)

    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_training_mode_stochasticity(backend: AttentionBackend) -> None:
    """Verifies stochastic behavior in train mode when dropout > 0.0 across backends."""
    config = make_config(backend=backend, dropout=0.5)
    attn = CausalSelfAttention(config)
    attn.train()

    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)

    out1 = attn(x)
    out2 = attn(x)

    assert not torch.equal(out1, out2), f"Outputs should differ in training mode with dropout for {backend}."


@pytest.mark.parametrize("backend", BACKENDS)
def test_parameter_shapes(backend: AttentionBackend) -> None:
    """Verifies structural weight and bias dimensions for linear projections across backends."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    C = config.d_model

    assert attn.qkv_proj.weight.shape == (3 * C, C)
    assert attn.qkv_proj.bias is not None
    assert attn.qkv_proj.bias.shape == (3 * C,)

    assert attn.out_proj.weight.shape == (C, C)
    assert attn.out_proj.bias is not None
    assert attn.out_proj.bias.shape == (C,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bias_disabled(backend: AttentionBackend) -> None:
    """Verifies that bias=False cleanly omits biases across backends."""
    config = make_config(backend=backend, bias=False)
    attn = CausalSelfAttention(config)

    assert attn.qkv_proj.bias is None
    assert attn.out_proj.bias is None

    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)
    out = attn(x)
    assert out.shape == (B, T, C)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_token_step(backend: AttentionBackend) -> None:
    """Verifies mathematical identity for T=1 across backends."""
    config = make_config(backend=backend)
    attn = CausalSelfAttention(config)
    attn.eval()

    B, T, C = 1, 1, config.d_model
    x = torch.randn(B, T, C)

    with torch.no_grad():
        out = attn(x)

    assert out.shape == (1, 1, C)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Numerical Parity & Equivalence Tests between Eager and SDPA
# ---------------------------------------------------------------------------


def test_eager_sdpa_forward_numerical_parity() -> None:
    """Verifies that Eager and SDPA backends produce numerically equivalent forward outputs.

    Condition:
        - Device: CPU
        - Precision: FP32
        - Dropout: 0.0
        - Weights: Exactly identical via state_dict transfer
    """
    cfg_eager = make_config(backend="eager", dropout=0.0)
    cfg_sdpa = make_config(backend="sdpa", dropout=0.0)

    attn_eager = CausalSelfAttention(cfg_eager)
    attn_sdpa = CausalSelfAttention(cfg_sdpa)

    # Transfer identical weights
    attn_sdpa.load_state_dict(attn_eager.state_dict())

    attn_eager.eval()
    attn_sdpa.eval()

    torch.manual_seed(1337)
    B, T, C = 4, 16, cfg_eager.d_model
    x = torch.randn(B, T, C, dtype=torch.float32)

    with torch.no_grad():
        out_eager = attn_eager(x)
        out_sdpa = attn_sdpa(x)

    # Numerical tolerance for FP32 operations (rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        out_eager,
        out_sdpa,
        rtol=1e-5,
        atol=1e-5,
        msg="Eager and SDPA forward outputs diverged beyond FP32 numerical tolerance.",
    )


@pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
def test_eager_sdpa_parity_across_sequence_lengths(seq_len: int) -> None:
    """Verifies forward numerical parity across various sequence lengths."""
    cfg_eager = make_config(backend="eager", dropout=0.0)
    cfg_sdpa = make_config(backend="sdpa", dropout=0.0)

    attn_eager = CausalSelfAttention(cfg_eager)
    attn_sdpa = CausalSelfAttention(cfg_sdpa)
    attn_sdpa.load_state_dict(attn_eager.state_dict())

    attn_eager.eval()
    attn_sdpa.eval()

    torch.manual_seed(42 + seq_len)
    B, C = 2, cfg_eager.d_model
    x = torch.randn(B, seq_len, C, dtype=torch.float32)

    with torch.no_grad():
        out_eager = attn_eager(x)
        out_sdpa = attn_sdpa(x)

    torch.testing.assert_close(out_eager, out_sdpa, rtol=1e-5, atol=1e-5)


def test_eager_sdpa_parameter_count_and_state_dict_parity() -> None:
    """Verifies parameter count invariance and seamless state_dict transfer between backends."""
    cfg_eager = make_config(backend="eager")
    cfg_sdpa = make_config(backend="sdpa")

    attn_eager = CausalSelfAttention(cfg_eager)
    attn_sdpa = CausalSelfAttention(cfg_sdpa)

    # Parameter counts must be strictly identical
    eager_params = list(attn_eager.parameters())
    sdpa_params = list(attn_sdpa.parameters())
    assert len(eager_params) == len(sdpa_params)

    eager_total_numel = sum(p.numel() for p in eager_params)
    sdpa_total_numel = sum(p.numel() for p in sdpa_params)
    assert eager_total_numel == sdpa_total_numel

    # State dict keys must match exactly (non-persistent buffer not in state_dict)
    eager_keys = set(attn_eager.state_dict().keys())
    sdpa_keys = set(attn_sdpa.state_dict().keys())
    assert eager_keys == sdpa_keys
    assert "causal_mask" not in eager_keys
    assert "causal_mask" not in sdpa_keys

    # Direct weight transfer without errors
    load_result = attn_sdpa.load_state_dict(attn_eager.state_dict())
    assert len(load_result.missing_keys) == 0
    assert len(load_result.unexpected_keys) == 0


def test_eager_sdpa_gradient_parity() -> None:
    """Verifies backward gradient parity between Eager and SDPA backends."""
    cfg_eager = make_config(backend="eager", dropout=0.0)
    cfg_sdpa = make_config(backend="sdpa", dropout=0.0)

    attn_eager = CausalSelfAttention(cfg_eager)
    attn_sdpa = CausalSelfAttention(cfg_sdpa)
    attn_sdpa.load_state_dict(attn_eager.state_dict())

    torch.manual_seed(999)
    B, T, C = 2, 8, cfg_eager.d_model

    x_base = torch.randn(B, T, C, dtype=torch.float32)
    x_eager = x_base.clone().requires_grad_(True)
    x_sdpa = x_base.clone().requires_grad_(True)

    out_eager = attn_eager(x_eager)
    out_sdpa = attn_sdpa(x_sdpa)

    loss_eager = out_eager.sum()
    loss_sdpa = out_sdpa.sum()

    loss_eager.backward()
    loss_sdpa.backward()

    # Compare input gradients
    assert x_eager.grad is not None and x_sdpa.grad is not None
    torch.testing.assert_close(x_eager.grad, x_sdpa.grad, rtol=1e-5, atol=1e-5)

    # Compare projection weight gradients
    assert attn_eager.qkv_proj.weight.grad is not None and attn_sdpa.qkv_proj.weight.grad is not None
    torch.testing.assert_close(
        attn_eager.qkv_proj.weight.grad,
        attn_sdpa.qkv_proj.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )

    assert attn_eager.out_proj.weight.grad is not None and attn_sdpa.out_proj.weight.grad is not None
    torch.testing.assert_close(
        attn_eager.out_proj.weight.grad,
        attn_sdpa.out_proj.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
