"""Unit and integration tests for TransformerBlock in basikGPT."""

from typing import Literal
import pytest
import torch

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.model.block import TransformerBlock

BACKENDS: list[AttentionBackend] = ["eager", "sdpa"]


def make_config(
    backend: AttentionBackend = "eager",
    dropout: float = 0.0,
    bias: bool = True,
) -> GPTConfig:
    """Helper to create a test GPTConfig."""
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
def test_block_output_shape(backend: AttentionBackend) -> None:
    """Verifies that TransformerBlock preserves the (B, T, C) tensor shape across backends."""
    config = make_config(backend=backend)
    block = TransformerBlock(config)
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)

    out = block(x)

    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seq_len", [1, 4, 16, 64])
def test_block_different_sequence_lengths(backend: AttentionBackend, seq_len: int) -> None:
    """Verifies that TransformerBlock supports various sequence lengths T <= context_length."""
    config = make_config(backend=backend)
    block = TransformerBlock(config)
    B, C = 2, config.d_model
    x = torch.randn(B, seq_len, C)

    out = block(x)

    assert out.shape == (B, seq_len, C)


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_gradient_flow(backend: AttentionBackend) -> None:
    """Verifies complete end-to-end backpropagation through LayerNorms, Attention, and MLP."""
    config = make_config(backend=backend, dropout=0.0)
    block = TransformerBlock(config)
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C, requires_grad=True)

    out = block(x)
    loss = out.sum()
    loss.backward()

    # Input gradient
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert x.grad.abs().sum().item() > 0.0

    # LayerNorm 1 gradients
    assert block.ln_1.weight.grad is not None
    assert block.ln_1.bias.grad is not None

    # Attention parameters gradients
    assert block.attn.qkv_proj.weight.grad is not None
    assert block.attn.out_proj.weight.grad is not None

    # LayerNorm 2 gradients
    assert block.ln_2.weight.grad is not None
    assert block.ln_2.bias.grad is not None

    # MLP parameters gradients
    assert block.mlp.fc_in.weight.grad is not None
    assert block.mlp.fc_out.weight.grad is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_residual_passthrough(backend: AttentionBackend) -> None:
    """Verifies the residual skip connections: when sublayer outputs are zero, output == input."""
    config = make_config(backend=backend, dropout=0.0)
    block = TransformerBlock(config)

    # Zero-out the projection layers that connect into the residual additions
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
        block.mlp.fc_out.weight.zero_()
        block.mlp.fc_out.bias.zero_()

    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)

    out = block(x)

    # Output must strictly equal input because output = x + 0 + 0
    torch.testing.assert_close(out, x, rtol=1e-6, atol=1e-6)


def test_block_pre_norm_topology_behavior() -> None:
    """Verifies Pre-Norm vs Post-Norm behavioral distinction.

    In Pre-Norm: output = x + sublayer(LN(x)).
    Scaling the input x by a factor of 100 scales the residual stream directly,
    resulting in an output whose standard deviation is approximately 100.
    (In a Post-Norm architecture LN(x + sublayer(x)), the output std would be forced to ~1.0).
    """
    config = make_config(backend="eager", dropout=0.0)
    block = TransformerBlock(config)
    block.eval()

    torch.manual_seed(42)
    x = torch.randn(4, 16, config.d_model) * 100.0  # Large scale input

    with torch.no_grad():
        out = block(x)

    # Output scale should be preserved near ~100.0 due to the un-normalized residual stream
    assert out.std().item() > 50.0, "Pre-Norm must preserve the scale of the residual stream."


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_causal_leakage(backend: AttentionBackend) -> None:
    """Verifies that the entire TransformerBlock maintains the causality invariant."""
    config = make_config(backend=backend, dropout=0.0)
    block = TransformerBlock(config)
    block.eval()

    B, T, C = 2, 6, config.d_model

    torch.manual_seed(42)
    input_a = torch.randn(B, T, C)

    # Input B: identical for positions 0 and 1, perturbed at future positions 2..5
    input_b = input_a.clone()
    input_b[:, 2:, :] += torch.randn_like(input_b[:, 2:, :]) * 5.0

    with torch.no_grad():
        output_a = block(input_a)
        output_b = block(input_b)

    # Past positions (0, 1) MUST be identical
    torch.testing.assert_close(
        output_a[:, :2, :],
        output_b[:, :2, :],
        rtol=1e-5,
        atol=1e-5,
        msg=f"Causal leakage detected in TransformerBlock with backend '{backend}'.",
    )

    # Future positions (2..) MUST differ
    diff = (output_a[:, 2:, :] - output_b[:, 2:, :]).abs().max().item()
    assert diff > 1e-3, "Future positions should produce different representations."


def test_block_parameter_counts_exact_match() -> None:
    """Verifies actual TransformerBlock parameter count against the analytical formula."""
    # 1. Test small configuration
    cfg_small = make_config(backend="eager")
    block_small = TransformerBlock(cfg_small)

    C = cfg_small.d_model
    F = cfg_small.d_ff
    bias_mult = 2 if cfg_small.bias else 1

    expected_ln1 = C * bias_mult
    expected_attn = (C * 3 * C + 3 * C) + (C * C + C)
    expected_ln2 = C * bias_mult
    expected_mlp = (C * F + F) + (F * C + C)
    expected_total_per_block = expected_ln1 + expected_attn + expected_ln2 + expected_mlp

    actual_small_params = sum(p.numel() for p in block_small.parameters())
    assert actual_small_params == expected_total_per_block

    # 2. Test canonical GPT-2 Small configuration (C=768, F=3072)
    # Per block params:
    # - ln_1: 768 * 2 = 1,536
    # - attn: (768 * 2304 + 2304) + (768 * 768 + 768) = 1,771,776 + 590,592 = 2,362,368
    # - ln_2: 768 * 2 = 1,536
    # - mlp:  (768 * 3072 + 3072) + (3072 * 768 + 768) = 2,362,368 + 2,360,064 = 4,722,432
    # Total per block = 1,536 + 2,362,368 + 1,536 + 4,722,432 = 7,087,872
    cfg_gpt2 = GPTConfig.gpt2_small()
    block_gpt2 = TransformerBlock(cfg_gpt2)

    actual_gpt2_block_params = sum(p.numel() for p in block_gpt2.parameters())
    expected_gpt2_block_params = 7_087_872

    assert actual_gpt2_block_params == expected_gpt2_block_params


def test_block_parameter_counts_without_bias() -> None:
    """Verifies that when bias=False, LayerNorms and Linear layers cleanly omit biases."""
    cfg_no_bias = make_config(bias=False)
    block_no_bias = TransformerBlock(cfg_no_bias)

    # LayerNorms must not have bias parameters when bias=False
    assert block_no_bias.ln_1.bias is None
    assert block_no_bias.ln_2.bias is None

    C = cfg_no_bias.d_model
    F = cfg_no_bias.d_ff

    expected_ln1 = C
    expected_attn = (C * 3 * C) + (C * C)
    expected_ln2 = C
    expected_mlp = (C * F) + (F * C)
    expected_total = expected_ln1 + expected_attn + expected_ln2 + expected_mlp

    actual_params = sum(p.numel() for p in block_no_bias.parameters())
    assert actual_params == expected_total
