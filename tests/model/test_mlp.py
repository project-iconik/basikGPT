"""Unit tests for GPT-2 MLP module in basikGPT."""

import pytest
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.model.mlp import MLP


@pytest.fixture
def small_config() -> GPTConfig:
    """Provides a lightweight GPTConfig for MLP tests."""
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


def test_mlp_output_shape(small_config: GPTConfig) -> None:
    """Verifies that MLP preserves the (B, T, C) tensor shape."""
    mlp = MLP(small_config)
    B, T, C = 2, 8, small_config.d_model
    x = torch.randn(B, T, C)

    out = mlp(x)

    assert out.shape == (B, T, C), f"Expected {(B, T, C)}, got {out.shape}"


def test_mlp_parameter_shapes(small_config: GPTConfig) -> None:
    """Verifies that linear projection weight and bias matrices match (F, C) and (C, F)."""
    mlp = MLP(small_config)
    C = small_config.d_model
    F = small_config.d_ff

    assert mlp.fc_in.weight.shape == (F, C)
    assert mlp.fc_in.bias is not None
    assert mlp.fc_in.bias.shape == (F,)

    assert mlp.fc_out.weight.shape == (C, F)
    assert mlp.fc_out.bias is not None
    assert mlp.fc_out.bias.shape == (C,)


def test_mlp_bias_disabled() -> None:
    """Verifies that bias=False cleanly omits biases from MLP linear projections."""
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
    mlp = MLP(config_no_bias)

    assert mlp.fc_in.bias is None
    assert mlp.fc_out.bias is None

    x = torch.randn(2, 8, config_no_bias.d_model)
    out = mlp(x)
    assert out.shape == (2, 8, config_no_bias.d_model)


def test_mlp_gradient_flow(small_config: GPTConfig) -> None:
    """Verifies backpropagation gradient flow through fc_in, GELU, fc_out, and input x."""
    mlp = MLP(small_config)
    B, T, C = 2, 8, small_config.d_model
    x = torch.randn(B, T, C, requires_grad=True)

    out = mlp(x)
    loss = out.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert x.grad.abs().sum().item() > 0.0

    # Check projection weights gradients
    assert mlp.fc_in.weight.grad is not None
    assert not torch.isnan(mlp.fc_in.weight.grad).any()
    assert mlp.fc_in.weight.grad.abs().sum().item() > 0.0

    assert mlp.fc_out.weight.grad is not None
    assert not torch.isnan(mlp.fc_out.weight.grad).any()
    assert mlp.fc_out.weight.grad.abs().sum().item() > 0.0

    # Check projection biases gradients
    assert mlp.fc_in.bias.grad is not None
    assert mlp.fc_out.bias.grad is not None


def test_mlp_eval_mode_determinism() -> None:
    """Verifies that MLP with dropout is completely deterministic in eval mode."""
    config_dropout = GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        bias=True,
    )
    mlp = MLP(config_dropout)
    mlp.eval()

    x = torch.randn(2, 8, config_dropout.d_model)
    with torch.no_grad():
        out1 = mlp(x)
        out2 = mlp(x)

    torch.testing.assert_close(out1, out2)


def test_mlp_training_mode_stochasticity() -> None:
    """Verifies that MLP dropout causes stochastic outputs during training mode."""
    config_dropout = GPTConfig(
        vocab_size=100,
        context_length=64,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.5,
        bias=True,
    )
    mlp = MLP(config_dropout)
    mlp.train()

    x = torch.randn(2, 8, config_dropout.d_model)
    out1 = mlp(x)
    out2 = mlp(x)

    assert not torch.equal(out1, out2), "MLP outputs should differ in training mode due to dropout."


def test_mlp_gelu_tanh_approximation(small_config: GPTConfig) -> None:
    """Verifies that the GELU activation explicitly uses the GPT-2 compatible 'tanh' approximation."""
    mlp = MLP(small_config)
    assert isinstance(mlp.activation, nn.GELU)
    assert mlp.activation.approximate == "tanh"
