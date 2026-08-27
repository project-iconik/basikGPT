"""Unit tests for causal cross-entropy loss computation."""

import pytest
import torch
import torch.nn.functional as F
from basikgpt.training.loss import compute_cross_entropy_loss


def test_loss_shape_and_known_answer() -> None:
    """Verifies that compute_cross_entropy_loss matches F.cross_entropy on flattened tensors."""
    B, T, V = 2, 4, 10
    torch.manual_seed(42)
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T), dtype=torch.long)

    loss = compute_cross_entropy_loss(logits, targets)

    # Manual reference
    expected = F.cross_entropy(logits.view(-1, V), targets.view(-1))

    assert loss.ndim == 0, "Loss must be a scalar tensor"
    torch.testing.assert_close(loss, expected)


def test_loss_shape_validation_errors() -> None:
    """Verifies that dimension and shape mismatches raise ValueError."""
    # 4D logits
    with pytest.raises(ValueError, match="Expected 3D logits tensor"):
        compute_cross_entropy_loss(torch.randn(2, 4, 8, 16), torch.randint(0, 16, (2, 4)))

    # 1D targets
    with pytest.raises(ValueError, match="Expected 2D targets tensor"):
        compute_cross_entropy_loss(torch.randn(2, 4, 16), torch.randint(0, 16, (8,)))

    # Mismatched B or T
    with pytest.raises(ValueError, match="Targets shape"):
        compute_cross_entropy_loss(torch.randn(2, 4, 16), torch.randint(0, 16, (2, 5)))


def test_loss_backward_gradient_flow() -> None:
    """Verifies that backward pass through compute_cross_entropy_loss produces non-zero gradients."""
    logits = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.randint(0, 16, (2, 4))

    loss = compute_cross_entropy_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.abs().sum() > 0
