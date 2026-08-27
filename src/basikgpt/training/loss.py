"""Causal language model cross-entropy loss function."""

import torch
import torch.nn.functional as F


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Computes mean cross-entropy loss for autoregressive next-token prediction.

    Shapes:
        logits:  (B, T, V) where B is batch size, T is sequence length, V is vocab size.
        targets: (B, T) containing ground-truth next token indices.

    Flattened View:
        logits:  (B * T, V)
        targets: (B * T,)
        Output:  Scalar tensor (mean negative log-likelihood).

    Args:
        logits: Unnormalized token prediction scores from GPT.
        targets: Target token IDs (shifted by 1 token relative to input).

    Returns:
        Scalar PyTorch tensor representing mean cross-entropy loss.

    Raises:
        ValueError: If input tensor shapes or dimensions are mismatched.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits tensor of shape (B, T, V), got {logits.ndim}D: {logits.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D targets tensor of shape (B, T), got {targets.ndim}D: {targets.shape}")

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(f"Targets shape {targets.shape} does not match logits batch/time dimensions {(B, T)}")

    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    return F.cross_entropy(flat_logits, flat_targets)
