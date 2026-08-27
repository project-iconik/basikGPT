"""Language model evaluation and perplexity calculation for basikGPT."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basikgpt.config import GPTConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import atomic_save_json
from basikgpt.training.reproducibility import get_git_metadata, get_system_metadata


def evaluate_language_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluates a language model on a dataloader computing exact token-weighted cross-entropy loss and perplexity.

    Aggregation:
        Calculates the true token-weighted negative log-likelihood:
            mean_loss = sum(batch_loss * batch_target_tokens) / sum(batch_target_tokens)
            perplexity = exp(mean_loss)

    Args:
        model: Pretrained or checkpointed GPT model instance.
        dataloader: PyTorch DataLoader yielding (input_ids, target_ids) batches.
        device: Device to execute forward pass on.
        max_batches: Maximum number of batches to evaluate (None evaluates full dataloader).

    Returns:
        Dictionary containing:
            - validation_loss: Mean cross-entropy loss across all evaluated tokens.
            - perplexity: Exponentiated cross-entropy loss (or float('inf') on overflow).
            - evaluated_tokens: Total target token count evaluated.
            - batches_evaluated: Total batch count processed.
    """
    was_training = model.training
    model.eval()

    device_obj = torch.device(device) if isinstance(device, str) else device
    model.to(device_obj)

    total_nll = 0.0
    total_tokens = 0
    batches_evaluated = 0

    with torch.inference_mode():
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            x = x.to(device_obj, non_blocking=True)
            y = y.to(device_obj, non_blocking=True)

            logits = model(x)
            loss = compute_cross_entropy_loss(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Non-finite loss encountered during evaluation at batch {i}: {loss.item()}")

            batch_tokens = y.numel()
            total_nll += loss.item() * batch_tokens
            total_tokens += batch_tokens
            batches_evaluated += 1

    if was_training:
        model.train()

    mean_loss = total_nll / max(1, total_tokens)

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "validation_loss": mean_loss,
        "perplexity": perplexity,
        "evaluated_tokens": total_tokens,
        "batches_evaluated": batches_evaluated,
    }


def save_evaluation_result(
    output_path: Path | str,
    eval_metrics: dict[str, Any],
    checkpoint_path: Path | str | None = None,
    model_config: GPTConfig | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    dataset_manifest_path: Path | str | None = None,
    device: str | torch.device = "cpu",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializes evaluation results and runtime provenance to a machine-readable JSON file."""
    model_cfg_dict = asdict(model_config) if is_dataclass(model_config) else getattr(model_config, "__dict__", None)

    git_info = get_git_metadata()
    sys_info = get_system_metadata()

    # Extract dataset provenance
    revision = None
    dataset_repo = None
    dataset_cfg = None
    if dataset_manifest:
        prov = dataset_manifest.get("dataset_provenance") or dataset_manifest.get("provenance") or {}
        revision = prov.get("revision") or prov.get("dataset_revision")
        dataset_repo = prov.get("repository") or prov.get("dataset_repository")
        dataset_cfg = prov.get("config") or prov.get("dataset_config")

    payload = {
        "evaluation_format_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "device": str(device),
        "metrics": eval_metrics,
        "model_config": model_cfg_dict,
        "dataset": {
            "manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
            "repository": dataset_repo,
            "config": dataset_cfg,
            "revision": revision,
        },
        "git": git_info,
        "system": sys_info,
        "extra": extra_metadata or {},
    }
    return atomic_save_json(output_path, payload)
