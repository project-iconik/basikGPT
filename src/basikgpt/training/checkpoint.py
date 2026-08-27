"""Atomic checkpoint saving and loading for basikGPT training."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import random
from typing import Any
import torch
import torch.nn as nn
from basikgpt.training.config import TrainingConfig


def save_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_seen: int,
    training_config: TrainingConfig,
    model_config: Any | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically saves model, optimizer, step, tokens, configs, and RNG states to disk.

    Writes to a temporary file first before renaming to prevent corrupted or partial
    checkpoint files if interrupted.

    Args:
        checkpoint_path: Target path for the .pt checkpoint file.
        model: PyTorch neural network module.
        optimizer: Configured PyTorch optimizer.
        global_step: Current global optimizer step index.
        tokens_seen: Total count of target training tokens processed.
        training_config: TrainingConfig instance.
        model_config: Optional GPTConfig instance.
        extra_state: Optional dictionary containing additional custom state metadata.

    Returns:
        Path to the saved checkpoint file.
    """
    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f"{target_path.name}.tmp"

    model_cfg_dict = asdict(model_config) if is_dataclass(model_config) else getattr(model_config, "__dict__", None)
    train_cfg_dict = asdict(training_config) if is_dataclass(training_config) else dict(training_config)

    rng_states = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "training_config": train_cfg_dict,
        "model_config": model_cfg_dict,
        "rng_states": rng_states,
        "extra_state": extra_state or {},
    }

    # 1. Save to temporary path
    torch.save(payload, temp_path)

    # 2. Atomic replacement
    if target_path.exists():
        target_path.unlink()
    temp_path.replace(target_path)

    return target_path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Loads a checkpoint from disk, restoring model and optimizer states.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: Optional optimizer instance to load state into.
        device: Device location to map tensors to during load.
        restore_rng: Whether to restore Python and PyTorch RNG states.

    Returns:
        Dictionary containing checkpoint metadata (global_step, tokens_seen, configs).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If state dict loading fails or weight tying is broken.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if restore_rng and "rng_states" in payload:
        rng = payload["rng_states"]
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if "torch_cpu" in rng and rng["torch_cpu"] is not None:
            torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and "torch_cuda" in rng and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    # Ensure weight tying identity remains intact if present
    if hasattr(model, "lm_head") and hasattr(model, "wte"):
        if model.lm_head.weight is not model.wte.weight:
            # Re-bind if PyTorch load_state_dict separated the parameter instances
            model.lm_head.weight = model.wte.weight

    return {
        "global_step": payload.get("global_step", 0),
        "tokens_seen": payload.get("tokens_seen", 0),
        "training_config": payload.get("training_config", {}),
        "model_config": payload.get("model_config", {}),
        "extra_state": payload.get("extra_state", {}),
    }
