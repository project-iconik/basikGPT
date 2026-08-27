"""Atomic, versioned, and validated checkpoint saving and loading for basikGPT."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import random
from typing import Any
import torch
import torch.nn as nn
from basikgpt.training.config import TrainingConfig

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    tokens_seen: int,
    training_config: TrainingConfig,
    model_config: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically saves model, optimizer, step, tokens, configs, scaler, and RNG states to disk.

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
        scaler: Optional GradScaler instance (used for FP16 mixed precision).
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
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
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
    scaler: torch.amp.GradScaler | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
    expected_model_config: Any | None = None,
) -> dict[str, Any]:
    """Loads a versioned checkpoint from disk, validating schema integrity and model compatibility.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: PyTorch model instance to load weights into.
        optimizer: Optional optimizer instance to load state into.
        scaler: Optional GradScaler instance to load state into.
        device: Device location to map tensors to during load.
        restore_rng: Whether to restore Python and PyTorch RNG states.
        expected_model_config: Optional GPTConfig to validate architecture compatibility.

    Returns:
        Dictionary containing checkpoint metadata (global_step, tokens_seen, configs).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint file is corrupted or unreadable.
        ValueError: If schema version is unsupported, required fields are missing, or architecture mismatches.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint '{path}': corrupted or unreadable file") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format in '{path}': expected dict, got {type(payload).__name__}")

    # 1. Schema version validation
    schema_ver = payload.get("schema_version", 1)
    if not isinstance(schema_ver, int) or schema_ver > CHECKPOINT_SCHEMA_VERSION or schema_ver < 1:
        raise ValueError(
            f"Unsupported checkpoint schema version '{schema_ver}'. "
            f"Maximum supported schema version is {CHECKPOINT_SCHEMA_VERSION}."
        )

    # 2. Required fields validation
    required_fields = [
        "model_state_dict",
        "optimizer_state_dict",
        "global_step",
        "tokens_seen",
        "model_config",
        "training_config",
    ]
    missing = [f for f in required_fields if f not in payload or payload[f] is None]
    if missing:
        raise ValueError(f"Checkpoint '{path}' is missing required fields: {', '.join(missing)}")

    # 3. Model architecture compatibility validation
    curr_cfg = expected_model_config or getattr(model, "config", None)
    saved_cfg = payload.get("model_config", {})
    if curr_cfg is not None and isinstance(saved_cfg, dict):
        arch_keys = ["vocab_size", "context_length", "n_layers", "n_heads", "d_model", "d_ff"]
        curr_cfg_dict = asdict(curr_cfg) if is_dataclass(curr_cfg) else getattr(curr_cfg, "__dict__", {})
        mismatches = []
        for key in arch_keys:
            if key in curr_cfg_dict and key in saved_cfg:
                if curr_cfg_dict[key] != saved_cfg[key]:
                    mismatches.append(f"{key}: current={curr_cfg_dict[key]} vs checkpoint={saved_cfg[key]}")
        if mismatches:
            raise ValueError(
                f"Checkpoint architecture is incompatible with current model:\n  "
                + "\n  ".join(mismatches)
            )

    # 4. Load state dicts
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

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
            model.lm_head.weight = model.wte.weight

    return {
        "schema_version": schema_ver,
        "global_step": payload.get("global_step", 0),
        "tokens_seen": payload.get("tokens_seen", 0),
        "training_config": payload.get("training_config", {}),
        "model_config": payload.get("model_config", {}),
        "extra_state": payload.get("extra_state", {}),
    }
