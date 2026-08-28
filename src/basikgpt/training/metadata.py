"""Experiment metadata serialization, run provenance tracking, and atomic JSON helpers."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
import basikgpt
from basikgpt.config import GPTConfig
from basikgpt.training.config import TrainingConfig
from basikgpt.training.reproducibility import get_git_metadata, get_system_metadata

RUN_FORMAT_VERSION = 1


def perplexity_from_loss(loss: float) -> float:
    """Converts mean cross-entropy to perplexity; overflow maps to +inf."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def gradient_was_clipped(grad_norm: float, max_grad_norm: float | None) -> bool:
    """True when clipping is enabled and the pre-clip L2 norm exceeded the threshold."""
    if max_grad_norm is None:
        return False
    return grad_norm > max_grad_norm


def atomic_save_json(path: Path | str, data: dict[str, Any], indent: int = 2) -> Path:
    """Atomically writes dictionary data to a JSON file via a temporary file replacement.

    Args:
        path: Destination target JSON file path.
        data: Data dictionary to serialize.
        indent: Indentation level for pretty-printing.

    Returns:
        Path to the saved JSON file.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.parent / f"{target_path.name}.tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
        f.flush()

    if target_path.exists():
        target_path.unlink()
    temp_path.replace(target_path)
    return target_path


def load_json(path: Path | str) -> dict[str, Any]:
    """Loads and parses a JSON file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_dataset_provenance(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Reads dataset identity fields from either the canonical or legacy manifest schema.

    Canonical keys live under `dataset_provenance` / `statistics`.
    Legacy test/CLI payloads may use `provenance` / `token_statistics`.
    """
    if not manifest:
        return {
            "revision": None,
            "repository": None,
            "config": None,
            "train_tokens": None,
            "validation_tokens": None,
            "tokenizer_encoding": None,
            "eot_token_id": None,
        }

    prov = manifest.get("dataset_provenance") or manifest.get("provenance") or {}
    stats = manifest.get("statistics") or manifest.get("token_statistics") or {}
    tok_info = manifest.get("tokenizer") or {}

    return {
        "revision": prov.get("revision") or prov.get("dataset_revision"),
        "repository": prov.get("repository") or prov.get("dataset_repository"),
        "config": prov.get("config") or prov.get("dataset_config"),
        "train_tokens": stats.get("train_tokens"),
        "validation_tokens": stats.get("validation_tokens"),
        "tokenizer_encoding": tok_info.get("encoding") or tok_info.get("encoding_name"),
        "eot_token_id": tok_info.get("eot_token_id"),
    }


def save_run_metadata(
    output_dir: Path | str,
    run_name: str,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    dataset_manifest: dict[str, Any] | None = None,
    dataset_manifest_path: Path | str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Serializes run provenance metadata, model config, and training config into output_dir.

    Files written:
        - run.json: Run provenance, git status, hardware and software environment.
        - model_config.json: Architecture parameters.
        - training_config.json: Optimization hyperparameters.
        - dataset.json: Dataset provenance reference if provided.

    Returns:
        Path to run.json.
    """
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    git_info = get_git_metadata()
    sys_info = get_system_metadata()

    # 1. model_config.json
    model_cfg_dict = asdict(model_config) if is_dataclass(model_config) else getattr(model_config, "__dict__", {})
    atomic_save_json(dir_path / "model_config.json", model_cfg_dict)

    # 2. training_config.json
    train_cfg_dict = asdict(training_config) if is_dataclass(training_config) else getattr(training_config, "__dict__", {})
    atomic_save_json(dir_path / "training_config.json", train_cfg_dict)

    # 3. dataset.json (if manifest exists)
    if dataset_manifest is not None:
        atomic_save_json(dir_path / "dataset.json", dataset_manifest)

    # Provenance fields from manifest
    extracted = extract_dataset_provenance(dataset_manifest)
    revision = extracted["revision"]
    train_toks = extracted["train_tokens"]
    val_toks = extracted["validation_tokens"]
    tokenizer_enc = extracted["tokenizer_encoding"]
    eot_id = extracted["eot_token_id"]
    dataset_repo = extracted["repository"]
    dataset_cfg = extracted["config"]

    # 4. run.json
    created_time = datetime.now(timezone.utc).isoformat()
    run_payload = {
        "run_format_version": RUN_FORMAT_VERSION,
        "run_name": run_name,
        "created_at_utc": created_time,
        "basikgpt_version": getattr(basikgpt, "__version__", "0.1.0"),
        "git": git_info,
        "system": sys_info,
        "seed": training_config.seed,
        "device": training_config.device,
        "precision": training_config.precision,
        "dataset_manifest_path": str(dataset_manifest_path) if dataset_manifest_path else None,
        "dataset_repository": dataset_repo,
        "dataset_config": dataset_cfg,
        "dataset_revision": revision,
        "train_tokens": train_toks,
        "validation_tokens": val_toks,
        "tokenizer_encoding": tokenizer_enc,
        "eot_token_id": eot_id,
        "extra": extra_metadata or {},
    }
    return atomic_save_json(dir_path / "run.json", run_payload)


def save_run_summary(
    output_dir: Path | str,
    status: str,
    final_step: int,
    tokens_seen: int,
    elapsed_seconds: float,
    final_train_loss: float | None = None,
    final_val_loss: float | None = None,
    best_val_loss: float | None = None,
    checkpoint_path: Path | str | None = None,
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Writes the final summary.json file for the training run."""
    dir_path = Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "status": status,  # "completed", "interrupted", "failed", "paused"
        "final_step": final_step,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": elapsed_seconds,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "final_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "error_message": error_message,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        summary_payload.update(extra)
    return atomic_save_json(dir_path / "summary.json", summary_payload)
