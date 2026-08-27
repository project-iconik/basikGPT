"""Training engine, optimization, reproducibility, and metadata utilities for basikGPT."""

from basikgpt.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from basikgpt.training.compatibility import validate_dataset_model_compatibility
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import (
    RUN_FORMAT_VERSION,
    atomic_save_json,
    load_json,
    save_run_metadata,
    save_run_summary,
)
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.reproducibility import (
    get_git_metadata,
    get_system_metadata,
    seed_everything,
)
from basikgpt.training.scheduler import get_learning_rate_at_step, update_learning_rate
from basikgpt.training.trainer import Trainer, resolve_device

__all__ = [
    "TrainingConfig",
    "compute_cross_entropy_loss",
    "configure_optimizers",
    "get_learning_rate_at_step",
    "update_learning_rate",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "CHECKPOINT_SCHEMA_VERSION",
    "Trainer",
    "resolve_device",
    "seed_everything",
    "get_git_metadata",
    "get_system_metadata",
    "atomic_save_json",
    "load_json",
    "save_run_metadata",
    "save_run_summary",
    "RUN_FORMAT_VERSION",
    "validate_dataset_model_compatibility",
]
