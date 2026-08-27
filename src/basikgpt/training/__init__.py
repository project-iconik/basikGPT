"""Training engine, optimization, reproducibility, and metadata utilities for basikGPT."""

from basikgpt.training.accounting import (
    TokenBudgetPlan,
    calculate_compile_break_even_tokens,
    calculate_tokens_seen,
    calculate_training_steps,
)
from basikgpt.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from basikgpt.training.compile import (
    SUPPORTED_COMPILE_MODES,
    compile_model,
    unwrap_compiled_model,
)
from basikgpt.training.compatibility import validate_dataset_model_compatibility
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import (
    RUN_FORMAT_VERSION,
    atomic_save_json,
    extract_dataset_provenance,
    load_json,
    save_run_metadata,
    save_run_summary,
)
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.reproducibility import (
    collect_cuda_device_metadata,
    get_git_metadata,
    get_system_metadata,
    seed_everything,
)
from basikgpt.training.scheduler import get_learning_rate_at_step, update_learning_rate
from basikgpt.training.sdpa import (
    ALLOWED_SDPA_KERNEL_NAMES,
    list_probe_sdpa_backends,
    sdpa_kernel_context,
)
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
    "collect_cuda_device_metadata",
    "atomic_save_json",
    "load_json",
    "extract_dataset_provenance",
    "save_run_metadata",
    "save_run_summary",
    "RUN_FORMAT_VERSION",
    "validate_dataset_model_compatibility",
    "compile_model",
    "unwrap_compiled_model",
    "SUPPORTED_COMPILE_MODES",
    "sdpa_kernel_context",
    "list_probe_sdpa_backends",
    "ALLOWED_SDPA_KERNEL_NAMES",
    "calculate_training_steps",
    "calculate_tokens_seen",
    "calculate_compile_break_even_tokens",
    "TokenBudgetPlan",
]
