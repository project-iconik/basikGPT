"""basikGPT: Educational, reproducible, and open-source GPT-2 Small in PyTorch."""

from typing import Any

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.conversion import (
    convert_hf_gpt2_state_dict,
    load_hf_gpt2_weights,
    validate_hf_config,
)
from basikgpt.data.shard import ShardedTokenDataset, TokenShardWriter
from basikgpt.model.attention import CausalSelfAttention
from basikgpt.model.block import TransformerBlock
from basikgpt.model.gpt import GPT
from basikgpt.model.mlp import MLP
from basikgpt.training import (
    Trainer,
    TrainingConfig,
    compute_cross_entropy_loss,
    configure_optimizers,
    load_checkpoint,
    save_checkpoint,
)

__version__ = "0.1.0"

__all__ = [
    "GPTConfig",
    "AttentionBackend",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
    "GPT",
    "convert_hf_gpt2_state_dict",
    "load_hf_gpt2_weights",
    "validate_hf_config",
    "GPT2Tokenizer",
    "ShardedTokenDataset",
    "TokenShardWriter",
    "prepare_fineweb_edu",
    "TrainingConfig",
    "Trainer",
    "compute_cross_entropy_loss",
    "configure_optimizers",
    "save_checkpoint",
    "load_checkpoint",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazily imports optional data-pipeline symbols that require tiktoken/datasets."""
    if name == "GPT2Tokenizer":
        from basikgpt.data.tokenizer import GPT2Tokenizer

        return GPT2Tokenizer
    if name == "prepare_fineweb_edu":
        from basikgpt.data.pipeline import prepare_fineweb_edu

        return prepare_fineweb_edu
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
