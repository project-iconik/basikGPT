"""basikGPT: Educational, reproducible, and open-source GPT-2 Small in PyTorch."""

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.conversion import (
    convert_hf_gpt2_state_dict,
    load_hf_gpt2_weights,
    validate_hf_config,
)
from basikgpt.model.attention import CausalSelfAttention
from basikgpt.model.block import TransformerBlock
from basikgpt.model.gpt import GPT
from basikgpt.model.mlp import MLP

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
    "__version__",
]
