"""basikGPT: Educational, reproducible, and open-source GPT-2 Small in PyTorch."""

from basikgpt.config import AttentionBackend, GPTConfig
from basikgpt.model.attention import CausalSelfAttention

__version__ = "0.1.0"

__all__ = [
    "GPTConfig",
    "AttentionBackend",
    "CausalSelfAttention",
    "__version__",
]
