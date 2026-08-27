"""Data processing, tokenization, and sharding utilities for basikGPT."""

from typing import Any

from basikgpt.data.shard import ShardedTokenDataset, TokenShardWriter
from basikgpt.data.split import get_document_split, is_validation_document

__all__ = [
    "GPT2Tokenizer",
    "get_document_split",
    "is_validation_document",
    "TokenShardWriter",
    "ShardedTokenDataset",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "process_document_stream",
    "prepare_fineweb_edu",
]


def __getattr__(name: str) -> Any:
    """Lazily imports tokenizer, manifest, and FineWeb pipeline modules."""
    if name == "GPT2Tokenizer":
        from basikgpt.data.tokenizer import GPT2Tokenizer

        return GPT2Tokenizer
    if name in ("create_manifest", "save_manifest", "load_manifest"):
        from basikgpt.data import manifest as _manifest

        return getattr(_manifest, name)
    if name in ("process_document_stream", "prepare_fineweb_edu"):
        from basikgpt.data import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
