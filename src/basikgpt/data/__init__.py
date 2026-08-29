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
    "prepare_hf_corpus",
    "combine_shard_directories",
    "interleave_cycle",
]


def __getattr__(name: str) -> Any:
    """Lazily imports tokenizer, manifest, and FineWeb pipeline modules."""
    if name == "GPT2Tokenizer":
        from basikgpt.data.tokenizer import GPT2Tokenizer

        return GPT2Tokenizer
    if name in ("create_manifest", "save_manifest", "load_manifest"):
        from basikgpt.data import manifest as _manifest

        return getattr(_manifest, name)
    if name in ("process_document_stream", "prepare_fineweb_edu", "prepare_hf_corpus"):
        from basikgpt.data import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in ("combine_shard_directories", "interleave_cycle"):
        from basikgpt.data import combine as _combine

        return getattr(_combine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
