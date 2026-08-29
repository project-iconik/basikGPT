"""Dataset manifest creation, loading, and integrity validation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np

import basikgpt


def create_manifest(
    dataset_repository: str,
    dataset_config: str,
    dataset_revision: str,
    validation_fraction: float,
    shard_token_target: int,
    stats: dict[str, int],
    shards: list[dict[str, Any]],
    dataset_license: str = "ODC-By 1.0",
    selection: str = "FineWeb-Edu upstream educational-quality filtering",
) -> dict[str, Any]:
    """Constructs a comprehensive manifest dictionary capturing provenance and metadata."""
    manifest = {
        "format_version": "1.0",
        "dataset_provenance": {
            "repository": dataset_repository,
            "config": dataset_config,
            "revision": dataset_revision,
            "language": "en",
            "license": dataset_license,
            "selection": selection,
        },
        "tokenizer": {
            "name": "tiktoken",
            "encoding": "gpt2",
            "vocab_size": 50257,
            "eot_token_id": 50256,
            "special_token_policy": "encode_ordinary + appended EOT",
        },
        "split_policy": {
            "method": "sha256-hash-bucket-v1",
            "validation_fraction": validation_fraction,
            "salt": "basikgpt-fineweb-edu-v1",
            "bucket_count": 10000,
        },
        "storage": {
            "format": ".npy",
            "dtype": "uint16",
            "shard_token_target": shard_token_target,
        },
        "statistics": {
            "total_documents_seen": stats.get("total_documents_seen", 0),
            "train_documents": stats.get("train_documents", 0),
            "validation_documents": stats.get("validation_documents", 0),
            "skipped_documents": stats.get("skipped_documents", 0),
            "skipped_for_budget": stats.get("skipped_for_budget", 0),
            "train_tokens": stats.get("train_tokens", 0),
            "validation_tokens": stats.get("validation_tokens", 0),
            "train_shards": len([s for s in shards if s.get("split") == "train"]),
            "validation_shards": len([s for s in shards if s.get("split") == "validation"]),
        },
        "shards": shards,
        "environment": {
            "python": sys.version.split()[0],
            "tiktoken": _optional_package_version("tiktoken"),
            "datasets": _optional_package_version("datasets"),
            "numpy": getattr(np, "__version__", "unknown"),
            "basikgpt": getattr(basikgpt, "__version__", "0.1.0"),
        },
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def _optional_package_version(module_name: str) -> str:
    """Returns an installed package version, or 'unknown' if the package is absent."""
    try:
        module = __import__(module_name)
    except ImportError:
        return "unknown"
    return getattr(module, "__version__", "unknown")


def save_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Saves manifest dictionary to JSON file with indentation."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Loads manifest dictionary from JSON file."""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)
