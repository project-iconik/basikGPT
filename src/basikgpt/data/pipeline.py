"""End-to-end tokenization and sharding pipeline for HuggingFace FineWeb-Edu."""

from collections.abc import Iterable
import hashlib
from pathlib import Path
import time
from typing import Any
from datasets import load_dataset

from basikgpt.data.manifest import create_manifest, save_manifest
from basikgpt.data.shard import TokenShardWriter
from basikgpt.data.split import get_document_split
from basikgpt.data.tokenizer import GPT2Tokenizer


def process_document_stream(
    doc_stream: Iterable[dict[str, Any]],
    output_dir: Path,
    tokenizer: GPT2Tokenizer,
    max_train_tokens: int,
    max_validation_tokens: int,
    shard_token_target: int = 1_000_000,
    val_fraction: float = 0.005,
    log_interval: int = 500,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Processes an iterable stream of document dicts into tokenized uint16 shards.

    Args:
        doc_stream: Iterable yielding dicts with at least 'text' and optional 'id'/'url'.
        output_dir: Directory where shards and manifest will be written.
        tokenizer: Initialized GPT2Tokenizer instance.
        max_train_tokens: Upper bound on total training tokens to collect.
        max_validation_tokens: Upper bound on total validation tokens to collect.
        shard_token_target: Target number of tokens per .npy shard.
        val_fraction: Fraction of document IDs to route to validation split.
        log_interval: Frequency (in documents seen) for progress output.

    Returns:
        Tuple of (statistics_dict, list_of_all_completed_shards).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_writer = TokenShardWriter(output_dir, "train", tokens_per_shard=shard_token_target)
    val_writer = TokenShardWriter(output_dir, "validation", tokens_per_shard=shard_token_target)

    docs_seen = 0
    docs_skipped = 0
    train_docs = 0
    val_docs = 0
    train_tokens = 0
    val_tokens = 0

    start_time = time.perf_counter()

    for item in doc_stream:
        # Check if both budgets are satisfied
        if train_tokens >= max_train_tokens and val_tokens >= max_validation_tokens:
            break

        docs_seen += 1
        text = item.get("text", "")
        if not text or not text.strip():
            docs_skipped += 1
            continue

        tokens = tokenizer.encode_document(text)
        if not tokens:
            docs_skipped += 1
            continue

        # Extract stable document ID with fallback
        doc_id = item.get("id")
        if not doc_id:
            doc_id = item.get("url") or hashlib.sha256(text.encode("utf-8")).hexdigest()

        split = get_document_split(str(doc_id), val_fraction=val_fraction)

        if split == "train":
            if train_tokens < max_train_tokens:
                remaining_budget = max_train_tokens - train_tokens
                if len(tokens) > remaining_budget:
                    tokens = tokens[:remaining_budget]
                train_writer.add_tokens(tokens)
                train_tokens += len(tokens)
                train_docs += 1
        else:
            if val_tokens < max_validation_tokens:
                remaining_budget = max_validation_tokens - val_tokens
                if len(tokens) > remaining_budget:
                    tokens = tokens[:remaining_budget]
                val_writer.add_tokens(tokens)
                val_tokens += len(tokens)
                val_docs += 1

        if docs_seen % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            rate = (train_tokens + val_tokens) / (elapsed + 1e-6)
            print(
                f"[Pipeline] Docs seen: {docs_seen:,} (skipped: {docs_skipped:,}) | "
                f"Train: {train_tokens:,}/{max_train_tokens:,} ({train_docs:,} docs) | "
                f"Val: {val_tokens:,}/{max_validation_tokens:,} ({val_docs:,} docs) | "
                f"{rate:,.0f} tok/s"
            )

    # Finalize remaining tokens in buffers
    train_writer.finalize(keep_tail=True)
    val_writer.finalize(keep_tail=True)

    all_shards = train_writer.completed_shards + val_writer.completed_shards

    stats = {
        "total_documents_seen": docs_seen,
        "train_documents": train_docs,
        "validation_documents": val_docs,
        "skipped_documents": docs_skipped,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
    }
    return stats, all_shards


def prepare_fineweb_edu(
    output_dir: str | Path,
    dataset_repo: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    dataset_revision: str = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    max_train_tokens: int = 100_000,
    max_validation_tokens: int = 10_000,
    shard_token_target: int = 50_000,
    val_fraction: float = 0.005,
    overwrite: bool = False,
    log_interval: int = 500,
) -> dict[str, Any]:
    """Prepares FineWeb-Edu tokenized binary shards and manifest from streaming dataset."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_path}' already exists and is not empty. "
                f"Pass overwrite=True to overwrite."
            )

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer()

    print(f"Loading streaming dataset: {dataset_repo} (config={dataset_config}, rev={dataset_revision[:8]}...)")
    ds = load_dataset(
        dataset_repo,
        name=dataset_config,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )

    stats, shards = process_document_stream(
        doc_stream=ds,
        output_dir=output_path,
        tokenizer=tokenizer,
        max_train_tokens=max_train_tokens,
        max_validation_tokens=max_validation_tokens,
        shard_token_target=shard_token_target,
        val_fraction=val_fraction,
        log_interval=log_interval,
    )

    manifest = create_manifest(
        dataset_repository=dataset_repo,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        validation_fraction=val_fraction,
        shard_token_target=shard_token_target,
        stats=stats,
        shards=shards,
    )
    manifest_file = output_path / "manifest.json"
    save_manifest(manifest, manifest_file)

    print(f"\n[Pipeline Complete] Manifest written to {manifest_file}")
    print(f"  Train tokens: {stats['train_tokens']:,} across {len([s for s in shards if s['split'] == 'train'])} shard(s)")
    print(f"  Val tokens:   {stats['validation_tokens']:,} across {len([s for s in shards if s['split'] == 'validation'])} shard(s)")
    print(f"  Docs seen:    {stats['total_documents_seen']:,} (skipped {stats['skipped_documents']:,})")
    return manifest
