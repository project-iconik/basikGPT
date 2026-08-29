"""End-to-end tokenization and sharding pipeline for HuggingFace FineWeb-Edu."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from basikgpt.data.manifest import create_manifest, save_manifest
from basikgpt.data.shard import TokenShardWriter
from basikgpt.data.split import get_document_split

if TYPE_CHECKING:
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
    text_field: str = "text",
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Processes an iterable stream of document dicts into tokenized uint16 shards.

    Args:
        doc_stream: Iterable yielding dicts with at least `text_field` and optional 'id'/'url'.
        output_dir: Directory where shards and manifest will be written.
        tokenizer: Initialized GPT2Tokenizer instance.
        max_train_tokens: Upper bound on total training tokens to collect.
        max_validation_tokens: Upper bound on total validation tokens to collect.
        shard_token_target: Target number of tokens per .npy shard.
        val_fraction: Fraction of document IDs to route to validation split.
        log_interval: Frequency (in documents seen) for progress output.
        text_field: Document dict key that holds raw text.

    Returns:
        Tuple of (statistics_dict, list_of_all_completed_shards).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_writer = TokenShardWriter(output_dir, "train", tokens_per_shard=shard_token_target)
    val_writer = TokenShardWriter(output_dir, "validation", tokens_per_shard=shard_token_target)

    docs_seen = 0
    docs_skipped = 0
    docs_skipped_for_budget = 0
    train_docs = 0
    val_docs = 0
    train_tokens = 0
    val_tokens = 0
    train_closed = False
    val_closed = False

    start_time = time.perf_counter()

    for item in doc_stream:
        if train_closed and val_closed:
            break
        if train_tokens >= max_train_tokens:
            train_closed = True
        if val_tokens >= max_validation_tokens:
            val_closed = True
        if train_closed and val_closed:
            break

        docs_seen += 1
        text = item.get(text_field, "")
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
            if train_closed:
                continue
            if train_tokens + len(tokens) > max_train_tokens:
                docs_skipped_for_budget += 1
                # A document larger than the full budget can never be stored intact;
                # skip it and keep looking. Once some tokens are stored, close the split
                # rather than scanning indefinitely for a tiny remainder filler.
                if train_tokens > 0:
                    train_closed = True
                continue
            train_writer.add_tokens(tokens)
            train_tokens += len(tokens)
            train_docs += 1
            if train_tokens >= max_train_tokens:
                train_closed = True
        else:
            if val_closed:
                continue
            if val_tokens + len(tokens) > max_validation_tokens:
                docs_skipped_for_budget += 1
                if val_tokens > 0:
                    val_closed = True
                continue
            val_writer.add_tokens(tokens)
            val_tokens += len(tokens)
            val_docs += 1
            if val_tokens >= max_validation_tokens:
                val_closed = True

        if docs_skipped_for_budget >= 10_000:
            train_closed = True
            val_closed = True

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
        "skipped_for_budget": docs_skipped_for_budget,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
    }
    return stats, all_shards


def resolve_dataset_revision(dataset_repo: str, revision: str | None) -> str:
    """Resolves a branch/tag/SHA to a pinned commit SHA via the Hugging Face Hub."""
    requested = (revision or "main").strip()
    hex_chars = set("0123456789abcdef")
    if len(requested) >= 12 and all(ch in hex_chars for ch in requested.lower()):
        return requested

    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_repo, revision=requested)
    sha = getattr(info, "sha", None) or ""
    if not sha:
        raise RuntimeError(f"Could not resolve commit SHA for {dataset_repo}@{requested}")
    return str(sha)


def prepare_hf_corpus(
    output_dir: str | Path,
    dataset_repo: str,
    dataset_config: str | None = None,
    dataset_revision: str | None = "main",
    max_train_tokens: int = 100_000,
    max_validation_tokens: int = 0,
    shard_token_target: int = 1_000_000,
    val_fraction: float = 0.0,
    overwrite: bool = False,
    log_interval: int = 500,
    text_field: str = "text",
    dataset_license: str = "ODC-By 1.0",
    selection: str = "HuggingFace streaming corpus",
) -> dict[str, Any]:
    """Streams a HuggingFace dataset, tokenizes with GPT-2 BPE, and writes uint16 shards."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_path}' already exists and is not empty. "
                f"Pass overwrite=True to overwrite."
            )

    from datasets import load_dataset
    from basikgpt.data.tokenizer import GPT2Tokenizer

    pinned_revision = resolve_dataset_revision(dataset_repo, dataset_revision)
    config_name = dataset_config or ""

    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer()

    print(
        f"Loading streaming dataset: {dataset_repo} "
        f"(config={config_name or 'default'}, rev={pinned_revision[:12]}...)"
    )
    if config_name:
        ds = load_dataset(
            dataset_repo,
            config_name,
            split="train",
            streaming=True,
            revision=pinned_revision,
        )
    else:
        ds = load_dataset(
            dataset_repo,
            split="train",
            streaming=True,
            revision=pinned_revision,
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
        text_field=text_field,
    )

    manifest = create_manifest(
        dataset_repository=dataset_repo,
        dataset_config=config_name,
        dataset_revision=pinned_revision,
        validation_fraction=val_fraction,
        shard_token_target=shard_token_target,
        stats=stats,
        shards=shards,
        dataset_license=dataset_license,
        selection=selection,
    )
    manifest_file = output_path / "manifest.json"
    save_manifest(manifest, manifest_file)

    print(f"\n[Pipeline Complete] Manifest written to {manifest_file}")
    print(f"  Train tokens: {stats['train_tokens']:,} across {len([s for s in shards if s['split'] == 'train'])} shard(s)")
    print(f"  Val tokens:   {stats['validation_tokens']:,} across {len([s for s in shards if s['split'] == 'validation'])} shard(s)")
    print(f"  Docs seen:    {stats['total_documents_seen']:,} (skipped {stats['skipped_documents']:,})")
    print(f"  Pinned revision: {pinned_revision}")
    return manifest


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
    return prepare_hf_corpus(
        output_dir=output_dir,
        dataset_repo=dataset_repo,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        max_train_tokens=max_train_tokens,
        max_validation_tokens=max_validation_tokens,
        shard_token_target=shard_token_target,
        val_fraction=val_fraction,
        overwrite=overwrite,
        log_interval=log_interval,
        text_field="text",
        dataset_license="ODC-By 1.0",
        selection="FineWeb-Edu upstream educational-quality filtering",
    )
