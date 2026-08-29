"""Unit tests for the end-to-end data pipeline on synthetic document streams (offline, zero-network)."""

from pathlib import Path
from basikgpt.data.pipeline import process_document_stream
from basikgpt.data.shard import ShardedTokenDataset
from basikgpt.data.tokenizer import GPT2Tokenizer
import numpy as np


def synthetic_doc_generator(n_docs: int = 50):
    """Generates synthetic documents with stable IDs for offline pipeline testing."""
    for i in range(n_docs):
        if i == 5:
            # Yield empty document to test skipping
            yield {"id": f"doc_{i}", "text": "   "}
        else:
            yield {
                "id": f"doc_{i}",
                "url": f"https://example.com/doc/{i}",
                "text": f"Document index {i}: Educational content about science and technology. " * 5,
            }


def test_process_document_stream_offline(tmp_path: Path) -> None:
    """Verifies end-to-end pipeline execution with token truncation, sharding, and dataset loading."""
    tokenizer = GPT2Tokenizer()
    max_train_tokens = 500
    max_val_tokens = 100
    shard_token_target = 200

    stats, shards = process_document_stream(
        doc_stream=synthetic_doc_generator(n_docs=50),
        output_dir=tmp_path,
        tokenizer=tokenizer,
        max_train_tokens=max_train_tokens,
        max_validation_tokens=max_val_tokens,
        shard_token_target=shard_token_target,
        val_fraction=0.1,
    )

    # Invariants: complete documents only, never exceed budget
    assert stats["train_tokens"] <= max_train_tokens
    assert stats["train_tokens"] > 0
    assert stats["validation_tokens"] <= max_val_tokens
    assert stats["skipped_documents"] >= 1
    assert stats["skipped_for_budget"] >= 0

    # Check shard files created
    train_shards = [s for s in shards if s["split"] == "train"]
    assert len(train_shards) >= 2

    # Load dataset from generated train shards
    dataset = ShardedTokenDataset(tmp_path / "train-000000.npy", context_length=8)
    assert len(dataset) > 0
    inp, tgt = dataset[0]
    assert inp.shape == (8,)
    assert tgt.shape == (8,)
    assert (inp[1:] == tgt[:-1]).all()

    train_tokens = np.concatenate(
        [np.load(tmp_path / s["filename"]) for s in shards if s["split"] == "train"]
    )
    assert int(train_tokens[-1]) == tokenizer.eot_token_id


def test_process_document_stream_skips_partial_documents(tmp_path: Path) -> None:
    """Verifies documents that do not fit the remaining token budget are skipped whole, keeping EOT."""
    tokenizer = GPT2Tokenizer()

    def docs():
        yield {"id": "doc_a", "text": "short educational sentence."}
        yield {"id": "doc_b", "text": "This is a much longer educational document. " * 40}

    stats, shards = process_document_stream(
        doc_stream=docs(),
        output_dir=tmp_path,
        tokenizer=tokenizer,
        max_train_tokens=30,
        max_validation_tokens=0,
        shard_token_target=1000,
        val_fraction=0.0,
    )

    assert stats["train_tokens"] <= 30
    assert stats["skipped_for_budget"] >= 1
    train_shards = [s for s in shards if s["split"] == "train"]
    assert train_shards
    tokens = np.load(tmp_path / train_shards[0]["filename"])
    assert int(tokens[-1]) == tokenizer.eot_token_id
    assert tokenizer.eot_token_id in set(int(t) for t in tokens)


def test_process_document_stream_text_field(tmp_path: Path) -> None:
    """Documents can store text under a non-default field name."""
    tokenizer = GPT2Tokenizer()

    def docs():
        yield {"id": "doc_a", "body": "short educational sentence about science."}

    stats, shards = process_document_stream(
        doc_stream=docs(),
        output_dir=tmp_path,
        tokenizer=tokenizer,
        max_train_tokens=100,
        max_validation_tokens=0,
        shard_token_target=1000,
        val_fraction=0.0,
        text_field="body",
    )
    assert stats["train_tokens"] > 0
    assert [s for s in shards if s["split"] == "train"]
