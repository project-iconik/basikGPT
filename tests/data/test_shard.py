"""Unit tests for token shard writer and ShardedTokenDataset reader."""

from pathlib import Path
import numpy as np
import pytest
import torch

from basikgpt.data.shard import ShardedTokenDataset, TokenShardWriter


def test_shard_writer_and_uint16_round_trip(tmp_path: Path) -> None:
    """Verifies that TokenShardWriter writes uint16 .npy files and produces valid metadata/checksums."""
    tokens_per_shard = 100
    writer = TokenShardWriter(tmp_path, "train", tokens_per_shard=tokens_per_shard)

    # Add 250 tokens -> should create 2 full shards (100 each) and keep 50 in buffer
    tokens = list(range(250))
    shards_1 = writer.add_tokens(tokens)
    assert len(shards_1) == 2
    assert shards_1[0]["filename"] == "train-000000.npy"
    assert shards_1[0]["token_count"] == 100
    assert shards_1[1]["filename"] == "train-000001.npy"
    assert shards_1[1]["token_count"] == 100

    # Finalize to flush the remaining 50 tokens
    shards_2 = writer.finalize(keep_tail=True)
    assert len(shards_2) == 1
    assert shards_2[0]["filename"] == "train-000002.npy"
    assert shards_2[0]["token_count"] == 50

    # Total completed shards
    all_shards = writer.completed_shards
    assert len(all_shards) == 3

    # Verify physical file existence, uint16 dtype, and exact data equality
    file0 = tmp_path / "train-000000.npy"
    assert file0.exists()
    arr0 = np.load(file0)
    assert arr0.dtype == np.uint16
    assert np.array_equal(arr0, np.array(list(range(100)), dtype=np.uint16))


def test_token_range_validation(tmp_path: Path) -> None:
    """Verifies that token IDs outside [0, 50256] raise ValueError before serialization."""
    writer = TokenShardWriter(tmp_path, "train", tokens_per_shard=10)

    # Negative token
    with pytest.raises(ValueError, match="Token values out of GPT-2 vocabulary range"):
        writer.add_tokens([-1, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Out-of-bounds token > 50256
    with pytest.raises(ValueError, match="Token values out of GPT-2 vocabulary range"):
        writer.add_tokens([50257, 10, 20, 30, 40, 50, 60, 70, 80, 90])


def test_sharded_token_dataset_causal_alignment(tmp_path: Path) -> None:
    """Verifies ShardedTokenDataset sequence sampling and strict next-token alignment."""
    # Write a shard with known sequence: 0, 1, 2, ..., 99
    shard_path = tmp_path / "train-000000.npy"
    data = np.arange(100, dtype=np.uint16)
    np.save(shard_path, data)

    context_length = 8
    stride = 8
    dataset = ShardedTokenDataset(shard_path, context_length=context_length, stride=stride)

    # 100 tokens with T=8: sample needs 9 tokens (0..8, 8..16, ... 88..96). (100 - 1) // 8 = 12 samples
    assert len(dataset) == 12

    # Verify sample 0
    input_ids_0, targets_0 = dataset[0]
    assert input_ids_0.shape == (8,)
    assert targets_0.shape == (8,)
    assert input_ids_0.dtype == torch.long
    assert targets_0.dtype == torch.long

    # Values check
    assert torch.equal(input_ids_0, torch.arange(0, 8, dtype=torch.long))
    assert torch.equal(targets_0, torch.arange(1, 9, dtype=torch.long))

    # Causal Next-Token Alignment Invariant: input_ids[1:] == targets[:-1]
    assert torch.equal(input_ids_0[1:], targets_0[:-1])

    # Verify sample 1
    input_ids_1, targets_1 = dataset[1]
    assert torch.equal(input_ids_1, torch.arange(8, 16, dtype=torch.long))
    assert torch.equal(targets_1, torch.arange(9, 17, dtype=torch.long))
    assert torch.equal(input_ids_1[1:], targets_1[:-1])


def test_multi_shard_continuous_reading(tmp_path: Path) -> None:
    """Verifies dataset reads across multiple contiguous shard files."""
    # Create two shards of 20 tokens each
    arr1 = np.arange(0, 20, dtype=np.uint16)
    arr2 = np.arange(20, 40, dtype=np.uint16)
    np.save(tmp_path / "train-000000.npy", arr1)
    np.save(tmp_path / "train-000001.npy", arr2)

    context_length = 4
    stride = 4
    dataset = ShardedTokenDataset(tmp_path, context_length=context_length, stride=stride)

    # Each shard has (20 - 1) // 4 = 4 samples. Total = 8 samples.
    assert len(dataset) == 8

    # Last sample in shard 0 (idx 3): tokens 12..16, targets 13..17
    inp_3, tgt_3 = dataset[3]
    assert torch.equal(inp_3, torch.tensor([12, 13, 14, 15], dtype=torch.long))
    assert torch.equal(tgt_3, torch.tensor([13, 14, 15, 16], dtype=torch.long))

    # First sample in shard 1 (idx 4): tokens 20..24, targets 21..25
    inp_4, tgt_4 = dataset[4]
    assert torch.equal(inp_4, torch.tensor([20, 21, 22, 23], dtype=torch.long))
    assert torch.equal(tgt_4, torch.tensor([21, 22, 23, 24], dtype=torch.long))
