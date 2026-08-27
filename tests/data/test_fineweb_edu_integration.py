"""Integration test streaming a small sample directly from HuggingFace FineWeb-Edu."""

from pathlib import Path
import pytest
import torch

try:
    import datasets
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from basikgpt.data.pipeline import prepare_fineweb_edu
from basikgpt.data.shard import ShardedTokenDataset


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets library is required for FineWeb-Edu integration test")
def test_fineweb_edu_streaming_smoke(tmp_path: Path) -> None:
    """Streams a small token budget (2,000 train tokens) from FineWeb-Edu and validates the pipeline."""
    try:
        manifest = prepare_fineweb_edu(
            output_dir=tmp_path / "fineweb_edu_smoke",
            dataset_repo="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-10BT",
            dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            max_train_tokens=2_000,
            max_validation_tokens=500,
            shard_token_target=1_000,
            val_fraction=0.1,
            overwrite=True,
            log_interval=50,
        )
    except Exception as e:
        pytest.skip(f"Network error accessing HuggingFace Hub: {e}")

    assert manifest["statistics"]["train_tokens"] <= 2_000
    assert manifest["statistics"]["train_tokens"] > 0
    assert manifest["statistics"]["validation_tokens"] <= 500

    out_dir = tmp_path / "fineweb_edu_smoke"
    train_shards = sorted(out_dir.glob("train-*.npy"))
    assert len(train_shards) >= 1

    # Load with ShardedTokenDataset
    dataset = ShardedTokenDataset(train_shards, context_length=64)
    assert len(dataset) > 0

    inp, tgt = dataset[0]
    assert inp.shape == (64,)
    assert tgt.shape == (64,)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long
    assert torch.equal(inp[1:], tgt[:-1])
