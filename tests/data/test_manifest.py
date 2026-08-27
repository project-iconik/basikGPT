"""Unit tests for dataset manifest serialization, checksums, and schema integrity."""

from pathlib import Path
from basikgpt.data.manifest import create_manifest, load_manifest, save_manifest


def test_manifest_creation_and_round_trip(tmp_path: Path) -> None:
    """Verifies that create_manifest populates all required metadata and saves/loads cleanly."""
    stats = {
        "total_documents_seen": 100,
        "train_documents": 95,
        "validation_documents": 4,
        "skipped_documents": 1,
        "train_tokens": 50000,
        "validation_tokens": 2000,
    }
    shards = [
        {
            "filename": "train-000000.npy",
            "split": "train",
            "token_count": 50000,
            "byte_size": 100128,
            "checksum": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
        {
            "filename": "validation-000000.npy",
            "split": "validation",
            "token_count": 2000,
            "byte_size": 4128,
            "checksum": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
        },
    ]

    manifest = create_manifest(
        dataset_repository="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        validation_fraction=0.005,
        shard_token_target=50000,
        stats=stats,
        shards=shards,
        dataset_license="ODC-By 1.0",
    )

    # Invariants
    assert manifest["format_version"] == "1.0"
    assert manifest["dataset_provenance"]["repository"] == "HuggingFaceFW/fineweb-edu"
    assert manifest["dataset_provenance"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert manifest["dataset_provenance"]["license"] == "ODC-By 1.0"
    assert manifest["tokenizer"]["vocab_size"] == 50257
    assert manifest["tokenizer"]["eot_token_id"] == 50256
    assert manifest["statistics"]["train_tokens"] == 50000
    assert manifest["statistics"]["validation_tokens"] == 2000
    assert manifest["statistics"]["train_shards"] == 1
    assert manifest["statistics"]["validation_shards"] == 1

    manifest_file = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_file)

    loaded = load_manifest(manifest_file)
    assert loaded == manifest
