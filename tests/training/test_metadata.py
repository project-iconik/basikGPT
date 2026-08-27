"""Unit tests for run provenance metadata, config serialization, and summary generation."""

from dataclasses import asdict
from pathlib import Path
from basikgpt.config import GPTConfig
from basikgpt.training.config import TrainingConfig
from basikgpt.training.metadata import (
    RUN_FORMAT_VERSION,
    atomic_save_json,
    extract_dataset_provenance,
    load_json,
    save_run_metadata,
    save_run_summary,
)
from basikgpt.training.reproducibility import get_git_metadata, get_system_metadata


def test_atomic_save_and_load_json(tmp_path: Path) -> None:
    """Verifies atomic write and load of dictionary data."""
    data = {"name": "basikgpt", "step": 100, "scores": [1.0, 2.5, 3.8]}
    file_path = tmp_path / "test.json"

    saved_path = atomic_save_json(file_path, data)
    assert saved_path.exists()

    loaded = load_json(saved_path)
    assert loaded == data


def test_config_serialization_roundtrip(tmp_path: Path) -> None:
    """Verifies that GPTConfig and TrainingConfig can be serialized and parsed back into dictionaries."""
    gpt_cfg = GPTConfig.gpt2_small(context_length=512)
    train_cfg = TrainingConfig(learning_rate=3e-4, seed=42)

    gpt_path = atomic_save_json(tmp_path / "model_config.json", asdict(gpt_cfg))
    train_path = atomic_save_json(tmp_path / "training_config.json", asdict(train_cfg))

    loaded_gpt = load_json(gpt_path)
    loaded_train = load_json(train_path)

    assert loaded_gpt["vocab_size"] == 50257
    assert loaded_gpt["context_length"] == 512
    assert loaded_train["learning_rate"] == 3e-4
    assert loaded_train["seed"] == 42


def test_save_run_metadata_and_summary(tmp_path: Path) -> None:
    """Verifies that save_run_metadata and save_run_summary produce complete structured records."""
    gpt_cfg = GPTConfig(vocab_size=128, context_length=16, n_layers=1, n_heads=2, d_model=32, d_ff=128)
    train_cfg = TrainingConfig(learning_rate=1e-3, seed=999)
    manifest_mock = {
        "provenance": {"dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"},
        "token_statistics": {"train_tokens": 50000, "validation_tokens": 5000},
    }

    run_json_path = save_run_metadata(
        output_dir=tmp_path,
        run_name="test_experiment_01",
        model_config=gpt_cfg,
        training_config=train_cfg,
        dataset_manifest=manifest_mock,
        dataset_manifest_path=tmp_path / "mock_manifest.json",
    )

    assert run_json_path.exists()
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "training_config.json").exists()
    assert (tmp_path / "dataset.json").exists()

    run_meta = load_json(run_json_path)
    assert run_meta["run_format_version"] == RUN_FORMAT_VERSION
    assert run_meta["run_name"] == "test_experiment_01"
    assert run_meta["seed"] == 999
    assert run_meta["train_tokens"] == 50000
    assert "git" in run_meta
    assert "system" in run_meta

    # Save summary
    summary_path = save_run_summary(
        output_dir=tmp_path,
        status="completed",
        final_step=50,
        tokens_seen=12800,
        elapsed_seconds=12.5,
        final_train_loss=2.45,
        final_val_loss=2.40,
        best_val_loss=2.38,
        checkpoint_path=tmp_path / "step-final.pt",
    )

    assert summary_path.exists()
    summary = load_json(summary_path)
    assert summary["status"] == "completed"
    assert summary["final_step"] == 50
    assert summary["tokens_seen"] == 12800
    assert summary["final_train_loss"] == 2.45
    assert summary["best_val_loss"] == 2.38


def test_extract_dataset_provenance_canonical_and_legacy_schemas() -> None:
    """Verifies provenance helper reads canonical manifest keys and legacy aliases."""
    canonical = {
        "dataset_provenance": {
            "repository": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        },
        "statistics": {"train_tokens": 50000, "validation_tokens": 5000},
        "tokenizer": {"encoding": "gpt2", "eot_token_id": 50256},
    }
    extracted = extract_dataset_provenance(canonical)
    assert extracted["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert extracted["repository"] == "HuggingFaceFW/fineweb-edu"
    assert extracted["config"] == "sample-10BT"
    assert extracted["train_tokens"] == 50000
    assert extracted["tokenizer_encoding"] == "gpt2"

    legacy = {
        "provenance": {"dataset_revision": "legacy-rev", "dataset_repository": "legacy-repo"},
        "token_statistics": {"train_tokens": 10, "validation_tokens": 1},
    }
    extracted_legacy = extract_dataset_provenance(legacy)
    assert extracted_legacy["revision"] == "legacy-rev"
    assert extracted_legacy["repository"] == "legacy-repo"
    assert extracted_legacy["train_tokens"] == 10


def test_git_and_system_metadata_helpers() -> None:
    """Verifies that get_git_metadata and get_system_metadata return well-formed dictionary structures."""
    git_meta = get_git_metadata()
    assert "git_commit" in git_meta
    assert "git_dirty" in git_meta

    sys_meta = get_system_metadata()
    assert "platform" in sys_meta
    assert "python_version" in sys_meta
    assert "pytorch_version" in sys_meta
    assert "numpy_version" in sys_meta
    assert "cuda_available" in sys_meta
    assert "gpu_count" in sys_meta
    assert "gpu_name" in sys_meta
    assert "compute_capability" in sys_meta
    assert "total_vram_bytes" in sys_meta
    assert "bf16_supported" in sys_meta
    assert "nvidia_driver" in sys_meta
    assert "cuda_runtime" in sys_meta
    assert "cloud_provider" in sys_meta
