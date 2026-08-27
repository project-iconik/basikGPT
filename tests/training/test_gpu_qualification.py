"""CPU-safe tests for GPU qualification helpers and summary schema."""

from pathlib import Path

from basikgpt.training.gpu_qualification import (
    collect_gpu_environment,
    largest_passing_batch,
    save_gpu_qualification_summary,
)
from basikgpt.training.metadata import load_json


def test_largest_passing_batch() -> None:
    rows = [
        {"micro_batch_size": 1, "status": "PASS"},
        {"micro_batch_size": 2, "status": "PASS"},
        {"micro_batch_size": 4, "status": "OOM"},
    ]
    assert largest_passing_batch(rows) == 2
    assert largest_passing_batch([{"micro_batch_size": 1, "status": "OOM"}]) is None


def test_gpu_qualification_summary_schema(tmp_path: Path) -> None:
    payload = {
        "status": "passed",
        "provider": "RunPod",
        "gpu": "example",
        "precision": "bf16",
        "model": "gpt2-small",
        "parameter_count": 124439808,
        "context_length": 1024,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "tokens_per_optimizer_step": 2048,
        "tokens_per_second": 1.0,
        "peak_allocated_vram_bytes": 1,
        "peak_reserved_vram_bytes": 2,
        "initial_loss": 10.8,
        "final_loss": 10.7,
        "validation_loss": 10.75,
        "checkpoint_resume_verified": True,
    }
    path = save_gpu_qualification_summary(tmp_path / "gpu_qualification.json", payload)
    loaded = load_json(path)
    assert loaded["parameter_count"] == 124439808
    assert loaded["checkpoint_resume_verified"] is True


def test_collect_gpu_environment_has_no_secret_keys() -> None:
    payload = collect_gpu_environment()
    serialized = str(payload).lower()
    for forbidden in ("api_key", "hf_token", "huggingface_hub_token", "password", "ssh-rsa", "begin rsa"):
        assert forbidden not in serialized
    assert "git" in payload
    assert "cuda" in payload
    assert "system" in payload
    assert "git_commit" in payload["git"]
    assert "cuda_available" in payload["cuda"]
