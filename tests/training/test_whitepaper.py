"""Unit tests for whitepaper snapshot helpers and HellaSwag default JSON path."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.config import TrainingConfig
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.whitepaper import (
    build_whitepaper_snapshot,
    collect_static_run_extra,
    extract_curve_extrema,
    format_whitepaper_markdown,
    resolve_hellaswag_output_json,
    uniform_ce_reference,
    write_whitepaper_snapshot,
)


def test_resolve_hellaswag_output_json_defaults(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-00000100.pt"
    assert resolve_hellaswag_output_json(None, checkpoint) == tmp_path / "hellaswag.json"
    custom = tmp_path / "custom.json"
    assert resolve_hellaswag_output_json(custom, checkpoint) == custom
    assert resolve_hellaswag_output_json(None, None) == Path("hellaswag.json")


def test_uniform_ce_reference_matches_ln_vocab() -> None:
    assert uniform_ce_reference(32) == pytest.approx(math.log(32))
    assert uniform_ce_reference(50257) == pytest.approx(math.log(50257))
    with pytest.raises(ValueError):
        uniform_ce_reference(0)


def test_extract_curve_extrema_from_jsonl(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"type": "val", "step": 0, "val_loss": 4.0, "val_perplexity": math.exp(4.0)},
        {"type": "train", "step": 1, "loss": 3.9, "peak_allocated_vram_bytes": 100},
        {"type": "train", "step": 10, "loss": 3.1, "peak_allocated_vram_bytes": 250},
        {"type": "val", "step": 10, "val_loss": 3.2, "val_perplexity": math.exp(3.2)},
        {"type": "train", "step": 20, "loss": 2.8, "peak_allocated_vram_bytes": 200},
        {"type": "val", "step": 20, "val_loss": 3.3, "val_perplexity": math.exp(3.3)},
    ]
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    extrema = extract_curve_extrema(metrics)
    assert extrema["first_train_step"] == 1
    assert extrema["first_train_loss"] == pytest.approx(3.9)
    assert extrema["last_train_step"] == 20
    assert extrema["last_train_loss"] == pytest.approx(2.8)
    assert extrema["min_val_step"] == 10
    assert extrema["min_val_loss"] == pytest.approx(3.2)
    assert extrema["min_val_perplexity"] == pytest.approx(math.exp(3.2))
    assert extrema["peak_allocated_vram_bytes"] == 250


def test_collect_static_run_extra_and_caller_token_budget_wins() -> None:
    cfg = GPTConfig(vocab_size=32, context_length=8, n_layers=1, n_heads=2, d_model=16, d_ff=64)
    model = GPT(cfg)
    train_cfg = TrainingConfig(
        max_steps=10,
        warmup_steps=0,
        batch_size=2,
        gradient_accumulation_steps=2,
        eval_batches=3,
    )
    optimizer = configure_optimizers(model, train_cfg)
    raw = torch.randint(0, 32, (4, 9), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw[:, :-1], raw[:, 1:]), batch_size=2)
    extra = collect_static_run_extra(
        model=model,
        config=train_cfg,
        optimizer=optimizer,
        parameter_count=model.num_parameters(),
        tokens_per_optimizer_step=32,
        train_loader=loader,
        extra_metadata={"token_budget": {"requested_token_budget": 100, "overshoot_tokens": 4}},
    )
    assert extra["token_budget"]["requested_token_budget"] == 100
    assert extra["token_budget"]["overshoot_tokens"] == 4
    assert extra["eval_tokens"] == 2 * 8 * 3
    assert extra["tie_word_embeddings"] is True
    assert extra["packed_data"]["train_sequences"] == 4
    assert extra["parameter_breakdown"]["head_dim"] == 8


def test_write_whitepaper_snapshot_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": "demo",
                "git": {"git_commit": "abc", "git_dirty": False},
                "system": {"gpu_name": "NVIDIA RTX PRO 4500 Blackwell", "pytorch_version": "2.8.0"},
                "extra": {
                    "parameter_count": 1000,
                    "eval_tokens": 128,
                    "uniform_ce_reference": math.log(32),
                    "tokens_per_optimizer_step": 16,
                    "tie_word_embeddings": True,
                    "head_dim": 8,
                    "packed_data": {"train_sequences": 10, "train_discarded_tail_tokens": 3},
                    "token_budget": {"requested_token_budget": 64, "actual_token_budget": 80, "overshoot_tokens": 16},
                    "parameter_breakdown": {"token_embedding": 512, "position_embedding": 128},
                    "optimizer_param_groups": {"decay_parameters": 900, "no_decay_parameters": 100},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "tokens_seen": 80,
                "elapsed_seconds": 3600.0,
                "gpu_hours": 1.0,
                "parameter_count": 1000,
                "tokens_per_parameter": 0.08,
                "first_train_loss": 3.5,
                "first_train_step": 1,
                "last_train_loss": 2.1,
                "last_train_step": 5,
                "min_val_loss": 2.2,
                "min_val_perplexity": math.exp(2.2),
                "min_val_step": 5,
                "end_to_end_tokens_per_sec": 20.0,
                "peak_allocated_vram_mib": 12.5,
                "uniform_ce_reference": math.log(32),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model_config.json").write_text(
        json.dumps({"vocab_size": 32, "context_length": 8, "n_layers": 1, "n_heads": 2, "d_model": 16, "d_ff": 64}),
        encoding="utf-8",
    )
    (run_dir / "training_config.json").write_text(
        json.dumps({"max_steps": 5, "batch_size": 2, "learning_rate": 0.001, "seed": 1337, "precision": "bf16"}),
        encoding="utf-8",
    )
    (run_dir / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_provenance": {
                    "repository": "HuggingFaceFW/fineweb-edu",
                    "config": "sample-10BT",
                    "revision": "abc",
                    "license": "ODC-By 1.0",
                },
                "statistics": {"train_tokens": 100, "validation_tokens": 10, "train_shards": 1, "validation_shards": 1},
                "tokenizer": {"encoding": "gpt2", "vocab_size": 50257},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "hellaswag.json").write_text(
        json.dumps({"norm_accuracy": 0.31, "raw_accuracy": 0.28, "num_examples": 10}),
        encoding="utf-8",
    )
    (run_dir / "evaluation.json").write_text(
        json.dumps({"metrics": {"validation_loss": 2.15, "perplexity": math.exp(2.15)}}),
        encoding="utf-8",
    )

    written = write_whitepaper_snapshot(run_dir)
    assert written["json"].exists()
    assert written["markdown"].exists()
    snapshot = json.loads(written["json"].read_text(encoding="utf-8"))
    assert snapshot["abstract"]["parameter_count"] == 1000
    assert snapshot["abstract"]["hellaswag_norm_accuracy"] == pytest.approx(0.31)
    assert snapshot["abstract"]["full_validation_loss"] == pytest.approx(2.15)
    assert snapshot["language_model_results"]["first_train_step"] == 1
    assert snapshot["data"]["repository"] == "HuggingFaceFW/fineweb-edu"
    markdown = written["markdown"].read_text(encoding="utf-8")
    assert "1,000" in markdown
    assert "HellaSwag acc_norm" in markdown
    rebuilt = build_whitepaper_snapshot(run_dir)
    assert "Unique parameters" in format_whitepaper_markdown(rebuilt)
