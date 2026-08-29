"""Unit tests for Milestone 16 candidate planning, canonical config, and sample-index resume."""

from pathlib import Path
import json
import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.config_pilot import (
    CANDIDATE_A,
    CANDIDATE_B,
    COMPARISON_TOKEN_BATCH,
    MAIN_TOKEN_BUDGET,
    PILOT_10M_TOKENS,
    PILOT_1M_TOKENS,
    assert_equal_token_batch,
    canonical_config_dict,
    checkpoint_steps_for_pilot,
    eval_interval_for_pilot,
    load_canonical_config,
    main_run_logging_protocol,
    plan_pilot_stage,
    save_canonical_config,
)
from basikgpt.training.trainer import Trainer


def _tiny_gpt() -> GPT:
    cfg = GPTConfig(
        vocab_size=32,
        context_length=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    return GPT(cfg)


def test_candidates_share_token_batch() -> None:
    assert_equal_token_batch()
    assert CANDIDATE_A.tokens_per_optimizer_step == COMPARISON_TOKEN_BATCH
    assert CANDIDATE_B.tokens_per_optimizer_step == COMPARISON_TOKEN_BATCH
    assert CANDIDATE_A.compile is False
    assert CANDIDATE_B.compile is True
    assert CANDIDATE_A.micro_batch_size == 8
    assert CANDIDATE_A.grad_accum_steps == 8
    assert CANDIDATE_B.micro_batch_size == 16
    assert CANDIDATE_B.grad_accum_steps == 4


def test_1m_and_10m_step_planning() -> None:
    plan_1m_a = plan_pilot_stage(CANDIDATE_A, PILOT_1M_TOKENS)
    plan_1m_b = plan_pilot_stage(CANDIDATE_B, PILOT_1M_TOKENS)
    assert plan_1m_a["plan"]["optimizer_steps"] == plan_1m_b["plan"]["optimizer_steps"] == 16
    assert plan_1m_a["plan"]["actual_token_budget"] == 1_048_576
    assert plan_1m_a["plan"]["overshoot_tokens"] == 48_576
    assert plan_1m_a["warmup_steps"] == plan_1m_b["warmup_steps"] == 2
    assert plan_1m_a["eval_batches"] == 16
    assert plan_1m_b["eval_batches"] == 8

    plan_10m_a = plan_pilot_stage(CANDIDATE_A, PILOT_10M_TOKENS)
    plan_10m_b = plan_pilot_stage(CANDIDATE_B, PILOT_10M_TOKENS)
    assert plan_10m_a["plan"]["optimizer_steps"] == plan_10m_b["plan"]["optimizer_steps"] == 153
    assert plan_10m_a["plan"]["actual_token_budget"] == 10_027_008
    assert plan_10m_a["plan"]["overshoot_tokens"] == 27_008
    assert plan_10m_a["warmup_steps"] == 15
    assert checkpoint_steps_for_pilot(16) == [8, 16]
    assert checkpoint_steps_for_pilot(153) == [38, 76, 115, 153]
    assert eval_interval_for_pilot(16) == 8
    assert eval_interval_for_pilot(153) == 38


def test_2_5b_planning_for_65536_token_batch() -> None:
    plan = CANDIDATE_A.plan(MAIN_TOKEN_BUDGET)
    assert plan.tokens_per_optimizer_step == COMPARISON_TOKEN_BATCH
    assert plan.optimizer_steps == 38_147
    assert plan.actual_token_budget == 2_500_001_792
    assert plan.overshoot_tokens == 1_792
    assert CANDIDATE_B.plan(MAIN_TOKEN_BUDGET).optimizer_steps == plan.optimizer_steps


def test_canonical_config_roundtrip(tmp_path: Path) -> None:
    payload = canonical_config_dict(CANDIDATE_A)
    path = tmp_path / "gpt2_small_fineweb_edu_single_gpu.json"
    save_canonical_config(path, payload)
    loaded = load_canonical_config(path)
    assert loaded["precision"] == "bf16"
    assert loaded["compile"] is False
    assert loaded["micro_batch_size"] == 8
    assert loaded["grad_accum_steps"] == 8
    assert loaded["tokens_per_optimizer_step"] == COMPARISON_TOKEN_BATCH
    assert loaded["main_plan"]["optimizer_steps"] == 38_147
    protocol = loaded["logging_protocol"]
    assert protocol["eval_at_start"] is True
    assert protocol["eval_tokens"] == 131_072
    assert protocol["checkpoint_steps"] == [1526, 7630, 15259, 38147]
    assert protocol["eval_interval_steps"] == 1526
    assert protocol["do_not_use_train_py_interval_defaults"] is True


def test_main_run_logging_protocol_matches_token_milestones() -> None:
    protocol = main_run_logging_protocol(CANDIDATE_A)
    plan = CANDIDATE_A.plan(MAIN_TOKEN_BUDGET)
    assert protocol["checkpoint_steps"][-1] == plan.optimizer_steps
    assert protocol["checkpoint_steps"] == main_run_logging_protocol(CANDIDATE_B)["checkpoint_steps"]
    assert "evaluate_hellaswag.py" in protocol["downstream_eval"]["evaluate_hellaswag_py"]
    assert "--eval-at-start" in protocol["cli_flags_example"]


def test_committed_canonical_config_includes_logging_protocol() -> None:
    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "gpt2_small_fineweb_edu_single_gpu.json"
    loaded = load_canonical_config(repo_cfg)
    assert loaded["logging_protocol"]["checkpoint_steps"] == main_run_logging_protocol(CANDIDATE_A)["checkpoint_steps"]
    assert loaded["logging_protocol"]["eval_at_start"] is True


def test_data_sample_index_resume(tmp_path: Path) -> None:
    """Sequential sample index is stored and used to continue without DataLoader shuffle."""
    torch.manual_seed(0)
    raw = torch.arange(64, dtype=torch.long).reshape(8, 8)
    # Distinct rows so skipped samples would change the next batch if resume ignored the index.
    ds = TensorDataset(raw, raw)
    model = _tiny_gpt()
    cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        stop_at_step=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        checkpoint_interval=1,
        log_interval=1,
        track_data_sample_index=True,
        save_step_final=False,
        output_dir=str(tmp_path / "first"),
        seed=1337,
    )
    trainer = Trainer(model, cfg, DataLoader(ds, batch_size=2, shuffle=False), overwrite=True)
    trainer.train()
    assert trainer.global_step == 1
    assert trainer.data_sample_index == 2
    ckpt = tmp_path / "first" / "step-00000001.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["extra_state"]["data_sample_index"] == 2
    assert payload["extra_state"]["resume_class"] == "exact-sample-index"

    resume_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        checkpoint_interval=1,
        log_interval=1,
        track_data_sample_index=True,
        save_step_final=False,
        output_dir=str(tmp_path / "first"),
        seed=1337,
    )
    resumed = Trainer(
        _tiny_gpt(),
        resume_cfg,
        DataLoader(ds, batch_size=2, shuffle=False),
        resume_from=ckpt,
    )
    resumed.train(resume_from=ckpt)
    assert resumed.global_step == 2
    assert resumed.data_sample_index == 4
    meta = load_checkpoint(tmp_path / "first" / "step-00000002.pt", _tiny_gpt())
    assert meta["extra_state"]["data_sample_index"] == 4


def test_reset_data_sample_index_on_resume(tmp_path: Path) -> None:
    """New-mix resume discards the checkpoint sample index and starts at 0."""
    torch.manual_seed(0)
    raw = torch.arange(64, dtype=torch.long).reshape(8, 8)
    ds = TensorDataset(raw, raw)
    cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        stop_at_step=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        checkpoint_interval=1,
        log_interval=1,
        track_data_sample_index=True,
        save_step_final=False,
        output_dir=str(tmp_path / "first"),
        seed=1337,
    )
    trainer = Trainer(_tiny_gpt(), cfg, DataLoader(ds, batch_size=2, shuffle=False), overwrite=True)
    trainer.train()
    ckpt = tmp_path / "first" / "step-00000001.pt"
    resume_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        checkpoint_interval=1,
        log_interval=1,
        track_data_sample_index=True,
        save_step_final=False,
        output_dir=str(tmp_path / "second"),
        seed=1337,
    )
    resumed = Trainer(
        _tiny_gpt(),
        resume_cfg,
        DataLoader(ds, batch_size=2, shuffle=False),
        resume_from=ckpt,
        reset_data_sample_index=True,
    )
    resumed.train(resume_from=ckpt)
    assert resumed.global_step == 2
    assert resumed.data_sample_index == 2


def test_save_step_final_can_be_disabled(tmp_path: Path) -> None:
    torch.manual_seed(0)
    raw = torch.randint(0, 32, (4, 8), dtype=torch.long)
    ds = TensorDataset(raw, raw)
    cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        checkpoint_interval=2,
        log_interval=1,
        save_step_final=False,
        output_dir=str(tmp_path),
        seed=1,
    )
    Trainer(_tiny_gpt(), cfg, DataLoader(ds, batch_size=2), overwrite=True).train()
    assert (tmp_path / "step-00000002.pt").exists()
    assert not (tmp_path / "step-final.pt").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert "training_only_tokens_per_sec" in summary


def test_eval_at_start_logs_step_zero(tmp_path: Path) -> None:
    torch.manual_seed(0)
    raw = torch.randint(0, 32, (4, 8), dtype=torch.long)
    ds = TensorDataset(raw, raw)
    cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        eval_interval=10_000,
        eval_batches=1,
        checkpoint_interval=10_000,
        log_interval=1,
        eval_at_start=True,
        save_step_final=True,
        output_dir=str(tmp_path),
        seed=1,
    )
    Trainer(
        _tiny_gpt(),
        cfg,
        DataLoader(ds, batch_size=2),
        DataLoader(ds, batch_size=2),
        overwrite=True,
    ).train()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    val_steps = [r["step"] for r in records if r.get("type") == "val"]
    assert 0 in val_steps
    assert all(math.isfinite(r["val_loss"]) for r in records if r.get("type") == "val")
    assert all("val_perplexity" in r for r in records if r.get("type") == "val")


def test_uncompiled_checkpoint_has_no_compile_keys(tmp_path: Path) -> None:
    torch.manual_seed(0)
    raw = torch.randint(0, 32, (4, 8), dtype=torch.long)
    ds = TensorDataset(raw, raw)
    cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=1,
        batch_size=2,
        checkpoint_interval=1,
        log_interval=1,
        eval_interval=10_000,
        save_step_final=False,
        output_dir=str(tmp_path),
    )
    Trainer(_tiny_gpt(), cfg, DataLoader(ds, batch_size=2), overwrite=True).train()
    payload = torch.load(tmp_path / "step-00000001.pt", map_location="cpu", weights_only=False)
    assert not any("_orig_mod" in key for key in payload["model_state_dict"])
    fresh = _tiny_gpt()
    load_checkpoint(tmp_path / "step-00000001.pt", fresh)
    assert fresh.wte.weight is fresh.lm_head.weight
