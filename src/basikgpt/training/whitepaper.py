"""Whitepaper snapshot fields: static run extras, curve extrema, and report rendering."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basikgpt.config import GPTConfig
from basikgpt.training.config import TrainingConfig
from basikgpt.training.metadata import atomic_save_json, load_json, perplexity_from_loss


SECONDS_PER_HOUR = 3600.0
BYTES_PER_MIB = 1024 * 1024


def uniform_ce_reference(vocab_size: int) -> float:
    """Cross-entropy of a uniform next-token predictor: ln(V)."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return math.log(vocab_size)


def eval_tokens_covered(batch_size: int, context_length: int, eval_batches: int) -> int:
    """Intended in-loop validation token count: B * T * eval_batches."""
    return int(batch_size) * int(context_length) * int(eval_batches)


def model_ties_embeddings(model: nn.Module) -> bool | None:
    """True when lm_head.weight is the token-embedding tensor; None if either is missing."""
    if not (hasattr(model, "lm_head") and hasattr(model, "wte")):
        return None
    lm_head = getattr(model, "lm_head", None)
    wte = getattr(model, "wte", None)
    lm_weight = getattr(lm_head, "weight", None)
    wte_weight = getattr(wte, "weight", None)
    if lm_weight is None or wte_weight is None:
        return None
    return lm_weight is wte_weight


def parameter_breakdown(model: nn.Module) -> dict[str, Any]:
    """Analytical GPT-2 parameter split plus the measured unique count."""
    payload: dict[str, Any] = {}
    cfg = getattr(model, "config", None)
    if isinstance(cfg, GPTConfig):
        payload.update(
            {
                "token_embedding": cfg.vocab_size * cfg.d_model,
                "position_embedding": cfg.context_length * cfg.d_model,
                "embeddings": cfg.num_embedding_parameters(),
                "transformer_blocks_and_final_ln": cfg.num_transformer_parameters(),
                "unique_total_tied": cfg.num_total_parameters(tied_weights=True),
                "unique_total_untied": cfg.num_total_parameters(tied_weights=False),
                "head_dim": cfg.head_dim,
                "vocab_size": cfg.vocab_size,
            }
        )
    if hasattr(model, "num_parameters"):
        payload["measured_unique"] = int(model.num_parameters())
    else:
        payload["measured_unique"] = sum(p.numel() for p in model.parameters())
    return payload


def optimizer_param_group_counts(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Counts unique parameters in the decay (group 0) and no-decay (group 1) AdamW groups."""
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        params = list(group.get("params") or [])
        groups.append(
            {
                "weight_decay": group.get("weight_decay"),
                "tensors": len(params),
                "parameters": int(sum(p.numel() for p in params)),
            }
        )
    decay_parameters = groups[0]["parameters"] if groups else 0
    no_decay_parameters = groups[1]["parameters"] if len(groups) > 1 else 0
    return {
        "groups": groups,
        "decay_parameters": decay_parameters,
        "no_decay_parameters": no_decay_parameters,
    }


def packed_split_stats(dataset: Any, split: str) -> dict[str, Any]:
    """Sequence / token / discarded-tail counts from a dataset, when those attributes exist."""
    payload: dict[str, Any] = {f"{split}_sequences": len(dataset)}
    if hasattr(dataset, "total_tokens"):
        payload[f"{split}_tokens"] = int(dataset.total_tokens)
    if hasattr(dataset, "discarded_tail_tokens"):
        payload[f"{split}_discarded_tail_tokens"] = int(dataset.discarded_tail_tokens)
    return payload


def packed_data_stats(
    train_loader: DataLoader | None,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Packed-sample accounting for the loaders attached to a Trainer."""
    payload: dict[str, Any] = {}
    if train_loader is not None and getattr(train_loader, "dataset", None) is not None:
        payload.update(packed_split_stats(train_loader.dataset, "train"))
    if val_loader is not None and getattr(val_loader, "dataset", None) is not None:
        payload.update(packed_split_stats(val_loader.dataset, "validation"))
    return payload


def default_token_budget(
    *,
    max_steps: int,
    batch_size: int,
    context_length: int | None,
    grad_accum_steps: int,
    tokens_per_optimizer_step: int | None,
) -> dict[str, Any]:
    """Planned token budget when `--target-tokens` was not supplied."""
    actual = None
    if tokens_per_optimizer_step is not None:
        actual = int(max_steps) * int(tokens_per_optimizer_step)
    return {
        "requested_token_budget": None,
        "micro_batch_size": batch_size,
        "context_length": context_length,
        "grad_accum_steps": grad_accum_steps,
        "world_size": 1,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "optimizer_steps": max_steps,
        "actual_token_budget": actual,
        "overshoot_tokens": None,
    }


def collect_static_run_extra(
    *,
    model: nn.Module,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    parameter_count: int,
    tokens_per_optimizer_step: int | None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fields known at Trainer init, merged with caller extras (caller keys win)."""
    cfg = getattr(model, "config", None)
    context_length = getattr(cfg, "context_length", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    payload: dict[str, Any] = {
        "parameter_count": parameter_count,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "parameter_breakdown": parameter_breakdown(model),
        "optimizer_param_groups": optimizer_param_group_counts(optimizer),
        "tie_word_embeddings": model_ties_embeddings(model),
        "packed_data": packed_data_stats(train_loader, val_loader),
        "token_budget": default_token_budget(
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            context_length=int(context_length) if context_length is not None else None,
            grad_accum_steps=config.gradient_accumulation_steps,
            tokens_per_optimizer_step=tokens_per_optimizer_step,
        ),
    }
    if context_length is not None:
        payload["eval_tokens"] = eval_tokens_covered(
            config.batch_size, int(context_length), config.eval_batches
        )
        payload["training_sequence_length"] = int(context_length)
        payload["head_dim"] = getattr(cfg, "head_dim", None)
    if vocab_size is not None:
        payload["uniform_ce_reference"] = uniform_ce_reference(int(vocab_size))
        payload["vocab_size"] = int(vocab_size)
    if extra_metadata:
        payload.update(dict(extra_metadata))
    return payload


def extract_curve_extrema(metrics_path: Path | str) -> dict[str, Any]:
    """First/last train loss and minimum val CE/PPL from metrics.jsonl."""
    path = Path(metrics_path)
    empty = {
        "first_train_loss": None,
        "first_train_step": None,
        "last_train_loss": None,
        "last_train_step": None,
        "min_val_loss": None,
        "min_val_perplexity": None,
        "min_val_step": None,
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
    }
    if not path.exists():
        return empty

    first_train: dict[str, Any] | None = None
    last_train: dict[str, Any] | None = None
    min_val: dict[str, Any] | None = None
    peak_alloc: int | None = None
    peak_reserved: int | None = None

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "train":
                if first_train is None:
                    first_train = record
                last_train = record
                alloc = record.get("peak_allocated_vram_bytes")
                reserved = record.get("peak_reserved_vram_bytes")
                if alloc is not None:
                    alloc_i = int(alloc)
                    peak_alloc = alloc_i if peak_alloc is None else max(peak_alloc, alloc_i)
                if reserved is not None:
                    reserved_i = int(reserved)
                    peak_reserved = reserved_i if peak_reserved is None else max(peak_reserved, reserved_i)
            elif kind == "val" and record.get("val_loss") is not None:
                if min_val is None or float(record["val_loss"]) < float(min_val["val_loss"]):
                    min_val = record

    result = dict(empty)
    if first_train is not None:
        result["first_train_loss"] = first_train.get("loss", first_train.get("train_loss"))
        result["first_train_step"] = first_train.get("step")
    if last_train is not None:
        result["last_train_loss"] = last_train.get("loss", last_train.get("train_loss"))
        result["last_train_step"] = last_train.get("step")
    if min_val is not None:
        val_loss = float(min_val["val_loss"])
        result["min_val_loss"] = val_loss
        result["min_val_step"] = min_val.get("step")
        ppl = min_val.get("val_perplexity")
        result["min_val_perplexity"] = float(ppl) if ppl is not None else perplexity_from_loss(val_loss)
    result["peak_allocated_vram_bytes"] = peak_alloc
    result["peak_reserved_vram_bytes"] = peak_reserved
    return result


def tokens_per_parameter(tokens_seen: int | None, parameter_count: int | None) -> float | None:
    if tokens_seen is None or parameter_count is None or parameter_count <= 0:
        return None
    return float(tokens_seen) / float(parameter_count)


def gpu_hours_from_seconds(elapsed_seconds: float | None) -> float | None:
    if elapsed_seconds is None:
        return None
    return float(elapsed_seconds) / SECONDS_PER_HOUR


def collect_summary_whitepaper_fields(
    *,
    output_dir: Path | str,
    tokens_seen: int,
    elapsed_seconds: float,
    parameter_count: int,
    tokens_per_optimizer_step: int | None,
    eval_tokens: int | None,
    uniform_ce: float | None,
    peak_allocated_vram_bytes: int | None = None,
    peak_reserved_vram_bytes: int | None = None,
    first_train_loss: float | None = None,
    first_train_step: int | None = None,
    last_train_loss: float | None = None,
    last_train_step: int | None = None,
    best_val_loss: float | None = None,
    best_val_step: int | None = None,
) -> dict[str, Any]:
    """Named whitepaper fields written into summary.json extra."""
    extrema = extract_curve_extrema(Path(output_dir) / "metrics.jsonl")
    first_loss = extrema["first_train_loss"] if extrema["first_train_loss"] is not None else first_train_loss
    first_step = extrema["first_train_step"] if extrema["first_train_step"] is not None else first_train_step
    last_loss = extrema["last_train_loss"] if extrema["last_train_loss"] is not None else last_train_loss
    last_step = extrema["last_train_step"] if extrema["last_train_step"] is not None else last_train_step
    min_val = extrema["min_val_loss"] if extrema["min_val_loss"] is not None else best_val_loss
    min_step = extrema["min_val_step"] if extrema["min_val_step"] is not None else best_val_step
    min_ppl = extrema["min_val_perplexity"]
    if min_ppl is None and min_val is not None:
        min_ppl = perplexity_from_loss(float(min_val))

    peak_alloc = peak_allocated_vram_bytes
    if extrema["peak_allocated_vram_bytes"] is not None:
        logged = int(extrema["peak_allocated_vram_bytes"])
        peak_alloc = logged if peak_alloc is None else max(int(peak_alloc), logged)
    peak_reserved = peak_reserved_vram_bytes
    if extrema["peak_reserved_vram_bytes"] is not None:
        logged_r = int(extrema["peak_reserved_vram_bytes"])
        peak_reserved = logged_r if peak_reserved is None else max(int(peak_reserved), logged_r)

    payload: dict[str, Any] = {
        "parameter_count": parameter_count,
        "tokens_per_parameter": tokens_per_parameter(tokens_seen, parameter_count),
        "gpu_hours": gpu_hours_from_seconds(elapsed_seconds),
        "uniform_ce_reference": uniform_ce,
        "eval_tokens": eval_tokens,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "first_train_loss": first_loss,
        "first_train_step": first_step,
        "last_train_loss": last_loss,
        "last_train_step": last_step,
        "min_val_loss": min_val,
        "min_val_perplexity": min_ppl,
        "min_val_step": min_step,
        "peak_allocated_vram_bytes": peak_alloc,
        "peak_reserved_vram_bytes": peak_reserved,
    }
    if peak_alloc is not None:
        payload["peak_allocated_vram_mib"] = peak_alloc / BYTES_PER_MIB
    if peak_reserved is not None:
        payload["peak_reserved_vram_mib"] = peak_reserved / BYTES_PER_MIB
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = load_json(path)
    return loaded if isinstance(loaded, dict) else None


def resolve_hellaswag_output_json(
    output_json: str | Path | None,
    checkpoint: str | Path | None,
) -> Path:
    """Default HellaSwag JSON path: checkpoint directory, otherwise CWD."""
    if output_json is not None:
        return Path(output_json)
    if checkpoint is not None:
        return Path(checkpoint).parent / "hellaswag.json"
    return Path("hellaswag.json")


def build_whitepaper_snapshot(
    run_dir: Path | str,
    *,
    evaluation_json: Path | str | None = None,
    hellaswag_json: Path | str | None = None,
) -> dict[str, Any]:
    """Assembles a copy-ready whitepaper payload from a finished run directory."""
    run_path = Path(run_dir)
    run_meta = _load_optional_json(run_path / "run.json") or {}
    summary = _load_optional_json(run_path / "summary.json") or {}
    model_cfg = _load_optional_json(run_path / "model_config.json") or {}
    train_cfg = _load_optional_json(run_path / "training_config.json") or {}
    dataset = _load_optional_json(run_path / "dataset.json") or {}
    extra = run_meta.get("extra") if isinstance(run_meta.get("extra"), dict) else {}
    system = run_meta.get("system") if isinstance(run_meta.get("system"), dict) else {}
    git = run_meta.get("git") if isinstance(run_meta.get("git"), dict) else {}

    eval_path = Path(evaluation_json) if evaluation_json is not None else run_path / "evaluation.json"
    hs_path = Path(hellaswag_json) if hellaswag_json is not None else run_path / "hellaswag.json"
    evaluation = _load_optional_json(eval_path)
    hellaswag = _load_optional_json(hs_path)

    eval_metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    hellaswag_norm = hellaswag.get("norm_accuracy") if isinstance(hellaswag, dict) else None
    hellaswag_raw = hellaswag.get("raw_accuracy") if isinstance(hellaswag, dict) else None

    stats = dataset.get("statistics") if isinstance(dataset.get("statistics"), dict) else {}
    provenance = (
        dataset.get("dataset_provenance")
        if isinstance(dataset.get("dataset_provenance"), dict)
        else {}
    )
    tokenizer_info = dataset.get("tokenizer") if isinstance(dataset.get("tokenizer"), dict) else {}
    packed = extra.get("packed_data") if isinstance(extra.get("packed_data"), dict) else {}
    breakdown = extra.get("parameter_breakdown") if isinstance(extra.get("parameter_breakdown"), dict) else {}
    token_budget = extra.get("token_budget") if isinstance(extra.get("token_budget"), dict) else {}
    opt_groups = extra.get("optimizer_param_groups") if isinstance(extra.get("optimizer_param_groups"), dict) else {}

    parameter_count = summary.get("parameter_count", extra.get("parameter_count"))
    tokens_seen = summary.get("tokens_seen")
    elapsed = summary.get("elapsed_seconds")
    gpu_hours = summary.get("gpu_hours")
    if gpu_hours is None:
        gpu_hours = gpu_hours_from_seconds(elapsed)

    snapshot = {
        "run_dir": str(run_path),
        "run_name": run_meta.get("run_name"),
        "status": summary.get("status"),
        "git": git,
        "abstract": {
            "parameter_count": parameter_count,
            "tokens_seen": tokens_seen,
            "tokens_per_parameter": summary.get("tokens_per_parameter")
            or tokens_per_parameter(tokens_seen, parameter_count),
            "elapsed_seconds": elapsed,
            "gpu_hours": gpu_hours,
            "gpu_name": system.get("gpu_name"),
            "min_val_loss": summary.get("min_val_loss", summary.get("best_val_loss")),
            "min_val_perplexity": summary.get("min_val_perplexity"),
            "min_val_step": summary.get("min_val_step"),
            "eval_tokens": summary.get("eval_tokens", extra.get("eval_tokens")),
            "hellaswag_norm_accuracy": hellaswag_norm,
            "hellaswag_raw_accuracy": hellaswag_raw,
            "full_validation_loss": (eval_metrics or {}).get("validation_loss") if eval_metrics else None,
            "full_validation_perplexity": (eval_metrics or {}).get("perplexity") if eval_metrics else None,
        },
        "model": {
            **model_cfg,
            "parameter_count": parameter_count,
            "parameter_breakdown": breakdown,
            "tie_word_embeddings": extra.get("tie_word_embeddings"),
            "head_dim": extra.get("head_dim") or breakdown.get("head_dim"),
            "training_sequence_length": extra.get("training_sequence_length")
            or model_cfg.get("context_length"),
        },
        "training": {
            **{k: train_cfg.get(k) for k in (
                "learning_rate",
                "min_learning_rate",
                "warmup_steps",
                "max_steps",
                "batch_size",
                "gradient_accumulation_steps",
                "weight_decay",
                "beta1",
                "beta2",
                "eps",
                "max_grad_norm",
                "precision",
                "seed",
                "eval_interval",
                "eval_batches",
                "eval_at_start",
                "log_interval",
                "checkpoint_steps",
                "checkpoint_interval",
            )},
            "eval_tokens": extra.get("eval_tokens") or summary.get("eval_tokens"),
            "tokens_per_optimizer_step": extra.get("tokens_per_optimizer_step")
            or summary.get("tokens_per_optimizer_step"),
            "token_budget": token_budget,
            "optimizer_param_groups": opt_groups,
            "uniform_ce_reference": summary.get("uniform_ce_reference", extra.get("uniform_ce_reference")),
        },
        "compute": {
            "gpu_name": system.get("gpu_name"),
            "total_vram_bytes": system.get("total_vram_bytes"),
            "pytorch_version": system.get("pytorch_version"),
            "cuda_runtime": system.get("cuda_runtime"),
            "nvidia_driver": system.get("nvidia_driver"),
            "bf16_supported": system.get("bf16_supported"),
            "elapsed_seconds": elapsed,
            "gpu_hours": gpu_hours,
            "training_only_tokens_per_sec": summary.get("training_only_tokens_per_sec"),
            "end_to_end_tokens_per_sec": summary.get("end_to_end_tokens_per_sec"),
            "peak_allocated_vram_bytes": summary.get("peak_allocated_vram_bytes"),
            "peak_allocated_vram_mib": summary.get("peak_allocated_vram_mib"),
            "peak_reserved_vram_bytes": summary.get("peak_reserved_vram_bytes"),
            "peak_reserved_vram_mib": summary.get("peak_reserved_vram_mib"),
        },
        "language_model_results": {
            "first_train_loss": summary.get("first_train_loss"),
            "first_train_step": summary.get("first_train_step"),
            "last_train_loss": summary.get("last_train_loss", summary.get("final_train_loss")),
            "last_train_step": summary.get("last_train_step", summary.get("final_step")),
            "min_val_loss": summary.get("min_val_loss", summary.get("best_val_loss")),
            "min_val_perplexity": summary.get("min_val_perplexity"),
            "min_val_step": summary.get("min_val_step"),
            "tokens_seen": tokens_seen,
            "elapsed_seconds": elapsed,
            "mean_tokens_per_sec": summary.get("end_to_end_tokens_per_sec"),
            "peak_allocated_vram_mib": summary.get("peak_allocated_vram_mib"),
            "uniform_ce_reference": summary.get("uniform_ce_reference", extra.get("uniform_ce_reference")),
        },
        "data": {
            "repository": provenance.get("repository") or run_meta.get("dataset_repository"),
            "config": provenance.get("config") or run_meta.get("dataset_config"),
            "revision": provenance.get("revision") or run_meta.get("dataset_revision"),
            "license": provenance.get("license"),
            "tokenizer_encoding": tokenizer_info.get("encoding") or run_meta.get("tokenizer_encoding"),
            "tokenizer_vocab_size": tokenizer_info.get("vocab_size"),
            "statistics": stats,
            "packed_data": packed,
        },
        "downstream": {
            "evaluation_path": str(eval_path) if eval_path.exists() else None,
            "hellaswag_path": str(hs_path) if hs_path.exists() else None,
            "evaluation": evaluation,
            "hellaswag": hellaswag,
        },
    }
    return snapshot


def _fmt(value: Any, digits: int | None = None) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if digits is None:
            return f"{value:.6g}"
        return f"{value:.{digits}f}"
    return str(value)


def format_whitepaper_markdown(snapshot: dict[str, Any]) -> str:
    """Renders copy-ready markdown tables from a whitepaper snapshot."""
    abstract = snapshot.get("abstract") or {}
    model = snapshot.get("model") or {}
    training = snapshot.get("training") or {}
    compute = snapshot.get("compute") or {}
    lm = snapshot.get("language_model_results") or {}
    data = snapshot.get("data") or {}
    stats = data.get("statistics") or {}
    packed = data.get("packed_data") or {}
    budget = training.get("token_budget") or {}
    breakdown = model.get("parameter_breakdown") or {}
    groups = training.get("optimizer_param_groups") or {}
    git = snapshot.get("git") or {}

    lines = [
        "# Whitepaper snapshot",
        "",
        f"Run: `{snapshot.get('run_dir')}`",
        f"Status: {snapshot.get('status') or '—'}",
        f"Git: `{git.get('git_commit') or '—'}` (dirty={git.get('git_dirty')})",
        "",
        "## Abstract",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Unique parameters | {_fmt(abstract.get('parameter_count'))} |",
        f"| Tokens processed | {_fmt(abstract.get('tokens_seen'))} |",
        f"| Tokens per parameter | {_fmt(abstract.get('tokens_per_parameter'), 4)} |",
        f"| Wall-clock (s) | {_fmt(abstract.get('elapsed_seconds'), 2)} |",
        f"| GPU hours | {_fmt(abstract.get('gpu_hours'), 4)} |",
        f"| GPU | {abstract.get('gpu_name') or '—'} |",
        f"| Min val CE / PPL | {_fmt(abstract.get('min_val_loss'), 4)} / {_fmt(abstract.get('min_val_perplexity'), 4)} |",
        f"| Min val step | {_fmt(abstract.get('min_val_step'))} |",
        f"| In-loop eval tokens | {_fmt(abstract.get('eval_tokens'))} |",
        f"| HellaSwag acc_norm | {_fmt(abstract.get('hellaswag_norm_accuracy'), 4)} |",
        f"| Full val CE / PPL | {_fmt(abstract.get('full_validation_loss'), 4)} / {_fmt(abstract.get('full_validation_perplexity'), 4)} |",
        "",
        "## Model",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Unique parameters | {_fmt(model.get('parameter_count'))} |",
        f"| vocab_size | {_fmt(model.get('vocab_size'))} |",
        f"| hidden_size (d_model) | {_fmt(model.get('d_model'))} |",
        f"| num_hidden_layers | {_fmt(model.get('n_layers'))} |",
        f"| num_attention_heads | {_fmt(model.get('n_heads'))} |",
        f"| head_dim | {_fmt(model.get('head_dim'))} |",
        f"| intermediate_size (d_ff) | {_fmt(model.get('d_ff'))} |",
        f"| max_position_embeddings | {_fmt(model.get('context_length'))} |",
        f"| Training sequence length | {_fmt(model.get('training_sequence_length'))} |",
        f"| layer_norm_eps | {model.get('layer_norm_eps') if model.get('layer_norm_eps') is not None else '—'} |",
        f"| bias | {model.get('bias') if model.get('bias') is not None else '—'} |",
        f"| tie_word_embeddings | {model.get('tie_word_embeddings') if model.get('tie_word_embeddings') is not None else '—'} |",
        f"| Token embedding params | {_fmt(breakdown.get('token_embedding'))} |",
        f"| Position embedding params | {_fmt(breakdown.get('position_embedding'))} |",
        f"| Transformer + final LN | {_fmt(breakdown.get('transformer_blocks_and_final_ln'))} |",
        "",
        "## Training",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| max_steps | {_fmt(training.get('max_steps'))} |",
        f"| Planned executed token count | {_fmt(budget.get('actual_token_budget'))} |",
        f"| Token budget (requested) | {_fmt(budget.get('requested_token_budget'))} |",
        f"| Overshoot tokens | {_fmt(budget.get('overshoot_tokens'))} |",
        f"| Tokens / optimizer step | {_fmt(training.get('tokens_per_optimizer_step'))} |",
        f"| micro_batch_size × grad_accum | {_fmt(training.get('batch_size'))} × {_fmt(training.get('gradient_accumulation_steps'))} |",
        f"| learning_rate | {training.get('learning_rate') if training.get('learning_rate') is not None else '—'} |",
        f"| min_learning_rate | {training.get('min_learning_rate') if training.get('min_learning_rate') is not None else '—'} |",
        f"| warmup_steps | {_fmt(training.get('warmup_steps'))} |",
        f"| betas / eps | [{training.get('beta1')}, {training.get('beta2')}] / {training.get('eps')} |",
        f"| weight_decay | {training.get('weight_decay')} (decay params {_fmt(groups.get('decay_parameters'))}; no-decay {_fmt(groups.get('no_decay_parameters'))}) |",
        f"| max_grad_norm | {training.get('max_grad_norm')} |",
        f"| precision | {training.get('precision') or '—'} |",
        f"| seed | {_fmt(training.get('seed'))} |",
        f"| eval_interval | {_fmt(training.get('eval_interval'))} |",
        f"| eval_tokens / eval_batches | {_fmt(training.get('eval_tokens'))} / {_fmt(training.get('eval_batches'))} |",
        f"| Uniform-over-vocab CE ln(V) | {_fmt(training.get('uniform_ce_reference'), 4)} |",
        "",
        "## Compute",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| GPU | {compute.get('gpu_name') or '—'} |",
        f"| VRAM (bytes) | {_fmt(compute.get('total_vram_bytes'))} |",
        f"| PyTorch | {compute.get('pytorch_version') or '—'} |",
        f"| CUDA (torch) | {compute.get('cuda_runtime') or '—'} |",
        f"| Driver | {compute.get('nvidia_driver') or '—'} |",
        f"| bf16 hardware | {compute.get('bf16_supported') if compute.get('bf16_supported') is not None else '—'} |",
        f"| Wall-clock (s) | {_fmt(compute.get('elapsed_seconds'), 2)} |",
        f"| GPU hours | {_fmt(compute.get('gpu_hours'), 4)} |",
        f"| Training-only tok/s | {_fmt(compute.get('training_only_tokens_per_sec'), 2)} |",
        f"| End-to-end tok/s | {_fmt(compute.get('end_to_end_tokens_per_sec'), 2)} |",
        f"| Peak CUDA allocated (MiB) | {_fmt(compute.get('peak_allocated_vram_mib'), 2)} |",
        "",
        "## Language-model results",
        "",
        "| Metric | Value | Step |",
        "|---|---|---|",
        f"| first train loss | {_fmt(lm.get('first_train_loss'), 4)} | {_fmt(lm.get('first_train_step'))} |",
        f"| last train loss | {_fmt(lm.get('last_train_loss'), 4)} | {_fmt(lm.get('last_train_step'))} |",
        f"| min val loss | {_fmt(lm.get('min_val_loss'), 4)} | {_fmt(lm.get('min_val_step'))} |",
        f"| min val perplexity | {_fmt(lm.get('min_val_perplexity'), 4)} | {_fmt(lm.get('min_val_step'))} |",
        f"| tokens processed | {_fmt(lm.get('tokens_seen'))} |  |",
        f"| wall time (s) | {_fmt(lm.get('elapsed_seconds'), 2)} |  |",
        f"| mean tokens/sec | {_fmt(lm.get('mean_tokens_per_sec'), 2)} |  |",
        f"| peak CUDA allocated (MiB) | {_fmt(lm.get('peak_allocated_vram_mib'), 2)} |  |",
        f"| Uniform-over-vocab reference | {_fmt(lm.get('uniform_ce_reference'), 4)} |  |",
        "",
        "## Data",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Repository | {data.get('repository') or '—'} |",
        f"| Config | {data.get('config') or '—'} |",
        f"| Revision | {data.get('revision') or '—'} |",
        f"| License | {data.get('license') or '—'} |",
        f"| Tokenizer | {data.get('tokenizer_encoding') or '—'} (vocab {_fmt(data.get('tokenizer_vocab_size'))}) |",
        f"| Train documents | {_fmt(stats.get('train_documents'))} |",
        f"| Validation documents | {_fmt(stats.get('validation_documents'))} |",
        f"| Train tokens | {_fmt(stats.get('train_tokens'))} |",
        f"| Validation tokens | {_fmt(stats.get('validation_tokens'))} |",
        f"| Train / val shards | {_fmt(stats.get('train_shards'))} / {_fmt(stats.get('validation_shards'))} |",
        f"| Packed train sequences | {_fmt(packed.get('train_sequences'))} |",
        f"| Packed train tokens | {_fmt(packed.get('train_tokens'))} |",
        f"| Discarded train tail tokens | {_fmt(packed.get('train_discarded_tail_tokens'))} |",
        f"| Packed val sequences | {_fmt(packed.get('validation_sequences'))} |",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_whitepaper_snapshot(
    run_dir: Path | str,
    *,
    evaluation_json: Path | str | None = None,
    hellaswag_json: Path | str | None = None,
    output_json: Path | str | None = None,
    output_md: Path | str | None = None,
) -> dict[str, Path]:
    """Writes whitepaper_snapshot.json and WHITEPAPER.md into the run directory."""
    run_path = Path(run_dir)
    snapshot = build_whitepaper_snapshot(
        run_path,
        evaluation_json=evaluation_json,
        hellaswag_json=hellaswag_json,
    )
    json_path = Path(output_json) if output_json is not None else run_path / "whitepaper_snapshot.json"
    md_path = Path(output_md) if output_md is not None else run_path / "WHITEPAPER.md"
    atomic_save_json(json_path, snapshot)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_whitepaper_markdown(snapshot), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
