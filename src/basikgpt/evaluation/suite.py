"""English LM suite: model loading, task runners, summary.json and REPORT.md writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from basikgpt.evaluation.hellaswag import evaluate_hellaswag, load_hellaswag_dataset
from basikgpt.evaluation.lambada import evaluate_lambada, load_lambada_dataset
from basikgpt.evaluation.multiple_choice import EncodeTokenizer, evaluate_multiple_choice
from basikgpt.evaluation.tasks import (
    DEFAULT_SUITE_TASKS,
    PROTOCOL_TASKS,
    iter_arc_easy_scored,
    iter_piqa_scored,
    iter_winogrande_scored,
    load_arc_easy_dataset,
    load_piqa_dataset,
    load_winogrande_dataset,
)
from basikgpt.training.metadata import atomic_save_json
from basikgpt.training.reproducibility import get_git_metadata, get_system_metadata

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks"
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "main_2p5b" / "step-00038147.pt"
DEFAULT_CHECKPOINT_5B = REPO_ROOT / "runs" / "cont_5b_mix" / "step-00076294.pt"


class HFEncodeAdapter:
    """Minimal encode() wrapper around a Hugging Face tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        self.eot_token_id = int(eos if eos is not None else (pad if pad is not None else 0))
        add_bos = bool(getattr(tokenizer, "add_bos_token", False))
        bos = tokenizer.bos_token_id
        self.bos_token_id = int(bos) if (add_bos and bos is not None) else None

    def encode(self, text: str) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)


@dataclass(frozen=True, slots=True)
class ProtocolModelSpec:
    """One model in the locked English comparison set."""

    id: str
    kind: str
    params_label: str
    family: str
    corpus: str
    hf_id: str | None = None
    checkpoint: str | None = None


PROTOCOL_MODELS: tuple[ProtocolModelSpec, ...] = (
    ProtocolModelSpec(
        id="basikgpt-2p5b",
        kind="checkpoint",
        params_label="124M",
        family="GPT-2 Small (basikGPT)",
        corpus="FineWeb-Edu 2.5B tokens",
        checkpoint=str(DEFAULT_CHECKPOINT),
        hf_id="project-iconik/basikGPT-1-v1.0",
    ),
    ProtocolModelSpec(
        id="basikgpt-5b",
        kind="checkpoint",
        params_label="124M",
        family="GPT-2 Small (basikGPT)",
        corpus="FineWeb-Edu 2.5B + FineWeb 2.25B + OpenWebMath 0.25B",
        checkpoint=str(DEFAULT_CHECKPOINT_5B),
        hf_id="project-iconik/basikGPT-1-v1.1",
    ),
    ProtocolModelSpec(
        id="gpt2",
        kind="gpt2",
        params_label="124M",
        family="GPT-2 Small",
        corpus="WebText",
        hf_id="openai-community/gpt2",
    ),
    ProtocolModelSpec(
        id="SmolLM2-135M",
        kind="hf",
        params_label="135M",
        family="SmolLM2",
        corpus="SmolLM2 (HuggingFaceTB)",
        hf_id="HuggingFaceTB/SmolLM2-135M",
    ),
    ProtocolModelSpec(
        id="SmolLM2-360M",
        kind="hf",
        params_label="360M",
        family="SmolLM2",
        corpus="SmolLM2 (HuggingFaceTB)",
        hf_id="HuggingFaceTB/SmolLM2-360M",
    ),
    ProtocolModelSpec(
        id="pythia-160m",
        kind="hf",
        params_label="160M",
        family="Pythia",
        corpus="The Pile",
        hf_id="EleutherAI/pythia-160m",
    ),
    ProtocolModelSpec(
        id="pythia-410m",
        kind="hf",
        params_label="410M",
        family="Pythia",
        corpus="The Pile",
        hf_id="EleutherAI/pythia-410m",
    ),
    ProtocolModelSpec(
        id="Qwen2.5-0.5B",
        kind="hf",
        params_label="0.5B",
        family="Qwen2.5",
        corpus="Qwen2.5 mix (Alibaba)",
        hf_id="Qwen/Qwen2.5-0.5B",
    ),
)

_HF_ID_TO_SPEC = {m.hf_id: m for m in PROTOCOL_MODELS if m.hf_id}


def lookup_protocol_model(model_id: str) -> ProtocolModelSpec | None:
    for spec in PROTOCOL_MODELS:
        if spec.id == model_id:
            return spec
    return None


def spec_for_hf_id(hf_id: str) -> ProtocolModelSpec:
    if hf_id in _HF_ID_TO_SPEC:
        return _HF_ID_TO_SPEC[hf_id]
    short = hf_id.split("/")[-1]
    return ProtocolModelSpec(
        id=short,
        kind="hf" if hf_id not in ("openai-community/gpt2", "gpt2") else "gpt2",
        params_label="?",
        family="Hugging Face CausalLM",
        corpus="unknown",
        hf_id=hf_id if hf_id != "gpt2" else "openai-community/gpt2",
    )


def resolve_context_length(model: nn.Module, fallback: int = 1024) -> int:
    """Native maximum context length for scoring truncation."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return fallback
    if hasattr(cfg, "context_length"):
        return int(cfg.context_length)
    for attr in (
        "n_positions",
        "max_position_embeddings",
        "max_sequence_length",
        "max_context_length",
    ):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return fallback


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_gpt2_path_model(
    *,
    checkpoint: str | Path | None = None,
    device: torch.device,
) -> tuple[nn.Module, EncodeTokenizer, dict[str, Any]]:
    """Loads our .pt checkpoint or official GPT-2 via the existing converter + tiktoken."""
    from basikgpt.config import GPTConfig
    from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
    from basikgpt.data.tokenizer import GPT2Tokenizer
    from basikgpt.model.gpt import GPT
    from basikgpt.training.checkpoint import load_model_from_checkpoint

    tokenizer: EncodeTokenizer = GPT2Tokenizer()
    if checkpoint is None:
        cfg = GPTConfig.gpt2_small(dropout=0.0, attention_backend="sdpa")
        model = GPT(cfg)
        load_hf_gpt2_weights(model, "openai-community/gpt2")
        if device.type == "cuda":
            model.to(device=device, dtype=torch.bfloat16)
        else:
            model.to(device)
        model.eval()
        meta = {
            "source": "gpt2",
            "hf_id": "openai-community/gpt2",
            "parameters": _count_parameters(model),
            "context_length": cfg.context_length,
            "forward_path": "gpt2",
            "attention_backend": "sdpa",
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "config": asdict(cfg) if is_dataclass(cfg) else {},
        }
        return model, tokenizer, meta

    model, ckpt_meta = load_model_from_checkpoint(checkpoint, device=device)
    if device.type == "cuda":
        model.to(device=device, dtype=torch.bfloat16)
    cfg = ckpt_meta["model_config"]
    meta = {
        "source": "checkpoint",
        "checkpoint_path": str(checkpoint),
        "global_step": ckpt_meta.get("global_step", 0),
        "tokens_seen": ckpt_meta.get("tokens_seen", 0),
        "parameters": _count_parameters(model),
        "context_length": int(getattr(cfg, "context_length", 1024)),
        "forward_path": "gpt2",
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "config": asdict(cfg) if is_dataclass(cfg) else {},
    }
    return model, tokenizer, meta


def load_hf_causal_lm(
    hf_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, EncodeTokenizer, dict[str, Any]]:
    """Loads AutoModelForCausalLM and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    adapter = HFEncodeAdapter(hf_tok)
    meta = {
        "source": "hf",
        "hf_id": hf_id,
        "parameters": _count_parameters(model),
        "context_length": resolve_context_length(model),
        "forward_path": "hf_causallm",
        "tokenizer_id": hf_id,
        "tokenizer_source": hf_id,
        "dtype": str(dtype).replace("torch.", ""),
        "trust_remote_code": True,
    }
    return model, adapter, meta


def run_task(
    task_name: str,
    model: nn.Module,
    tokenizer: EncodeTokenizer,
    device: torch.device,
    max_context_length: int,
    max_examples: int | None = None,
    progress_interval: int = 50,
    local_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Runs one protocol task and returns a JSON-serializable result dict."""
    local_paths = local_paths or {}
    spec = PROTOCOL_TASKS[task_name]
    local = local_paths.get(task_name)

    if task_name == "hellaswag":
        dataset = load_hellaswag_dataset(split=spec.split, streaming=True, local_path=local)
        summary, _results = evaluate_hellaswag(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            max_examples=max_examples,
            format_style="activity_ctx",
            split_name=spec.split,
            max_context_length=max_context_length,
            progress_interval=progress_interval,
        )
        payload = summary.to_dict()
        payload["task"] = "hellaswag"
        payload["metric_primary"] = "acc_norm"
        payload["score"] = summary.norm_accuracy
        payload["acc_raw"] = summary.raw_accuracy
        payload["acc_norm"] = summary.norm_accuracy
        payload["chance"] = spec.chance
        return payload

    if task_name == "lambada_openai":
        dataset = load_lambada_dataset(split=spec.split, local_path=local)
        summary, _results = evaluate_lambada(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            max_examples=max_examples,
            max_context_length=max_context_length,
            split_name=spec.split,
            progress_interval=progress_interval,
        )
        return summary.to_dict()

    if task_name == "piqa":
        examples = iter_piqa_scored(load_piqa_dataset(split=spec.split, local_path=local))
        summary, _results = evaluate_multiple_choice(
            model=model,
            examples=examples,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
            max_examples=max_examples,
            task="piqa",
            split=spec.split,
            primary_metric=spec.primary_metric,
            chance=spec.chance,
            progress_interval=progress_interval,
        )
        return summary.to_dict()

    if task_name == "winogrande":
        examples = iter_winogrande_scored(
            load_winogrande_dataset(split=spec.split, local_path=local)
        )
        summary, _results = evaluate_multiple_choice(
            model=model,
            examples=examples,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
            max_examples=max_examples,
            task="winogrande",
            split=spec.split,
            primary_metric=spec.primary_metric,
            chance=spec.chance,
            progress_interval=progress_interval,
        )
        return summary.to_dict()

    if task_name == "arc_easy":
        examples = iter_arc_easy_scored(load_arc_easy_dataset(split=spec.split, local_path=local))
        summary, _results = evaluate_multiple_choice(
            model=model,
            examples=examples,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
            max_examples=max_examples,
            task="arc_easy",
            split=spec.split,
            primary_metric=spec.primary_metric,
            chance=spec.chance,
            progress_interval=progress_interval,
        )
        payload = summary.to_dict()
        payload["chance_note"] = "1 / number of choices (typically 4 → 25%)"
        return payload

    raise ValueError(f"Unknown task: {task_name}")


def load_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if not path.exists():
        return {
            "protocol": "english-lm-suite-v1",
            "git": get_git_metadata(),
            "system": get_system_metadata(),
            "models": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_model_summary(
    output_dir: Path,
    spec: ProtocolModelSpec,
    model_meta: dict[str, Any],
    tasks: dict[str, Any],
) -> dict[str, Any]:
    """Merges one model's scores into summary.json and rewrites REPORT.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(output_dir)
    summary["git"] = get_git_metadata()
    summary["system"] = get_system_metadata()
    entry = {
        "id": spec.id,
        "kind": spec.kind,
        "params_label": spec.params_label,
        "family": spec.family,
        "corpus": spec.corpus,
        "hf_id": spec.hf_id,
        "checkpoint": spec.checkpoint,
        "parameters": model_meta.get("parameters"),
        "forward_path": model_meta.get("forward_path"),
        "context_length": model_meta.get("context_length"),
        "tokenizer_source": model_meta.get("tokenizer_source")
        or ("tiktoken gpt2" if model_meta.get("forward_path") == "gpt2" else spec.hf_id),
        "tasks": tasks,
    }
    models = dict(summary.get("models") or {})
    existing = models.get(spec.id, {})
    merged_tasks = dict(existing.get("tasks") or {})
    merged_tasks.update(tasks)
    entry["tasks"] = merged_tasks
    models[spec.id] = entry
    summary["models"] = models
    atomic_save_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    return summary


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _task_score(entry: dict[str, Any], task: str, key: str) -> Any:
    tasks = entry.get("tasks") or {}
    rec = tasks.get(task) or {}
    return rec.get(key)


def write_report(output_dir: Path, summary: dict[str, Any] | None = None) -> Path:
    """Writes the protocol body and comparison table to benchmarks/REPORT.md."""
    if summary is None:
        summary = load_summary(output_dir)
    models: dict[str, Any] = summary.get("models") or {}
    ordered_ids = [m.id for m in PROTOCOL_MODELS]
    extra_ids = [mid for mid in models if mid not in ordered_ids]
    rows = []
    for mid in ordered_ids + extra_ids:
        if mid not in models:
            continue
        e = models[mid]
        hs_norm = _task_score(e, "hellaswag", "acc_norm")
        hs_raw = _task_score(e, "hellaswag", "acc_raw")
        lambada = _task_score(e, "lambada_openai", "accuracy")
        piqa = _task_score(e, "piqa", "acc_norm")
        wg = _task_score(e, "winogrande", "acc_raw")
        arc = _task_score(e, "arc_easy", "acc_norm")
        rows.append(
            f"| `{e.get('id', mid)}` | {e.get('params_label', '')} | {e.get('family', '')} "
            f"| {e.get('corpus', '')} | {_fmt_pct(hs_norm)} | {_fmt_pct(hs_raw)} "
            f"| {_fmt_pct(lambada)} | {_fmt_pct(piqa)} | {_fmt_pct(wg)} | {_fmt_pct(arc)} |"
        )
    table = "\n".join(rows) if rows else "| *(no scores yet)* | | | | | | | | | |"

    body = f"""# English LM suite (zero-shot)

This file is the protocol. Scores below were measured in this repository with the **same
splits, prompts, and scoring formulas**. Published numbers from other papers are not mixed in.

## Tasks

| Task | Split | Primary metric | Also reported |
|---|---|---|---|
| HellaSwag | validation | acc_norm (mean completion log-likelihood) | acc_raw (sum LL) |
| LAMBADA (OpenAI) | test | last-word accuracy (greedy match) | — |
| PIQA | validation (`baber/piqa`) | acc_norm | acc_raw |
| WinoGrande | validation (`winogrande_xl`) | accuracy = acc_raw | acc_norm |
| ARC-Easy | test | acc_norm | acc_raw |

Not in this suite: KoBEST, MMLU, GSM8K, HumanEval, WikiText perplexity.

Chance rates (for calibration only, not subtracted from scores): HellaSwag 25%; PIQA and
WinoGrande 50%; ARC-Easy 1/N choices (typically 4 → 25%). LAMBADA is open-vocab.

## Scoring rules (shared by both forward paths)

Multiple-choice (HellaSwag, PIQA, WinoGrande, ARC-Easy):

1. Encode **context** and each **choice** separately. Choice tokens are `" " + ending`.
2. Concatenate `[context || choice]`. Left-truncate context if the pair exceeds the model's
   context length, keeping at least one context token.
3. Score **choice tokens only**. Logits at position k-1 parameterize token k.
4. **acc_raw** = argmax of sum log-likelihood. **acc_norm** = argmax of mean log-likelihood.

LAMBADA:

1. Split on the last space: prefix / last word (OpenAI / lm-eval convention).
2. Target continuation is `" " + last_word`.
3. Accuracy is 1 iff greedy argmax tokens equal the target token ids on every position.

WinoGrande blank: context is the text **left of `_`**; each completion is
`option + remainder after the blank`.

ARC-Easy context: `Question: {{question}}\\nAnswer:`.

## Comparison set

Token counts and architectures are **not** matched. The table lists parameter size, family,
and training corpus only. All external models are **base** (not Instruct), 0.1B–0.5B.

**Ours:** two basikGPT GPT-2 Small checkpoints (124M parameters; token counts are
tokens seen, not parameters):

- `basikgpt-2p5b` — FineWeb-Edu 2.5B (`runs/main_2p5b/step-00038147.pt`)
- `basikgpt-5b` — same run continued to 5B on FineWeb 2.25B + OpenWebMath 0.25B
  (`runs/cont_5b_mix/step-00076294.pt`)

Intermediate 100M / 500M / 1B / 3.5B checkpoints are not in this suite.

External Hugging Face bases:

- `openai-community/gpt2` — 124M (GPT-2 forward path: tiktoken + existing converter)
- `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolLM2-360M`
- `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`
- `Qwen/Qwen2.5-0.5B`

Not included: GPT-2 Medium/Large/XL, OPT, Gemma, Llama, OpenELM, TinyLlama, Qwen2.5-1.5B.
lm-eval-harness is not a dependency.

## Two forward paths

| Path | Models | Tokenizer | Forward |
|---|---|---|---|
| GPT-2 | basikGPT `.pt`, official `gpt2` | tiktoken `gpt2` | basikGPT `GPT` logits tensor |
| HF CausalLM | SmolLM2, Pythia, Qwen2.5-0.5B | each model's tokenizer | `AutoModelForCausalLM` `.logits` |

Prompts, length normalization, and argmax rules are the same. Tokenization is not: scores
are comparable as a protocol, not as matched-token perplexity.

## Results

| Model | Params | Family | Corpus | HS acc_norm | HS acc_raw | LAMBADA | PIQA acc_norm | WG acc | ARC-E acc_norm |
|---|---|---|---|---|---|---|---|---|---|
{table}

Per-task JSON is written locally under `benchmarks/models/` (gitignored). Machine-readable rollup: `summary.json`.

## How to reproduce

```text
python scripts/evaluate_lm_suite.py --checkpoint runs/main_2p5b/step-00038147.pt
python scripts/evaluate_lm_suite.py --checkpoint runs/cont_5b_mix/step-00076294.pt --model-id basikgpt-5b
python scripts/evaluate_lm_suite.py --hf-model openai-community/gpt2
python scripts/evaluate_lm_suite.py --protocol-all --device cuda
```

`--output-dir` defaults to `benchmarks/`. After each model the suite rewrites `summary.json`
and this report so a crash does not drop finished scores.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "REPORT.md"
    path.write_text(body, encoding="utf-8")
    return path
