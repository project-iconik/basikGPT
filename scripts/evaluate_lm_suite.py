"""CLI: zero-shot English LM suite (HellaSwag, LAMBADA, PIQA, WinoGrande, ARC-Easy)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.evaluation.suite import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    PROTOCOL_MODELS,
    lookup_protocol_model,
    load_gpt2_path_model,
    load_hf_causal_lm,
    resolve_context_length,
    run_task,
    spec_for_hf_id,
    upsert_model_summary,
    write_report,
)
from basikgpt.evaluation.tasks import DEFAULT_SUITE_TASKS  # noqa: E402
from basikgpt.training.metadata import atomic_save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "basikGPT English LM suite: HellaSwag, LAMBADA OpenAI, PIQA, "
            "WinoGrande, ARC-Easy (same splits and scoring for every model)."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a basikGPT .pt checkpoint (GPT-2 forward path).",
    )
    model_group.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help=(
            "Hugging Face model id. openai-community/gpt2 uses the GPT-2 path; "
            "all other ids use AutoModelForCausalLM."
        ),
    )
    model_group.add_argument(
        "--protocol-all",
        action="store_true",
        help="Run the locked comparison set (basikGPT 2.5B ckpt + 6 HF bases).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Directory name under benchmarks/models/ (default: protocol id).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_SUITE_TASKS),
        help="Comma-separated task names (default: all five protocol tasks).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root for REPORT.md, summary.json, and models/<id>/ (default: benchmarks/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="DType for Hugging Face CausalLM weights (GPT-2 path keeps checkpoint dtype).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap examples per task (default: full split).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Progress print frequency in examples.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Optional directory of JSON/JSONL fixtures named <task>.jsonl (offline).",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _local_paths(local_dir: str | None) -> dict[str, str]:
    if not local_dir:
        return {}
    root = Path(local_dir)
    mapping: dict[str, str] = {}
    for task in DEFAULT_SUITE_TASKS:
        for suffix in (".jsonl", ".json"):
            candidate = root / f"{task}{suffix}"
            if candidate.exists():
                mapping[task] = str(candidate)
                break
    return mapping


def _run_one(
    spec,
    *,
    checkpoint: str | None,
    hf_model: str | None,
    device: torch.device,
    dtype: torch.dtype,
    tasks: list[str],
    output_dir: Path,
    max_examples: int | None,
    progress_interval: int,
    local_paths: dict[str, str],
) -> None:
    print("=" * 70)
    print(f"  model id: {spec.id}")
    print(f"  family:   {spec.family}  params={spec.params_label}")
    print(f"  corpus:   {spec.corpus}")
    print("=" * 70)

    if spec.kind in ("checkpoint", "gpt2") and (checkpoint is not None or spec.kind == "gpt2"):
        ckpt = checkpoint if spec.kind == "checkpoint" else None
        model, tokenizer, meta = load_gpt2_path_model(checkpoint=ckpt, device=device)
    elif spec.kind == "checkpoint":
        model, tokenizer, meta = load_gpt2_path_model(checkpoint=checkpoint, device=device)
    else:
        assert spec.hf_id is not None or hf_model is not None
        model, tokenizer, meta = load_hf_causal_lm(
            spec.hf_id or hf_model, device=device, dtype=dtype
        )

    max_ctx = int(meta.get("context_length") or resolve_context_length(model))
    print(f"  parameters: {meta.get('parameters', 0):,}")
    print(f"  context:    {max_ctx}")
    print(f"  path:       {meta.get('forward_path')}")
    print("-" * 70)

    model_dir = output_dir / "models" / spec.id
    model_dir.mkdir(parents=True, exist_ok=True)
    task_payloads: dict[str, dict] = {}
    try:
        for task_name in tasks:
            print(f"\n--- {task_name} ({spec.id}) ---")
            payload = run_task(
                task_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_context_length=max_ctx,
                max_examples=max_examples,
                progress_interval=progress_interval,
                local_paths=local_paths,
            )
            task_payloads[task_name] = payload
            atomic_save_json(model_dir / f"{task_name}.json", payload)
            upsert_model_summary(output_dir, spec, meta, task_payloads)
            score = payload.get("score")
            if score is not None:
                print(f"  {task_name} {payload.get('metric_primary')}: {float(score) * 100:.2f}%")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(output_dir)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in DEFAULT_SUITE_TASKS]
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}. Choose from {list(DEFAULT_SUITE_TASKS)}")
    local_paths = _local_paths(args.local_dir)
    dtype = _dtype(args.dtype)

    jobs: list[tuple] = []
    if args.protocol_all:
        for spec in PROTOCOL_MODELS:
            ckpt = spec.checkpoint if spec.kind == "checkpoint" else None
            jobs.append((spec, ckpt, spec.hf_id))
    elif args.checkpoint:
        spec = lookup_protocol_model(args.model_id or "basikgpt-2p5b")
        if spec is None or spec.kind != "checkpoint":
            from basikgpt.evaluation.suite import ProtocolModelSpec

            spec = ProtocolModelSpec(
                id=args.model_id or "basikgpt-2p5b",
                kind="checkpoint",
                params_label="124M",
                family="GPT-2 Small (basikGPT)",
                corpus="checkpoint",
                checkpoint=args.checkpoint,
            )
        jobs.append((spec, args.checkpoint, None))
    else:
        hf_id = args.hf_model
        if hf_id in ("gpt2", "openai-community/gpt2"):
            spec = lookup_protocol_model("gpt2") or spec_for_hf_id("openai-community/gpt2")
            if args.model_id:
                spec = replace(spec, id=args.model_id)
            jobs.append((spec, None, spec.hf_id))
        else:
            spec = spec_for_hf_id(hf_id)
            if args.model_id:
                spec = replace(spec, id=args.model_id)
            jobs.append((spec, None, spec.hf_id))

    print(f"Output dir: {output_dir}")
    print(f"Device:     {device}")
    print(f"Tasks:      {', '.join(tasks)}")
    print(f"Jobs:       {len(jobs)}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for spec, ckpt, hf_id in jobs:
        if spec.kind == "checkpoint" and ckpt and not Path(ckpt).exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        try:
            _run_one(
                spec,
                checkpoint=ckpt,
                hf_model=hf_id,
                device=device,
                dtype=dtype,
                tasks=tasks,
                output_dir=output_dir,
                max_examples=args.max_examples,
                progress_interval=args.progress_interval,
                local_paths=local_paths,
            )
        except Exception as exc:
            print(f"FAILED {spec.id}: {type(exc).__name__}: {exc}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not args.protocol_all:
                raise

    print(f"\nDone. Report: {output_dir / 'REPORT.md'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
