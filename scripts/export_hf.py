"""Export a basikGPT checkpoint to HuggingFace GPT2LMHeadModel format.

Writes safetensors + GPT-2 tokenizer + model card. Optionally pushes to the Hub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from basikgpt.conversion.gpt2 import (  # noqa: E402
    convert_basikgpt_state_dict_to_hf,
    gpt_config_to_hf_kwargs,
)
from basikgpt.model.gpt import GPT  # noqa: E402
from basikgpt.training.checkpoint import load_model_from_checkpoint  # noqa: E402

VARIANT_DEFAULTS: dict[str, dict[str, str]] = {
    "v1.0": {
        "repo_id": "project-iconik/basikGPT-1-v1.0",
        "card": "docs/hf/MODEL_CARD_v1.0.md",
        "checkpoint": "runs/main_2p5b/step-00038147.pt",
    },
    "v1.1": {
        "repo_id": "project-iconik/basikGPT-1-v1.1",
        "card": "docs/hf/MODEL_CARD_v1.1.md",
        "checkpoint": "runs/cont_5b_mix/step-00076294.pt",
    },
}

HF_BUFFER_SUFFIXES = (".attn.bias", ".attn.masked_bias")
LOGIT_MAX_ABS_TOL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a basikGPT .pt checkpoint to a HuggingFace GPT-2 directory."
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_DEFAULTS),
        required=True,
        help="Release id: v1.0 (2.5B tokens) or v1.1 (5B tokens).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a basikGPT .pt checkpoint (default depends on --variant).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local export directory (default: /tmp/hf-basikGPT-1-<variant>).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hub repo id for --push (default depends on --variant).",
    )
    parser.add_argument(
        "--card",
        type=str,
        default=None,
        help="Model card markdown to copy as README.md.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for load and logit check (default: cpu).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the export directory to Hugging Face Hub.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip basikGPT vs exported HuggingFace logit comparison.",
    )
    return parser.parse_args()


def build_hf_model(basik_model: GPT):
    """Build a GPT2LMHeadModel populated from a loaded basikGPT model."""
    from transformers import GPT2Config, GPT2LMHeadModel

    hf_config = GPT2Config(**gpt_config_to_hf_kwargs(basik_model.config))
    hf_model = GPT2LMHeadModel(hf_config)
    hf_state = convert_basikgpt_state_dict_to_hf(basik_model.state_dict(), basik_model.config)
    missing, unexpected = hf_model.load_state_dict(hf_state, strict=False)
    unexpected_real = [k for k in unexpected if not k.endswith(HF_BUFFER_SUFFIXES)]
    missing_real = [k for k in missing if not k.endswith(HF_BUFFER_SUFFIXES)]
    if unexpected_real:
        raise ValueError(f"Unexpected keys while loading HF model: {unexpected_real}")
    if missing_real:
        raise ValueError(f"Missing keys while loading HF model: {missing_real}")
    hf_model.tie_weights()
    hf_model.eval()
    return hf_model


def compare_logits(basik_model: GPT, hf_model, device: torch.device, seq_len: int = 32) -> dict[str, float]:
    """Compare next-token logits on a short random prompt."""
    vocab = basik_model.config.vocab_size
    seq_len = min(seq_len, basik_model.config.context_length)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, seq_len), device=device)
    with torch.no_grad():
        basik_logits = basik_model(input_ids)
        hf_out = hf_model(input_ids)
        hf_logits = hf_out.logits if hasattr(hf_out, "logits") else hf_out
    diff = (basik_logits.float() - hf_logits.float()).abs()
    metrics = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }
    if metrics["max_abs"] > LOGIT_MAX_ABS_TOL:
        raise AssertionError(
            f"Logit mismatch between basikGPT and HuggingFace export: "
            f"max_abs={metrics['max_abs']:.3e} (tol={LOGIT_MAX_ABS_TOL})"
        )
    return metrics


def assert_no_optimizer_in_safetensors(export_dir: Path) -> None:
    """Fail if the safetensors snapshot contains optimizer / trainer keys."""
    from safetensors import safe_open

    st_path = export_dir / "model.safetensors"
    if not st_path.is_file():
        shards = sorted(export_dir.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors file under {export_dir}")
        st_path = shards[0]
    forbidden = ("optimizer", "scaler", "rng_states", "global_step")
    with safe_open(str(st_path), framework="pt") as handle:
        keys = list(handle.keys())
    bad = [k for k in keys if any(part in k.lower() for part in forbidden)]
    if bad:
        raise ValueError(f"Safetensors contains non-weight keys: {bad}")
    if any(k.startswith("blocks.") or k == "wte.weight" for k in keys):
        raise ValueError("Safetensors still uses basikGPT key names; expected HuggingFace GPT-2 keys.")


def export_directory(
    basik_model: GPT,
    output: Path,
    card_path: Path,
    tokenizer_id: str = "openai-community/gpt2",
) -> None:
    """Write GPT2LMHeadModel weights, tokenizer, and README.md."""
    from transformers import GPT2TokenizerFast

    output.mkdir(parents=True, exist_ok=True)
    hf_model = build_hf_model(basik_model)
    hf_model.save_pretrained(output, safe_serialization=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_id)
    tokenizer.save_pretrained(output)
    shutil.copyfile(card_path, output / "README.md")


def push_folder(export_dir: Path, repo_id: str) -> None:
    """Upload a local export directory to the Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add basikGPT-1 HuggingFace GPT-2 export",
    )


def main() -> None:
    args = parse_args()
    defaults = VARIANT_DEFAULTS[args.variant]
    checkpoint = Path(args.checkpoint or (repo_root / defaults["checkpoint"]))
    output = Path(args.output or f"/tmp/hf-basikGPT-1-{args.variant}")
    repo_id = args.repo_id or defaults["repo_id"]
    card_path = Path(args.card or (repo_root / defaults["card"]))
    device = torch.device(args.device)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not card_path.is_file():
        raise FileNotFoundError(f"Model card not found: {card_path}")

    print(f"Loading {checkpoint} on {device} ...")
    basik_model, meta = load_model_from_checkpoint(checkpoint, device=device)
    print(
        f"Loaded step={meta.get('global_step')} tokens_seen={meta.get('tokens_seen')} "
        f"params={basik_model.num_parameters():,}"
    )

    print(f"Writing HuggingFace snapshot to {output} ...")
    if output.exists():
        shutil.rmtree(output)
    export_directory(basik_model, output, card_path)
    assert_no_optimizer_in_safetensors(output)
    print("Safetensors: weight-only HuggingFace GPT-2 keys.")

    if not args.skip_verify:
        from transformers import GPT2LMHeadModel

        hf_reloaded = GPT2LMHeadModel.from_pretrained(output)
        hf_reloaded.to(device)
        hf_reloaded.eval()
        metrics = compare_logits(basik_model, hf_reloaded, device)
        print(f"Logit check vs original checkpoint: max_abs={metrics['max_abs']:.3e} mean_abs={metrics['mean_abs']:.3e}")

    if args.push:
        print(f"Pushing to https://huggingface.co/{repo_id} ...")
        push_folder(output, repo_id)
        print(f"Uploaded: https://huggingface.co/{repo_id}")
    else:
        print(f"Local export ready (no --push): {output}")


if __name__ == "__main__":
    main()
