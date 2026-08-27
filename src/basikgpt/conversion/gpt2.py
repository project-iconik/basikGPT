"""Checkpoint converter and weight loader for HuggingFace GPT-2 models.

Maps parameters from HuggingFace/OpenAI `GPT2LMHeadModel` format to `basikGPT.GPT`,
handling Conv1D weight transpositions and preserving Weight Tying.
"""

from collections.abc import Mapping
from typing import Any
import torch
import torch.nn as nn

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT


def validate_hf_config(hf_config: Any, config: GPTConfig) -> None:
    """Validates that a HuggingFace GPT2Config matches the target basikGPT configuration.

    Args:
        hf_config: HuggingFace GPT2Config object or dictionary-like config object.
        config: Target basikGPT GPTConfig instance.

    Raises:
        ValueError: If any architectural dimension or hyperparameter diverges.
    """
    hf_vocab_size = getattr(hf_config, "vocab_size", None)
    hf_n_positions = getattr(hf_config, "n_positions", getattr(hf_config, "max_position_embeddings", None))
    hf_n_layer = getattr(hf_config, "n_layer", getattr(hf_config, "num_hidden_layers", None))
    hf_n_head = getattr(hf_config, "n_head", getattr(hf_config, "num_attention_heads", None))
    hf_n_embd = getattr(hf_config, "n_embd", getattr(hf_config, "hidden_size", None))
    hf_n_inner = getattr(hf_config, "n_inner", None)
    if hf_n_inner is None and hf_n_embd is not None:
        hf_n_inner = 4 * hf_n_embd
    hf_layer_norm_eps = getattr(hf_config, "layer_norm_epsilon", getattr(hf_config, "layer_norm_eps", None))

    mismatches = []
    if hf_vocab_size is not None and hf_vocab_size != config.vocab_size:
        mismatches.append(f"vocab_size: HF={hf_vocab_size} vs basikGPT={config.vocab_size}")
    if hf_n_positions is not None and hf_n_positions != config.context_length:
        mismatches.append(f"context_length: HF={hf_n_positions} vs basikGPT={config.context_length}")
    if hf_n_layer is not None and hf_n_layer != config.n_layers:
        mismatches.append(f"n_layers: HF={hf_n_layer} vs basikGPT={config.n_layers}")
    if hf_n_head is not None and hf_n_head != config.n_heads:
        mismatches.append(f"n_heads: HF={hf_n_head} vs basikGPT={config.n_heads}")
    if hf_n_embd is not None and hf_n_embd != config.d_model:
        mismatches.append(f"d_model: HF={hf_n_embd} vs basikGPT={config.d_model}")
    if hf_n_inner is not None and hf_n_inner != config.d_ff:
        mismatches.append(f"d_ff: HF={hf_n_inner} vs basikGPT={config.d_ff}")
    if hf_layer_norm_eps is not None and abs(hf_layer_norm_eps - config.layer_norm_eps) > 1e-9:
        mismatches.append(f"layer_norm_eps: HF={hf_layer_norm_eps} vs basikGPT={config.layer_norm_eps}")

    if mismatches:
        raise ValueError(
            "HuggingFace model configuration is incompatible with basikGPT config:\n  "
            + "\n  ".join(mismatches)
        )


def convert_hf_gpt2_state_dict(
    hf_state_dict: Mapping[str, torch.Tensor],
    config: GPTConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Converts a HuggingFace GPT-2 state dictionary into a basikGPT state dictionary.

    Performs parameter name remapping and transposes 2D weights originating
    from HuggingFace's Conv1D module (which stores weights as (in_features, out_features))
    to PyTorch's nn.Linear layout (which expects (out_features, in_features)).

    Args:
        hf_state_dict: Source state dictionary from HuggingFace GPT2LMHeadModel.
        config: Optional GPTConfig to validate shapes against.

    Returns:
        A dictionary containing keys and tensors matching basikGPT.GPT.state_dict().

    Raises:
        KeyError: If required parameter keys are missing in the source dictionary.
        ValueError: If tensor shapes do not match expected dimensions.
    """
    converted: dict[str, torch.Tensor] = {}
    used_hf_keys: set[str] = set()

    # Determine number of layers from config or state_dict keys
    if config is not None:
        n_layers = config.n_layers
    else:
        layer_indices = [
            int(k.split(".")[2])
            for k in hf_state_dict.keys()
            if k.startswith("transformer.h.") and k.split(".")[2].isdigit()
        ]
        n_layers = max(layer_indices) + 1 if layer_indices else 12

    # 1. Token Embeddings: (V, C) -> (V, C) [Direct copy]
    if "transformer.wte.weight" not in hf_state_dict:
        raise KeyError("Missing required key 'transformer.wte.weight' in HuggingFace state_dict.")
    converted["wte.weight"] = hf_state_dict["transformer.wte.weight"]
    used_hf_keys.add("transformer.wte.weight")

    # 2. Positional Embeddings: (T, C) -> (T, C) [Direct copy]
    if "transformer.wpe.weight" not in hf_state_dict:
        raise KeyError("Missing required key 'transformer.wpe.weight' in HuggingFace state_dict.")
    converted["wpe.weight"] = hf_state_dict["transformer.wpe.weight"]
    used_hf_keys.add("transformer.wpe.weight")

    # 3. Transformer Decoder Blocks (0 to n_layers - 1)
    for l in range(n_layers):
        prefix = f"transformer.h.{l}"
        target_prefix = f"blocks.{l}"

        # 3.1 LayerNorm 1 (weight & bias)
        converted[f"{target_prefix}.ln_1.weight"] = hf_state_dict[f"{prefix}.ln_1.weight"]
        used_hf_keys.add(f"{prefix}.ln_1.weight")
        if f"{prefix}.ln_1.bias" in hf_state_dict:
            converted[f"{target_prefix}.ln_1.bias"] = hf_state_dict[f"{prefix}.ln_1.bias"]
            used_hf_keys.add(f"{prefix}.ln_1.bias")

        # 3.2 Attention QKV Projection (Conv1D weight requires .t(), bias is 1D)
        qkv_w = hf_state_dict[f"{prefix}.attn.c_attn.weight"]
        converted[f"{target_prefix}.attn.qkv_proj.weight"] = qkv_w.t()
        used_hf_keys.add(f"{prefix}.attn.c_attn.weight")

        if f"{prefix}.attn.c_attn.bias" in hf_state_dict:
            converted[f"{target_prefix}.attn.qkv_proj.bias"] = hf_state_dict[f"{prefix}.attn.c_attn.bias"]
            used_hf_keys.add(f"{prefix}.attn.c_attn.bias")

        # 3.3 Attention Output Projection (Conv1D weight requires .t(), bias is 1D)
        out_w = hf_state_dict[f"{prefix}.attn.c_proj.weight"]
        converted[f"{target_prefix}.attn.out_proj.weight"] = out_w.t()
        used_hf_keys.add(f"{prefix}.attn.c_proj.weight")

        if f"{prefix}.attn.c_proj.bias" in hf_state_dict:
            converted[f"{target_prefix}.attn.out_proj.bias"] = hf_state_dict[f"{prefix}.attn.c_proj.bias"]
            used_hf_keys.add(f"{prefix}.attn.c_proj.bias")

        # 3.4 LayerNorm 2 (weight & bias)
        converted[f"{target_prefix}.ln_2.weight"] = hf_state_dict[f"{prefix}.ln_2.weight"]
        used_hf_keys.add(f"{prefix}.ln_2.weight")
        if f"{prefix}.ln_2.bias" in hf_state_dict:
            converted[f"{target_prefix}.ln_2.bias"] = hf_state_dict[f"{prefix}.ln_2.bias"]
            used_hf_keys.add(f"{prefix}.ln_2.bias")

        # 3.5 MLP Expansion (Conv1D weight requires .t(), bias is 1D)
        fc_in_w = hf_state_dict[f"{prefix}.mlp.c_fc.weight"]
        converted[f"{target_prefix}.mlp.fc_in.weight"] = fc_in_w.t()
        used_hf_keys.add(f"{prefix}.mlp.c_fc.weight")

        if f"{prefix}.mlp.c_fc.bias" in hf_state_dict:
            converted[f"{target_prefix}.mlp.fc_in.bias"] = hf_state_dict[f"{prefix}.mlp.c_fc.bias"]
            used_hf_keys.add(f"{prefix}.mlp.c_fc.bias")

        # 3.6 MLP Contraction (Conv1D weight requires .t(), bias is 1D)
        fc_out_w = hf_state_dict[f"{prefix}.mlp.c_proj.weight"]
        converted[f"{target_prefix}.mlp.fc_out.weight"] = fc_out_w.t()
        used_hf_keys.add(f"{prefix}.mlp.c_proj.weight")

        if f"{prefix}.mlp.c_proj.bias" in hf_state_dict:
            converted[f"{target_prefix}.mlp.fc_out.bias"] = hf_state_dict[f"{prefix}.mlp.c_proj.bias"]
            used_hf_keys.add(f"{prefix}.mlp.c_proj.bias")

    # 4. Final LayerNorm
    converted["ln_f.weight"] = hf_state_dict["transformer.ln_f.weight"]
    used_hf_keys.add("transformer.ln_f.weight")
    if "transformer.ln_f.bias" in hf_state_dict:
        converted["ln_f.bias"] = hf_state_dict["transformer.ln_f.bias"]
        used_hf_keys.add("transformer.ln_f.bias")

    # 5. LM Head
    # If lm_head.weight exists in HF state dict, track it
    if "lm_head.weight" in hf_state_dict:
        converted["lm_head.weight"] = hf_state_dict["lm_head.weight"]
        used_hf_keys.add("lm_head.weight")
    else:
        # If lm_head is omitted due to tying in HF, tie it to wte.weight
        converted["lm_head.weight"] = converted["wte.weight"]

    # 6. Check for unexpected keys in source state_dict
    # (Allow attention bias buffers / causal masks if present in older HF versions)
    ignored_suffixes = (".attn.bias", ".attn.masked_bias")
    for k in hf_state_dict.keys():
        if k not in used_hf_keys and not k.endswith(ignored_suffixes):
            raise ValueError(f"Unrecognized or unmapped key in HuggingFace state_dict: '{k}'")

    return converted


def load_hf_gpt2_weights(
    target_model: GPT,
    source: str | Any = "gpt2",
) -> GPT:
    """Loads pretrained HuggingFace GPT-2 weights into a basikGPT.GPT model instance.

    Preserves exact weight tying (target_model.lm_head.weight is target_model.wte.weight).

    Args:
        target_model: The basikGPT.GPT model to populate with pretrained weights.
        source: A model identifier string (e.g. 'gpt2', 'openai-community/gpt2'),
                a loaded transformers.GPT2LMHeadModel instance,
                or a raw state_dict Mapping[str, Tensor].

    Returns:
        The target_model instance with loaded weights in eval mode.
    """
    if isinstance(source, str):
        try:
            from transformers import GPT2LMHeadModel
        except ImportError as e:
            raise ImportError(
                "The 'transformers' library is required to load models by name. "
                "Install it with `pip install transformers`."
            ) from e
        hf_model = GPT2LMHeadModel.from_pretrained(source)
        validate_hf_config(hf_model.config, target_model.config)
        hf_state_dict = hf_model.state_dict()
    elif isinstance(source, nn.Module):
        if hasattr(source, "config"):
            validate_hf_config(source.config, target_model.config)
        hf_state_dict = source.state_dict()
    elif isinstance(source, Mapping):
        hf_state_dict = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    converted_sd = convert_hf_gpt2_state_dict(hf_state_dict, config=target_model.config)

    # In-place parameter copy to preserve PyTorch module graph and weight tying
    target_sd = target_model.state_dict()
    missing_keys = []
    shape_mismatches = []

    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name not in converted_sd:
                missing_keys.append(name)
                continue
            src_tensor = converted_sd[name]
            if param.shape != src_tensor.shape:
                shape_mismatches.append(
                    f"Parameter '{name}': target shape {param.shape} != source shape {src_tensor.shape}"
                )
                continue
            param.copy_(src_tensor)

    if missing_keys:
        raise KeyError(f"Failed to populate parameters in basikGPT: missing keys {missing_keys}")
    if shape_mismatches:
        raise ValueError("Shape mismatches during weight loading:\n  " + "\n  ".join(shape_mismatches))

    # Strict assertion that weight tying was preserved
    if target_model.lm_head.weight is not target_model.wte.weight:
        raise RuntimeError("Weight tying was broken during HuggingFace checkpoint loading!")

    target_model.eval()
    return target_model
