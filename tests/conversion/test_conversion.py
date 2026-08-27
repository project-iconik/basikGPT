"""Unit tests for HuggingFace GPT-2 state dict conversion and validation (offline, zero-network)."""

from types import SimpleNamespace
import pytest
import torch

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import (
    convert_hf_gpt2_state_dict,
    load_hf_gpt2_weights,
    validate_hf_config,
)
from basikgpt.model.gpt import GPT


def create_synthetic_hf_state_dict(
    vocab_size: int = 128,
    context_length: int = 32,
    d_model: int = 32,
    d_ff: int = 128,
    n_layers: int = 2,
) -> dict[str, torch.Tensor]:
    """Generates a synthetic HuggingFace GPT-2 state dict for offline testing."""
    sd: dict[str, torch.Tensor] = {
        "transformer.wte.weight": torch.randn(vocab_size, d_model),
        "transformer.wpe.weight": torch.randn(context_length, d_model),
    }

    for l in range(n_layers):
        p = f"transformer.h.{l}"
        sd[f"{p}.ln_1.weight"] = torch.randn(d_model)
        sd[f"{p}.ln_1.bias"] = torch.randn(d_model)
        # Conv1D weights: (in, out)
        sd[f"{p}.attn.c_attn.weight"] = torch.randn(d_model, 3 * d_model)
        sd[f"{p}.attn.c_attn.bias"] = torch.randn(3 * d_model)
        sd[f"{p}.attn.c_proj.weight"] = torch.randn(d_model, d_model)
        sd[f"{p}.attn.c_proj.bias"] = torch.randn(d_model)
        sd[f"{p}.ln_2.weight"] = torch.randn(d_model)
        sd[f"{p}.ln_2.bias"] = torch.randn(d_model)
        sd[f"{p}.mlp.c_fc.weight"] = torch.randn(d_model, d_ff)
        sd[f"{p}.mlp.c_fc.bias"] = torch.randn(d_ff)
        sd[f"{p}.mlp.c_proj.weight"] = torch.randn(d_ff, d_model)
        sd[f"{p}.mlp.c_proj.bias"] = torch.randn(d_model)

    sd["transformer.ln_f.weight"] = torch.randn(d_model)
    sd["transformer.ln_f.bias"] = torch.randn(d_model)
    sd["lm_head.weight"] = sd["transformer.wte.weight"]
    return sd


def test_validate_hf_config_success_and_failure() -> None:
    """Verifies that validate_hf_config accepts valid configs and rejects mismatched ones."""
    cfg = GPTConfig.gpt2_small()

    # Valid config namespace
    valid_hf = SimpleNamespace(
        vocab_size=50257,
        n_positions=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_inner=3072,
        layer_norm_epsilon=1e-5,
    )
    validate_hf_config(valid_hf, cfg)  # Should not raise

    # Invalid vocab_size
    invalid_hf = SimpleNamespace(
        vocab_size=32000,
        n_positions=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_inner=3072,
        layer_norm_epsilon=1e-5,
    )
    with pytest.raises(ValueError, match="vocab_size: HF=32000 vs basikGPT=50257"):
        validate_hf_config(invalid_hf, cfg)


def test_convert_synthetic_state_dict_shapes_and_transposes() -> None:
    """Verifies key mapping and Conv1D weight transpositions on synthetic state dict."""
    V, T, C, F, L = 128, 32, 32, 128, 2
    cfg = GPTConfig(
        vocab_size=V,
        context_length=T,
        d_model=C,
        d_ff=F,
        n_layers=L,
        n_heads=4,
    )
    hf_sd = create_synthetic_hf_state_dict(vocab_size=V, context_length=T, d_model=C, d_ff=F, n_layers=L)

    converted = convert_hf_gpt2_state_dict(hf_sd, config=cfg)

    # 1. Embeddings: Direct copy
    assert torch.equal(converted["wte.weight"], hf_sd["transformer.wte.weight"])
    assert torch.equal(converted["wpe.weight"], hf_sd["transformer.wpe.weight"])

    # 2. Block 0 weights: Transposed Conv1D weights
    assert converted["blocks.0.attn.qkv_proj.weight"].shape == (3 * C, C)
    assert torch.equal(converted["blocks.0.attn.qkv_proj.weight"], hf_sd["transformer.h.0.attn.c_attn.weight"].t())

    assert converted["blocks.0.attn.out_proj.weight"].shape == (C, C)
    assert torch.equal(converted["blocks.0.attn.out_proj.weight"], hf_sd["transformer.h.0.attn.c_proj.weight"].t())

    assert converted["blocks.0.mlp.fc_in.weight"].shape == (F, C)
    assert torch.equal(converted["blocks.0.mlp.fc_in.weight"], hf_sd["transformer.h.0.mlp.c_fc.weight"].t())

    assert converted["blocks.0.mlp.fc_out.weight"].shape == (C, F)
    assert torch.equal(converted["blocks.0.mlp.fc_out.weight"], hf_sd["transformer.h.0.mlp.c_proj.weight"].t())

    # 3. Biases and LayerNorms: Direct copy
    assert torch.equal(converted["blocks.0.ln_1.weight"], hf_sd["transformer.h.0.ln_1.weight"])
    assert torch.equal(converted["blocks.0.ln_1.bias"], hf_sd["transformer.h.0.ln_1.bias"])
    assert torch.equal(converted["ln_f.weight"], hf_sd["transformer.ln_f.weight"])
    assert torch.equal(converted["ln_f.bias"], hf_sd["transformer.ln_f.bias"])


def test_conversion_missing_keys_raises_key_error() -> None:
    """Verifies that missing essential keys in source state_dict raises KeyError."""
    hf_sd = create_synthetic_hf_state_dict()
    del hf_sd["transformer.wte.weight"]

    with pytest.raises(KeyError, match="Missing required key 'transformer.wte.weight'"):
        convert_hf_gpt2_state_dict(hf_sd)


def test_conversion_unexpected_keys_raises_value_error() -> None:
    """Verifies that unrecognized keys in source state_dict raise ValueError."""
    hf_sd = create_synthetic_hf_state_dict()
    hf_sd["transformer.extra_unknown_param.weight"] = torch.randn(10)

    with pytest.raises(ValueError, match="Unrecognized or unmapped key"):
        convert_hf_gpt2_state_dict(hf_sd)


def test_load_hf_gpt2_weights_synthetic_in_place() -> None:
    """Verifies that load_hf_gpt2_weights properly loads weights in-place and preserves weight tying."""
    V, T, C, F, L = 128, 32, 32, 128, 2
    cfg = GPTConfig(
        vocab_size=V,
        context_length=T,
        d_model=C,
        d_ff=F,
        n_layers=L,
        n_heads=4,
    )
    model = GPT(cfg)
    hf_sd = create_synthetic_hf_state_dict(vocab_size=V, context_length=T, d_model=C, d_ff=F, n_layers=L)

    load_hf_gpt2_weights(model, hf_sd)

    # 1. Verify weight values matched converted synthetic state dict
    assert torch.equal(model.wte.weight, hf_sd["transformer.wte.weight"])
    assert torch.equal(model.wpe.weight, hf_sd["transformer.wpe.weight"])
    assert torch.equal(model.blocks[0].attn.qkv_proj.weight, hf_sd["transformer.h.0.attn.c_attn.weight"].t())

    # 2. Strict check that weight tying was preserved
    assert model.lm_head.weight is model.wte.weight
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()


def test_all_12_layers_mapping_coverage() -> None:
    """Verifies 100% parameter mapping coverage for full 12-layer GPT-2 Small configuration."""
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)
    hf_sd = create_synthetic_hf_state_dict(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
    )

    converted = convert_hf_gpt2_state_dict(hf_sd, config=cfg)

    # Check that all target model parameter keys are in converted
    target_param_names = [name for name, _ in model.named_parameters()]
    for name in target_param_names:
        assert name in converted, f"Missing target parameter: {name}"

    # Total unique parameters should match
    load_hf_gpt2_weights(model, hf_sd)
    assert model.num_parameters() == 124_439_808
