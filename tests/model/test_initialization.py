"""Unit and statistical verification tests for GPT-2 weight initialization in basikGPT.

Tests:
1. Linear and Embedding weight distribution properties (mean ~ 0, std ~ initializer_range).
2. Residual projection scaled initialization (std ~ initializer_range / sqrt(2 * n_layers)).
3. LayerNorm scale (1.0) and bias (0.0) exact initializations.
4. Linear bias (0.0) exact initializations.
5. Weight tying identity preservation and parameter deduplication.
6. Same-seed initialization determinism (bitwise identical tensors).
7. Different-seed initialization diversity (distinct tensors).
8. Exact GPT-2 Small parameter count preservation (124,439,808).
9. Pretrained checkpoint loading cleanly overwrites random initialization.
10. Theoretical initial cross-entropy loss sanity (~ ln(vocab_size)).
11. Custom initializer_range scalability and validation error handling.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
from basikgpt.model.gpt import GPT
from basikgpt.training.reproducibility import seed_everything


# =====================================================================
# 1. Statistical Distribution Verification
# =====================================================================

def test_linear_and_embedding_distribution_statistics() -> None:
    """Verifies that non-residual Linear weights and Embedding tables follow N(0, 0.02^2)."""
    seed_everything(42)
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)

    # 1. Token Embeddings (wte): 50,257 x 768 = ~3.86M parameters
    wte_mean = float(model.wte.weight.mean().item())
    wte_std = float(model.wte.weight.std().item())
    assert abs(wte_mean) < 1e-3, f"wte mean {wte_mean} deviated from 0.0"
    assert pytest.approx(wte_std, abs=1e-3) == cfg.initializer_range

    # 2. Positional Embeddings (wpe): 1,024 x 768 = ~786k parameters
    wpe_mean = float(model.wpe.weight.mean().item())
    wpe_std = float(model.wpe.weight.std().item())
    assert abs(wpe_mean) < 1e-3, f"wpe mean {wpe_mean} deviated from 0.0"
    assert pytest.approx(wpe_std, abs=1e-3) == cfg.initializer_range

    # 3. Attention QKV Linear Projection: 768 x (3*768) = ~1.77M parameters
    qkv_weight = model.blocks[0].attn.qkv_proj.weight
    qkv_mean = float(qkv_weight.mean().item())
    qkv_std = float(qkv_weight.std().item())
    assert abs(qkv_mean) < 1e-3, f"qkv_proj mean {qkv_mean} deviated from 0.0"
    assert pytest.approx(qkv_std, abs=1e-3) == cfg.initializer_range

    # 4. MLP Expansion Linear (fc_in): 768 x 3,072 = ~2.36M parameters
    fc_in_weight = model.blocks[0].mlp.fc_in.weight
    fc_in_mean = float(fc_in_weight.mean().item())
    fc_in_std = float(fc_in_weight.std().item())
    assert abs(fc_in_mean) < 1e-3, f"fc_in mean {fc_in_mean} deviated from 0.0"
    assert pytest.approx(fc_in_std, abs=1e-3) == cfg.initializer_range


def test_residual_projection_scaling_statistics() -> None:
    """Verifies that residual projections (attn.out_proj and mlp.fc_out) are scaled by 1/sqrt(2*L)."""
    seed_everything(42)
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)

    # Expected standard deviation for L = 12: 0.02 / sqrt(24) ≈ 0.00408248
    expected_resid_std = cfg.initializer_range / math.sqrt(2 * cfg.n_layers)

    for i, block in enumerate(model.blocks):
        # 1. Attention output projection (out_proj)
        out_proj_w = block.attn.out_proj.weight
        out_mean = float(out_proj_w.mean().item())
        out_std = float(out_proj_w.std().item())
        assert abs(out_mean) < 1e-3, f"Block {i} out_proj mean {out_mean} deviated from 0.0"
        assert pytest.approx(out_std, abs=3e-4) == expected_resid_std, (
            f"Block {i} out_proj std {out_std:.6f} deviated from expected {expected_resid_std:.6f}"
        )

        # 2. MLP contraction projection (fc_out)
        fc_out_w = block.mlp.fc_out.weight
        fc_mean = float(fc_out_w.mean().item())
        fc_out_std = float(fc_out_w.std().item())
        assert abs(fc_mean) < 1e-3, f"Block {i} fc_out mean {fc_mean} deviated from 0.0"
        assert pytest.approx(fc_out_std, abs=3e-4) == expected_resid_std, (
            f"Block {i} fc_out std {fc_out_std:.6f} deviated from expected {expected_resid_std:.6f}"
        )

        # Confirm residual std is significantly smaller than base std
        assert out_std < (cfg.initializer_range / 2.0)
        assert fc_out_std < (cfg.initializer_range / 2.0)


# =====================================================================
# 2. LayerNorm and Bias Deterministic Initialization
# =====================================================================

def test_layernorm_initialization() -> None:
    """Verifies that all LayerNorm modules are initialized with weight=1.0 and bias=0.0."""
    cfg = GPTConfig.gpt2_small(bias=True)
    model = GPT(cfg)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight)), (
                f"LayerNorm {name} weight not all ones"
            )
            if module.bias is not None:
                assert torch.equal(module.bias, torch.zeros_like(module.bias)), (
                    f"LayerNorm {name} bias not all zeros"
                )


def test_linear_bias_initialization() -> None:
    """Verifies that all Linear layer biases are initialized to exact zero."""
    cfg = GPTConfig.gpt2_small(bias=True)
    model = GPT(cfg)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias)), (
                f"Linear layer {name} bias not all zeros"
            )


# =====================================================================
# 3. Weight Tying and Parameter Invariants
# =====================================================================

def test_weight_tying_identity_and_deduplication() -> None:
    """Verifies that lm_head.weight shares identical object reference with wte.weight."""
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)

    assert model.lm_head.weight is model.wte.weight
    assert id(model.lm_head.weight) == id(model.wte.weight)

    # In-place modification should be reflected in both
    orig_val = model.wte.weight[0, 0].item()
    with torch.no_grad():
        model.wte.weight[0, 0].add_(1.0)
    assert model.lm_head.weight[0, 0].item() == model.wte.weight[0, 0].item()
    assert pytest.approx(model.lm_head.weight[0, 0].item(), abs=1e-5) == orig_val + 1.0


def test_parameter_count_gpt2_small_preserved() -> None:
    """Verifies that the canonical GPT-2 Small parameter counts remain exactly intact."""
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)

    assert model.num_parameters() == 124_439_808
    assert model.num_parameters(non_embedding=True) == 85_056_000


# =====================================================================
# 4. Reproducibility & Determinism
# =====================================================================

def test_initialization_seed_reproducibility() -> None:
    """Verifies that identical seed produces bitwise identical model parameters."""
    cfg = GPTConfig.gpt2_small()

    seed_everything(1337)
    m1 = GPT(cfg)

    seed_everything(1337)
    m2 = GPT(cfg)

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} diverged with same seed"


def test_different_seeds_produce_different_weights() -> None:
    """Verifies that different seeds produce distinct parameter weights."""
    cfg = GPTConfig.gpt2_small()

    seed_everything(42)
    m1 = GPT(cfg)

    seed_everything(999)
    m2 = GPT(cfg)

    assert not torch.equal(m1.wte.weight, m2.wte.weight)
    assert not torch.equal(m1.blocks[0].attn.qkv_proj.weight, m2.blocks[0].attn.qkv_proj.weight)


# =====================================================================
# 5. Checkpoint Overwrite & Loss Sanity
# =====================================================================

def test_checkpoint_load_overwrites_init() -> None:
    """Verifies that loading official pretrained checkpoint completely overwrites random initialization."""
    cfg = GPTConfig.gpt2_small(dropout=0.0)
    model = GPT(cfg)

    init_wte = model.wte.weight.clone()

    load_hf_gpt2_weights(model, "openai-community/gpt2")

    # Weights must be overwritten
    assert not torch.equal(model.wte.weight, init_wte)
    # Weight tying must be preserved after checkpoint loading
    assert model.lm_head.weight is model.wte.weight


def test_initial_loss_theoretical_sanity() -> None:
    """Verifies that initial cross-entropy loss is finite and close to theoretical ln(vocab_size)."""
    seed_everything(42)
    cfg = GPTConfig.gpt2_small()
    model = GPT(cfg)
    model.eval()

    V = cfg.vocab_size
    theoretical_loss = math.log(V)  # ~10.8249

    # Generate random batch of token inputs and next-token targets
    x = torch.randint(0, V, (2, 128))
    y = torch.randint(0, V, (2, 128))

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))

    loss_val = float(loss.item())
    assert torch.isfinite(loss), f"Initial loss is non-finite: {loss_val}"
    # Loss should be near ln(V) within a reasonable statistical bound
    assert abs(loss_val - theoretical_loss) < 1.0, (
        f"Initial loss {loss_val:.4f} deviated too far from ln(V)={theoretical_loss:.4f}"
    )


# =====================================================================
# 6. Custom Initializer Range & Configuration Validation
# =====================================================================

def test_custom_initializer_range() -> None:
    """Verifies that custom initializer_range scales the resulting parameter distributions."""
    seed_everything(42)
    custom_std = 0.05
    cfg = GPTConfig(
        vocab_size=1000,
        context_length=64,
        n_layers=4,
        n_heads=4,
        d_model=64,
        d_ff=256,
        initializer_range=custom_std,
    )
    model = GPT(cfg)

    wte_std = float(model.wte.weight.std().item())
    assert pytest.approx(wte_std, rel=0.1) == custom_std

    expected_resid_std = custom_std / math.sqrt(2 * 4)  # 0.05 / sqrt(8) ≈ 0.01768
    out_std = float(model.blocks[0].attn.out_proj.weight.std().item())
    assert pytest.approx(out_std, rel=0.1) == expected_resid_std


def test_invalid_initializer_range_raises() -> None:
    """Verifies that non-positive initializer_range values raise ValueError."""
    with pytest.raises(ValueError, match="initializer_range must be strictly positive"):
        GPTConfig(initializer_range=0.0)

    with pytest.raises(ValueError, match="initializer_range must be strictly positive"):
        GPTConfig(initializer_range=-0.02)
