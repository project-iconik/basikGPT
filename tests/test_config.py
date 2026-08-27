"""Unit tests for GPTConfig and configuration presets in basikGPT."""

from dataclasses import FrozenInstanceError
import pytest

from basikgpt.config import GPTConfig


def test_default_config_matches_gpt2_small() -> None:
    """Verifies that default GPTConfig matches canonical GPT-2 Small configuration."""
    config = GPTConfig()

    assert config.vocab_size == 50_257
    assert config.context_length == 1_024
    assert config.n_layers == 12
    assert config.n_heads == 12
    assert config.d_model == 768
    assert config.d_ff == 3_072
    assert config.dropout == 0.1
    assert config.layer_norm_eps == 1e-5
    assert config.bias is True
    assert config.attention_backend == "eager"
    assert config.head_dim == 64


def test_gpt2_small_preset() -> None:
    """Verifies explicit GPTConfig.gpt2_small() preset values."""
    config = GPTConfig.gpt2_small()

    assert config.vocab_size == 50_257
    assert config.context_length == 1_024
    assert config.n_layers == 12
    assert config.n_heads == 12
    assert config.d_model == 768
    assert config.d_ff == 3_072
    assert config.head_dim == 64
    assert config.bias is True
    assert config.attention_backend == "eager"


def test_gpt2_presets_scaling() -> None:
    """Verifies architectural scaling parameters across Medium, Large, and XL presets."""
    medium = GPTConfig.gpt2_medium()
    assert medium.n_layers == 24
    assert medium.n_heads == 16
    assert medium.d_model == 1_024
    assert medium.d_ff == 4_096
    assert medium.head_dim == 64

    large = GPTConfig.gpt2_large()
    assert large.n_layers == 36
    assert large.n_heads == 20
    assert large.d_model == 1_280
    assert large.d_ff == 5_120
    assert large.head_dim == 64

    xl = GPTConfig.gpt2_xl()
    assert xl.n_layers == 48
    assert xl.n_heads == 25
    assert xl.d_model == 1_600
    assert xl.d_ff == 6_400
    assert xl.head_dim == 64


def test_preset_overrides() -> None:
    """Verifies that preset kwargs can override specific parameters safely."""
    config = GPTConfig.gpt2_small(dropout=0.0, attention_backend="sdpa")
    assert config.dropout == 0.0
    assert config.attention_backend == "sdpa"
    assert config.d_model == 768  # other defaults preserved


def test_immutability() -> None:
    """Verifies that GPTConfig is a frozen dataclass and prevents field mutations."""
    config = GPTConfig.gpt2_small()
    with pytest.raises(FrozenInstanceError):
        config.d_model = 1024  # type: ignore[misc]


def test_validation_divisibility() -> None:
    """Verifies ValueError when d_model is not divisible by n_heads."""
    with pytest.raises(ValueError, match="must be divisible by n_heads"):
        GPTConfig(d_model=768, n_heads=13)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("vocab_size", 0),
        ("vocab_size", -10),
        ("context_length", 0),
        ("context_length", -1024),
        ("n_layers", 0),
        ("n_layers", -1),
        ("n_heads", 0),
        ("n_heads", -4),
        ("d_model", 0),
        ("d_model", -768),
        ("d_ff", 0),
        ("d_ff", -3072),
        ("layer_norm_eps", 0.0),
        ("layer_norm_eps", -1e-5),
    ],
)
def test_validation_positive_dimensions(field: str, invalid_value: int | float) -> None:
    """Verifies ValueError for zero or negative structural dimensions."""
    kwargs = {field: invalid_value}
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


@pytest.mark.parametrize("invalid_dropout", [-0.1, 1.0, 1.5])
def test_validation_dropout_bounds(invalid_dropout: float) -> None:
    """Verifies ValueError for dropout values outside [0.0, 1.0)."""
    with pytest.raises(ValueError, match="dropout must be in the half-open interval"):
        GPTConfig(dropout=invalid_dropout)


def test_validation_attention_backend() -> None:
    """Verifies ValueError when an unsupported attention backend is specified."""
    with pytest.raises(ValueError, match="attention_backend must be either 'eager' or 'sdpa'"):
        GPTConfig(attention_backend="flash_attention_unsupported")  # type: ignore[arg-type]


def test_analytical_parameter_counts_gpt2_small() -> None:
    """Verifies exact analytical parameter calculation for GPT-2 Small against official specification."""
    config = GPTConfig.gpt2_small()

    # Embedding: (50,257 + 1,024) * 768 = 39,383,808
    expected_emb = 39_383_808
    assert config.num_embedding_parameters() == expected_emb

    # Transformer Blocks:
    # 12 blocks * 7,087,872 per block + 1,536 final LN = 85,056,000
    expected_transformer = 85_056_000
    assert config.num_transformer_parameters() == expected_transformer

    # Total with weight tying (LM Head shares weight with wte): 124,439,808 (~124.44M)
    expected_total_tied = 124_439_808
    assert config.num_total_parameters(tied_weights=True) == expected_total_tied

    # Total without weight tying (LM head adds 50,257 * 768 = 38,597,376): 163,037,184
    expected_total_untied = 163_037_184
    assert config.num_total_parameters(tied_weights=False) == expected_total_untied


def test_analytical_parameter_counts_without_bias() -> None:
    """Verifies analytical parameter calculation changes correctly when bias=False."""
    config_bias = GPTConfig.gpt2_small(bias=True)
    config_no_bias = GPTConfig.gpt2_small(bias=False)

    assert config_no_bias.num_embedding_parameters() == config_bias.num_embedding_parameters()
    assert config_no_bias.num_transformer_parameters() < config_bias.num_transformer_parameters()
