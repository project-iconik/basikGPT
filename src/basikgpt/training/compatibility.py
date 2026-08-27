"""Dataset and model compatibility validation utilities."""

from pathlib import Path
from typing import Any
from basikgpt.config import GPTConfig
from basikgpt.training.metadata import load_json


def validate_dataset_model_compatibility(
    model_config: GPTConfig,
    manifest: dict[str, Any] | Path | str | None = None,
    requested_context_length: int | None = None,
) -> None:
    """Validates compatibility between the dataset manifest and model configuration.

    Checks:
        1. Context length: requested sequence length T does not exceed model maximum context_length.
        2. Vocabulary size: dataset manifest vocab_size matches model_config.vocab_size.
        3. Tokenizer: dataset manifest encoding is compatible with GPT-2.

    Args:
        model_config: GPTConfig instance of the model.
        manifest: Loaded manifest dictionary or Path to manifest.json (optional).
        requested_context_length: Specific context length requested by user / dataloader.

    Raises:
        ValueError: If any compatibility invariant is violated.
    """
    # 1. Context length check
    if requested_context_length is not None:
        if requested_context_length <= 0:
            raise ValueError(f"Context length must be positive, got {requested_context_length}")
        if requested_context_length > model_config.context_length:
            raise ValueError(
                f"Requested context length ({requested_context_length}) exceeds model "
                f"maximum context length ({model_config.context_length})"
            )

    if manifest is None:
        return

    manifest_dict = load_json(manifest) if isinstance(manifest, (str, Path)) else manifest

    tokenizer_info = manifest_dict.get("tokenizer", {})

    # 2. Vocabulary size check
    if "vocab_size" in tokenizer_info:
        manifest_vocab = tokenizer_info["vocab_size"]
        if manifest_vocab != model_config.vocab_size:
            raise ValueError(
                f"Dataset vocabulary size ({manifest_vocab:,}) does not match model "
                f"vocab_size ({model_config.vocab_size:,})"
            )

    # 3. Tokenizer encoding check
    encoding = tokenizer_info.get("encoding") or tokenizer_info.get("encoding_name")
    if encoding is not None:
        if model_config.vocab_size == 50257 and encoding != "gpt2":
            raise ValueError(
                f"Dataset tokenizer encoding '{encoding}' is incompatible with canonical GPT-2 configuration"
            )
