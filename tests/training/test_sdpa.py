"""CPU-safe tests for SDPA kernel name validation and runtime backend listing."""

import pytest

from basikgpt.training.sdpa import (
    ALLOWED_SDPA_KERNEL_NAMES,
    list_probe_sdpa_backends,
    normalize_sdpa_kernel_name,
    validate_sdpa_kernel_name,
)


def test_normalize_and_validate_sdpa_kernel_names() -> None:
    assert normalize_sdpa_kernel_name("FLASH-ATTENTION") == "flash_attention"
    assert validate_sdpa_kernel_name("auto") == "auto"
    assert validate_sdpa_kernel_name("MATH") == "math"
    with pytest.raises(ValueError, match="sdpa_kernel"):
        validate_sdpa_kernel_name("triton")


def test_list_probe_sdpa_backends_is_documented_subset() -> None:
    names = list_probe_sdpa_backends()
    allowed_enum = {item.upper() for item in ALLOWED_SDPA_KERNEL_NAMES if item != "auto"}
    assert set(names).issubset(allowed_enum)
    # MATH is the portable backend and is present in PyTorch 2.8.
    assert "MATH" in names or names == []
