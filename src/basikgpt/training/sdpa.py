"""Runtime SDPA backend inspection and exclusive kernel selection.

PyTorch 2.8 exposes `torch.nn.attention.sdpa_kernel` and `SDPBackend`.
This module does not guess which fused backends exist: it queries the
installed enum and only forces members that are present.

Exclusive `sdpa_kernel(BACKEND)` enables that backend alone. A successful
fallback to a different kernel must not be reported as support for the
requested backend. Callers should treat exceptions as `UNSUPPORTED`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

# Documented fused/math candidates. ERROR / OVERRIDEABLE are not user-facing.
PROBE_SDPA_BACKEND_NAMES: tuple[str, ...] = (
    "MATH",
    "FLASH_ATTENTION",
    "EFFICIENT_ATTENTION",
    "CUDNN_ATTENTION",
)

ALLOWED_SDPA_KERNEL_NAMES: tuple[str, ...] = (
    "auto",
    "math",
    "flash_attention",
    "efficient_attention",
    "cudnn_attention",
)


def normalize_sdpa_kernel_name(name: str) -> str:
    """Normalizes CLI/config kernel names to the canonical lowercase form."""
    return name.strip().lower().replace("-", "_")


def validate_sdpa_kernel_name(name: str) -> str:
    """Validates a TrainingConfig/CLI sdpa_kernel string."""
    normalized = normalize_sdpa_kernel_name(name)
    if normalized not in ALLOWED_SDPA_KERNEL_NAMES:
        raise ValueError(
            f"sdpa_kernel must be one of {ALLOWED_SDPA_KERNEL_NAMES}, got '{name}'."
        )
    return normalized


def _sdp_backend_members() -> dict[str, Any]:
    from torch.nn.attention import SDPBackend

    members = getattr(SDPBackend, "__members__", None)
    if isinstance(members, dict) and members:
        return members
    found: dict[str, Any] = {}
    for name in ("ERROR", "MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE"):
        if hasattr(SDPBackend, name):
            found[name] = getattr(SDPBackend, name)
    return found


def list_probe_sdpa_backends() -> list[str]:
    """Returns documented probe backends that exist in this PyTorch build."""
    members = _sdp_backend_members()
    return [name for name in PROBE_SDPA_BACKEND_NAMES if name in members]


def resolve_sdp_backend(kernel_name: str) -> Any:
    """Maps a canonical kernel name to `SDPBackend`. Raises if missing at runtime."""
    normalized = validate_sdpa_kernel_name(kernel_name)
    if normalized == "auto":
        raise ValueError("resolve_sdp_backend does not accept 'auto'; use sdpa_kernel_context instead.")
    enum_name = normalized.upper()
    members = _sdp_backend_members()
    if enum_name not in members:
        raise RuntimeError(
            f"SDPBackend.{enum_name} is not available in this PyTorch build. "
            f"Present members: {sorted(members)}"
        )
    return members[enum_name]


@contextmanager
def sdpa_kernel_context(kernel_name: str) -> Iterator[None]:
    """Context manager: `auto` is a no-op; otherwise force one SDPBackend exclusively."""
    normalized = validate_sdpa_kernel_name(kernel_name)
    if normalized == "auto":
        with nullcontext():
            yield
        return

    from torch.nn.attention import sdpa_kernel

    backend = resolve_sdp_backend(normalized)
    with sdpa_kernel(backend):
        yield
