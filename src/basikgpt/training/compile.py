"""Opt-in torch.compile helpers with raw-model checkpoint ownership.

The Trainer compiles the GPT module only. Optimizer, loss, and checkpoint I/O stay
on the uncompiled `raw_model`. Default training behavior is uncompiled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# PyTorch 2.8 documents default / reduce-overhead / max-autotune /
# max-autotune-no-cudagraphs. This milestone only opts into the first two.
SUPPORTED_COMPILE_MODES: tuple[str, ...] = ("default", "reduce-overhead")
COMPILE_BACKEND = "inductor"


def validate_compile_mode(mode: str) -> str:
    """Returns `mode` if it is supported by this project; otherwise raises ValueError."""
    if mode not in SUPPORTED_COMPILE_MODES:
        raise ValueError(
            f"compile_mode must be one of {SUPPORTED_COMPILE_MODES}, got '{mode}'. "
            "max-autotune is intentionally not enabled in this milestone."
        )
    return mode


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Returns the innermost uncompiled module, stripping torch.compile wrappers.

    `torch.compile` may wrap a module in `OptimizedModule` with an `_orig_mod`
    attribute. Checkpoint ownership must not persist those wrapper keys.
    """
    current: nn.Module = model
    seen: set[int] = set()
    while True:
        marker = id(current)
        if marker in seen:
            return current
        seen.add(marker)
        orig = getattr(current, "_orig_mod", None)
        if orig is None or orig is current:
            return current
        if not isinstance(orig, nn.Module):
            return current
        current = orig


def state_dict_has_compile_wrapper_keys(state_dict: dict[str, object]) -> bool:
    """True if any state_dict key encodes a torch.compile wrapper prefix."""
    return any("_orig_mod." in key or key.startswith("_orig_mod") for key in state_dict)


def compile_model(
    model: nn.Module,
    *,
    mode: str = "default",
    backend: str = COMPILE_BACKEND,
) -> nn.Module:
    """Compiles `model` with TorchInductor. Does not fall back to eager on failure.

    `dynamic=False` matches fixed pretraining shapes (B, T=1024). `fullgraph` is
    left at the PyTorch default so small graph breaks do not hard-fail training.
    """
    validate_compile_mode(mode)
    return torch.compile(model, backend=backend, mode=mode, dynamic=False)
