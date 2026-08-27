"""Reproducibility utilities, RNG seed management, and provenance metadata collection."""

import os
import platform
import random
import subprocess
from typing import Any
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed: Integer random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_metadata() -> dict[str, Any]:
    """Safely retrieves git commit SHA and working tree dirty status.

    Returns:
        Dictionary with 'commit' (SHA string or None) and 'is_dirty' (bool or None).
    """
    try:
        # 1. Commit SHA
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = commit_res.stdout.strip() if commit_res.returncode == 0 else None

        # 2. Dirty status
        dirty_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        is_dirty = bool(dirty_res.stdout.strip()) if dirty_res.returncode == 0 else None

        return {
            "git_commit": commit,
            "git_dirty": is_dirty,
        }
    except Exception:
        return {
            "git_commit": None,
            "git_dirty": None,
        }


def get_system_metadata() -> dict[str, Any]:
    """Collects runtime software and hardware platform metadata.

    Returns:
        Dictionary containing platform, CPU, Python, PyTorch, and package versions.
    """
    # Safe import checks for optional dependencies
    tiktoken_ver = None
    try:
        import tiktoken
        tiktoken_ver = getattr(tiktoken, "__version__", "unknown")
    except ImportError:
        pass

    datasets_ver = None
    try:
        import datasets
        datasets_ver = getattr(datasets, "__version__", "unknown")
    except ImportError:
        pass

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "tiktoken_version": tiktoken_ver,
        "datasets_version": datasets_ver,
        "cuda_available": torch.cuda.is_available(),
    }
