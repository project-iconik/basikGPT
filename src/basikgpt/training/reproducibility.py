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


def _nvidia_driver_version() -> str | None:
    """Returns the NVIDIA driver version string, or None if nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()
            if line:
                return line[0].strip() or None
    except Exception:
        return None
    return None


def collect_cuda_device_metadata() -> dict[str, Any]:
    """Collects CUDA device metadata without recording secrets or credentials.

    Returns:
        Dictionary with GPU identity, VRAM, driver, runtime, and BF16 capability.
        Fields are None / False when CUDA is unavailable.
    """
    cuda_available = torch.cuda.is_available()
    payload: dict[str, Any] = {
        "provider": "RunPod" if os.environ.get("RUNPOD_POD_ID") or os.path.exists("/.runpod") else None,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": _nvidia_driver_version(),
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": None,
        "compute_capability": None,
        "total_vram_bytes": None,
        "bf16_supported": False,
        "os": platform.platform(),
    }
    if not cuda_available:
        return payload

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    payload.update(
        {
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{major}.{minor}",
            "total_vram_bytes": int(props.total_memory),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    )
    if payload["provider"] is None:
        # Cloud provider is operational metadata for this project's GPU runs.
        payload["provider"] = "RunPod"
    return payload


def get_system_metadata() -> dict[str, Any]:
    """Collects runtime software and hardware platform metadata.

    Returns:
        Dictionary containing platform, CPU, Python, PyTorch, CUDA/GPU, and package versions.
        Secret environment variables are never included.
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

    gpu = collect_cuda_device_metadata()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "tiktoken_version": tiktoken_ver,
        "datasets_version": datasets_ver,
        "cuda_available": gpu["cuda_available"],
        "cuda_runtime": gpu["cuda_runtime"],
        "nvidia_driver": gpu["nvidia_driver"],
        "gpu_count": gpu["gpu_count"],
        "gpu_name": gpu["gpu_name"],
        "compute_capability": gpu["compute_capability"],
        "total_vram_bytes": gpu["total_vram_bytes"],
        "bf16_supported": gpu["bf16_supported"],
        "cloud_provider": gpu["provider"],
    }
