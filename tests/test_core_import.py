"""Tests that the core package imports without optional data dependencies."""

import os
import subprocess
import sys
from pathlib import Path


def test_core_import_without_optional_data_deps() -> None:
    """Verifies `import basikgpt` works when tiktoken and datasets are unavailable."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    code = r"""
import sys

class _BlockOptionalDataDeps:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in ("tiktoken", "datasets"):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, _BlockOptionalDataDeps())
import basikgpt
from basikgpt import GPT, GPTConfig, Trainer
assert GPTConfig.gpt2_small().n_layers == 12
print("ok")
"""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    assert "ok" in result.stdout
