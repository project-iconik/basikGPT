"""CPU-safe tests for performance JSON helpers and speedup/break-even attachment."""

from pathlib import Path

from basikgpt.training.performance import (
    append_jsonl,
    attach_speedup,
    save_performance_summary,
)
from basikgpt.training.metadata import load_json


def test_attach_speedup_and_break_even() -> None:
    faster = attach_speedup(
        {
            "status": "PASS",
            "compiled": True,
            "compile_seconds": 10.0,
            "steady_state_tokens_per_second": 2000.0,
        },
        1000.0,
    )
    assert faster["speedup_vs_baseline"] == 2.0
    assert faster["break_even_tokens"] == 20_000.0

    slower = attach_speedup(
        {
            "status": "PASS",
            "compiled": True,
            "compile_seconds": 10.0,
            "steady_state_tokens_per_second": 500.0,
        },
        1000.0,
    )
    assert slower["speedup_vs_baseline"] == 0.5
    assert slower["break_even_tokens"] is None

    failed = attach_speedup({"status": "COMPILE_FAILED", "compiled": True}, 1000.0)
    assert failed["speedup_vs_baseline"] is None


def test_performance_summary_and_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks.jsonl"
    append_jsonl(path, {"status": "PASS", "B": 1})
    append_jsonl(path, {"status": "OOM", "B": 32})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    summary = save_performance_summary(tmp_path / "performance_summary.json", {"status": "complete", "n": 2})
    loaded = load_json(summary)
    assert loaded["n"] == 2
