"""Unit tests for offline math/FineWeb shard interleave."""

from pathlib import Path
import numpy as np

from basikgpt.data.combine import combine_shard_directories, interleave_cycle
from basikgpt.data.manifest import create_manifest, save_manifest


def test_interleave_cycle_math_first_fineweb_tail() -> None:
    math = [Path(f"m{i}") for i in range(2)]
    fw = [Path(f"f{i}") for i in range(18)]
    ordered = interleave_cycle(math, fw, leading_per_cycle=1, trailing_per_cycle=9)
    assert [p.name for p in ordered[:10]] == ["m0", *[f"f{i}" for i in range(9)]]
    assert [p.name for p in ordered[10:20]] == ["m1", *[f"f{i}" for i in range(9, 18)]]
    assert ordered[-1].name.startswith("f")


def test_combine_shard_directories_links_and_val(tmp_path: Path) -> None:
    math_dir = tmp_path / "math"
    fw_dir = tmp_path / "fw"
    val_dir = tmp_path / "valsrc"
    out_dir = tmp_path / "mix"
    math_dir.mkdir()
    fw_dir.mkdir()
    val_dir.mkdir()

    math_shards = []
    fw_shards = []
    for i in range(2):
        arr = np.arange(i * 10, i * 10 + 8, dtype=np.uint16)
        path = math_dir / f"train-{i:06d}.npy"
        np.save(path, arr)
        math_shards.append(
            {
                "filename": path.name,
                "split": "train",
                "token_count": 8,
                "byte_size": int(path.stat().st_size),
                "checksum": "abc",
            }
        )
    for i in range(18):
        arr = np.arange(100 + i, 108 + i, dtype=np.uint16)
        path = fw_dir / f"train-{i:06d}.npy"
        np.save(path, arr)
        fw_shards.append(
            {
                "filename": path.name,
                "split": "train",
                "token_count": 8,
                "byte_size": int(path.stat().st_size),
                "checksum": "def",
            }
        )
    val_path = val_dir / "validation-000000.npy"
    np.save(val_path, np.arange(50, 58, dtype=np.uint16))
    save_manifest(
        create_manifest(
            "open-web-math/open-web-math",
            "",
            "mathsha",
            0.0,
            8,
            {"train_tokens": 16, "validation_tokens": 0},
            math_shards,
            selection="math",
        ),
        math_dir / "manifest.json",
    )
    save_manifest(
        create_manifest(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            "fwsha",
            0.0,
            8,
            {"train_tokens": 144, "validation_tokens": 0},
            fw_shards,
            selection="fineweb",
        ),
        fw_dir / "manifest.json",
    )
    save_manifest(
        create_manifest(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            "edusha",
            0.005,
            8,
            {"train_tokens": 0, "validation_tokens": 8},
            [
                {
                    "filename": "validation-000000.npy",
                    "split": "validation",
                    "token_count": 8,
                    "byte_size": int(val_path.stat().st_size),
                    "checksum": "val",
                }
            ],
        ),
        val_dir / "manifest.json",
    )

    manifest = combine_shard_directories(out_dir, fw_dir, math_dir, val_dir=val_dir)
    names = [p.name for p in sorted(out_dir.glob("train-*.npy"))]
    assert names[0] == "train-000000.npy"
    first = np.load(out_dir / "train-000000.npy")
    assert list(first) == list(np.arange(0, 8))
    tenth = np.load(out_dir / "train-000009.npy")
    assert list(tenth) == list(np.arange(108, 116))
    assert (out_dir / "validation-000000.npy").is_symlink()
    assert manifest["mix"]["last_train_source"] == "fineweb"
    assert manifest["mix"]["math_train_tokens"] == 16
    assert manifest["mix"]["fineweb_train_tokens"] == 144
