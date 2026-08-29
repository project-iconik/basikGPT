"""Combine tokenized shard directories into one sequentially mixed training set."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from basikgpt.data.manifest import create_manifest, load_manifest, save_manifest


def list_train_shards(directory: Path) -> list[Path]:
    """Returns sorted train-*.npy paths in `directory`."""
    return sorted(Path(directory).glob("train-*.npy"))


def list_validation_shards(directory: Path) -> list[Path]:
    """Returns sorted validation-*.npy paths in `directory`."""
    return sorted(Path(directory).glob("validation-*.npy"))


def interleave_cycle(
    leading: list[Path],
    trailing: list[Path],
    leading_per_cycle: int,
    trailing_per_cycle: int,
) -> list[Path]:
    """Interleaves paths as [leading × L, trailing × T] cycles, leftover trailing last.

    Putting leftover `trailing` at the end keeps a FineWeb-heavy tail when leading is math
    and trailing is FineWeb (9:1 mix, English-final checkpoint).
    """
    if leading_per_cycle <= 0 or trailing_per_cycle <= 0:
        raise ValueError("Cycle counts must be positive")

    out: list[Path] = []
    li = 0
    ti = 0
    while li < len(leading) and ti < len(trailing):
        taken_lead = 0
        while taken_lead < leading_per_cycle and li < len(leading):
            out.append(leading[li])
            li += 1
            taken_lead += 1
        taken_trail = 0
        while taken_trail < trailing_per_cycle and ti < len(trailing):
            out.append(trailing[ti])
            ti += 1
            taken_trail += 1
    out.extend(leading[li:])
    out.extend(trailing[ti:])
    return out


def _link_or_copy(src: Path, dest: Path) -> None:
    """Hard-links `src` to `dest`, copying if the filesystem rejects the link."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        dest.write_bytes(src.read_bytes())


def _symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def _shard_lookup(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {str(item.get("filename")): item for item in manifest.get("shards", [])}


def _token_count(path: Path, meta: dict[str, Any] | None) -> int:
    if meta and "token_count" in meta:
        return int(meta["token_count"])
    import numpy as np

    return int(len(np.load(path, mmap_mode="r")))


def combine_shard_directories(
    output_dir: Path,
    fineweb_dir: Path,
    math_dir: Path,
    val_dir: Path | None = None,
    math_per_cycle: int = 1,
    fineweb_per_cycle: int = 9,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Hard-links math+FineWeb train shards in M1/F9 cycles and optionally symlinks val shards."""
    output_dir = Path(output_dir)
    fineweb_dir = Path(fineweb_dir)
    math_dir = Path(math_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists and is not empty. "
            "Pass overwrite=True to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    math_shards = list_train_shards(math_dir)
    fw_shards = list_train_shards(fineweb_dir)
    if not math_shards:
        raise FileNotFoundError(f"No train-*.npy shards in {math_dir}")
    if not fw_shards:
        raise FileNotFoundError(f"No train-*.npy shards in {fineweb_dir}")

    ordered = interleave_cycle(
        leading=math_shards,
        trailing=fw_shards,
        leading_per_cycle=math_per_cycle,
        trailing_per_cycle=fineweb_per_cycle,
    )

    math_manifest = load_manifest(math_dir / "manifest.json") if (math_dir / "manifest.json").exists() else None
    fw_manifest = load_manifest(fineweb_dir / "manifest.json") if (fineweb_dir / "manifest.json").exists() else None
    math_lookup = _shard_lookup(math_manifest)
    fw_lookup = _shard_lookup(fw_manifest)

    math_resolved = {path.resolve() for path in math_shards}
    fw_resolved = {path.resolve() for path in fw_shards}

    train_entries: list[dict[str, Any]] = []
    math_tokens = 0
    fw_tokens = 0
    for index, src in enumerate(ordered):
        dest_name = f"train-{index:06d}.npy"
        dest = output_dir / dest_name
        _link_or_copy(src, dest)
        resolved = src.resolve()
        if resolved in math_resolved:
            source = "open-web-math"
            src_meta = math_lookup.get(src.name)
        elif resolved in fw_resolved:
            source = "fineweb"
            src_meta = fw_lookup.get(src.name)
        else:
            source = "unknown"
            src_meta = None
        token_count = _token_count(src, src_meta)
        if source == "open-web-math":
            math_tokens += token_count
        else:
            fw_tokens += token_count
        entry = {
            "filename": dest_name,
            "split": "train",
            "token_count": token_count,
            "byte_size": int(src.stat().st_size),
            "checksum": (src_meta or {}).get("checksum"),
            "source": source,
            "source_filename": src.name,
        }
        train_entries.append(entry)

    val_entries: list[dict[str, Any]] = []
    val_tokens = 0
    if val_dir is not None:
        val_dir = Path(val_dir)
        val_manifest = load_manifest(val_dir / "manifest.json") if (val_dir / "manifest.json").exists() else None
        val_lookup = _shard_lookup(val_manifest)
        for src in list_validation_shards(val_dir):
            dest = output_dir / src.name
            _symlink(src, dest)
            meta = val_lookup.get(src.name)
            token_count = _token_count(src, meta)
            val_tokens += token_count
            val_entries.append(
                {
                    "filename": src.name,
                    "split": "validation",
                    "token_count": token_count,
                    "byte_size": int(src.stat().st_size),
                    "checksum": (meta or {}).get("checksum"),
                    "source": "fineweb-edu-val",
                    "source_filename": src.name,
                }
            )

    last_source = train_entries[-1]["source"] if train_entries else None
    if last_source == "open-web-math":
        print("[combine] Warning: mix ends on a math shard; FineWeb tail was shorter than expected.")

    stats = {
        "total_documents_seen": 0,
        "train_documents": 0,
        "validation_documents": 0,
        "skipped_documents": 0,
        "skipped_for_budget": 0,
        "train_tokens": math_tokens + fw_tokens,
        "validation_tokens": val_tokens,
        "fineweb_train_tokens": fw_tokens,
        "math_train_tokens": math_tokens,
    }
    shards = train_entries + val_entries
    fw_rev = ((fw_manifest or {}).get("dataset_provenance") or {}).get("revision") or "unknown"
    math_rev = ((math_manifest or {}).get("dataset_provenance") or {}).get("revision") or "unknown"
    manifest = create_manifest(
        dataset_repository="mix:fineweb+open-web-math",
        dataset_config=f"math{math_per_cycle}_fineweb{fineweb_per_cycle}",
        dataset_revision=f"fineweb={fw_rev};math={math_rev}",
        validation_fraction=0.0,
        shard_token_target=1_000_000,
        stats=stats,
        shards=shards,
        dataset_license="ODC-By 1.0",
        selection=(
            f"Offline shard interleave: {math_per_cycle} OpenWebMath + "
            f"{fineweb_per_cycle} FineWeb per cycle, FineWeb tail"
        ),
    )
    manifest["mix"] = {
        "math_per_cycle": math_per_cycle,
        "fineweb_per_cycle": fineweb_per_cycle,
        "math_dir": str(math_dir),
        "fineweb_dir": str(fineweb_dir),
        "val_dir": str(val_dir) if val_dir is not None else None,
        "train_shards": len(train_entries),
        "validation_shards": len(val_entries),
        "last_train_source": last_source,
        "fineweb_train_tokens": fw_tokens,
        "math_train_tokens": math_tokens,
    }
    save_manifest(manifest, output_dir / "manifest.json")
    print(f"[combine] Wrote {len(train_entries)} train shards to {output_dir}")
    print(f"  FineWeb tokens: {fw_tokens:,}")
    print(f"  Math tokens:    {math_tokens:,}")
    print(f"  Val tokens:     {val_tokens:,}")
    print(f"  Last train source: {last_source}")
    return manifest
