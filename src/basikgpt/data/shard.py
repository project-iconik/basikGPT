"""Binary token shard serialization and memory-mapped dataset reader.

Stores token streams as uint16 NumPy arrays (.npy) with atomic writes, SHA-256 checksums,
and vocabulary range validation. Provides a PyTorch Dataset for causal LM sequence sampling.
"""

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_GPT2_VOCAB_INDEX = 50256


class TokenShardWriter:
    """Serializes integer token streams into fixed-size uint16 .npy binary shard files.

    Ensures atomic disk writes (write to temp file then rename) and computes SHA-256
    cryptographic checksums for data integrity verification.
    """

    def __init__(
        self,
        output_dir: Path,
        split_name: str,
        tokens_per_shard: int = 1_000_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.split_name = split_name
        self.tokens_per_shard = tokens_per_shard

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[int] = []
        self._shard_index = 0
        self._completed_shards: list[dict[str, Any]] = []

    @property
    def completed_shards(self) -> list[dict[str, Any]]:
        """List of metadata dictionaries for all finalized shards."""
        return list(self._completed_shards)

    def add_tokens(self, tokens: Sequence[int]) -> list[dict[str, Any]]:
        """Appends tokens to the internal buffer and flushes full shards to disk."""
        if not tokens:
            return []

        self._buffer.extend(tokens)
        newly_created: list[dict[str, Any]] = []

        while len(self._buffer) >= self.tokens_per_shard:
            shard_tokens = self._buffer[: self.tokens_per_shard]
            self._buffer = self._buffer[self.tokens_per_shard :]
            shard_meta = self._write_shard(shard_tokens)
            newly_created.append(shard_meta)

        return newly_created

    def finalize(self, keep_tail: bool = True) -> list[dict[str, Any]]:
        """Flushes any remaining tokens in the buffer as a final tail shard."""
        newly_created: list[dict[str, Any]] = []
        if keep_tail and len(self._buffer) > 0:
            shard_tokens = self._buffer
            self._buffer = []
            shard_meta = self._write_shard(shard_tokens)
            newly_created.append(shard_meta)
        return newly_created

    def _write_shard(self, token_list: list[int]) -> dict[str, Any]:
        """Validates token bounds and atomically writes uint16 NumPy array to disk."""
        arr = np.asarray(token_list, dtype=np.int64)

        # Range validation to prevent silent uint16 overflow/underflow
        if arr.size > 0:
            min_val, max_val = int(arr.min()), int(arr.max())
            if min_val < 0 or max_val > MAX_GPT2_VOCAB_INDEX:
                raise ValueError(
                    f"Token values out of GPT-2 vocabulary range [0, {MAX_GPT2_VOCAB_INDEX}]: "
                    f"min={min_val}, max={max_val}"
                )

        uint16_arr = arr.astype(np.uint16)
        filename = f"{self.split_name}-{self._shard_index:06d}.npy"
        target_path = self.output_dir / filename
        temp_path = self.output_dir / f"{filename}.tmp"

        # 1. Write to temporary file using explicit binary stream
        with open(temp_path, "wb") as f:
            np.save(f, uint16_arr)

        # 2. Compute SHA-256 checksum and file size
        hasher = hashlib.sha256()
        with open(temp_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()
        byte_size = os.path.getsize(temp_path)

        # 3. Atomic rename/replace (cross-platform)
        if target_path.exists():
            target_path.unlink()
        temp_path.replace(target_path)

        shard_info = {
            "filename": filename,
            "split": self.split_name,
            "token_count": len(token_list),
            "byte_size": byte_size,
            "checksum": checksum,
        }

        self._completed_shards.append(shard_info)
        self._shard_index += 1
        return shard_info


class ShardedTokenDataset(Dataset):
    """PyTorch Dataset reading uint16 token shards for causal language modeling.

    Yields contiguous input_ids and targets shifted by 1 token:
        input_ids = tokens[i : i + context_length]
        targets   = tokens[i + 1 : i + context_length + 1]
    """

    def __init__(
        self,
        shard_paths: list[Path] | Path,
        context_length: int = 1024,
        stride: int | None = None,
    ) -> None:
        if isinstance(shard_paths, (str, Path)):
            p = Path(shard_paths)
            if p.is_dir():
                self.shard_paths = sorted(p.glob("*.npy"))
            else:
                self.shard_paths = [p]
        else:
            self.shard_paths = [Path(p) for p in shard_paths]

        if not self.shard_paths:
            raise ValueError("No valid .npy shard files found.")

        self.context_length = context_length
        self.stride = context_length if stride is None else stride

        # Index shards and cumulative sample counts
        self._shard_arrays: list[np.ndarray] = []
        self._shard_sample_counts: list[int] = []
        self._cumulative_samples: list[int] = []

        total_samples = 0
        self._total_tokens = 0
        self._discarded_tail_tokens = 0

        for path in self.shard_paths:
            mmap_arr = np.load(path, mmap_mode="r")
            n_tokens = len(mmap_arr)
            self._total_tokens += n_tokens
            self._shard_arrays.append(mmap_arr)

            # A sample of length context_length requires (context_length + 1) tokens
            if n_tokens >= context_length + 1:
                n_samples = (n_tokens - context_length - 1) // self.stride + 1
                used_tokens = (n_samples - 1) * self.stride + context_length + 1
                discarded = n_tokens - used_tokens
            else:
                n_samples = 0
                discarded = n_tokens

            self._discarded_tail_tokens += discarded
            self._shard_sample_counts.append(n_samples)
            total_samples += n_samples
            self._cumulative_samples.append(total_samples)

        self._total_samples = total_samples

    def __len__(self) -> int:
        return self._total_samples

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all indexed shards."""
        return self._total_tokens

    @property
    def discarded_tail_tokens(self) -> int:
        """Number of tail tokens discarded because they could not form a full context."""
        return self._discarded_tail_tokens

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Sample index {idx} out of range [0, {self._total_samples})")

        # Binary search for the shard containing this sample index
        import bisect

        shard_idx = bisect.bisect_right(self._cumulative_samples, idx)
        prev_cumulative = self._cumulative_samples[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = idx - prev_cumulative

        start = local_idx * self.stride
        end = start + self.context_length + 1

        tokens = np.array(self._shard_arrays[shard_idx][start:end], dtype=np.int64)

        input_ids = torch.from_numpy(tokens[: self.context_length]).to(torch.long)
        targets = torch.from_numpy(tokens[1 : self.context_length + 1]).to(torch.long)

        return input_ids, targets
