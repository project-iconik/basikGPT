"""Single-device baseline pretraining loop and orchestration for basikGPT."""

from collections.abc import Iterator
import contextlib
import json
import math
from pathlib import Path
import time
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basikgpt.training.accounting import estimate_training_flops
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.compile import compile_model
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import (
    gradient_was_clipped,
    perplexity_from_loss,
    save_run_metadata,
    save_run_summary,
)
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.reproducibility import seed_everything
from basikgpt.training.scheduler import get_learning_rate_at_step, update_learning_rate
from basikgpt.training.sdpa import sdpa_kernel_context


def resolve_device(device_str: str) -> torch.device:
    """Resolves device string ('auto', 'cpu', 'cuda', 'cuda:X') into a torch.device.

    Raises:
        RuntimeError: If a CUDA device is explicitly requested but CUDA is unavailable.
        ValueError: If device_str is unrecognized.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device_str}' but CUDA is not available on this system."
            )
        return torch.device(device_str)
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device specification: '{device_str}'")


class Trainer:
    """Orchestrates single-device autoregressive pretraining with failure safety and provenance tracking."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        run_name: str | None = None,
        dataset_manifest: dict[str, Any] | None = None,
        dataset_manifest_path: Path | str | None = None,
        resume_from: Path | str | None = None,
        overwrite: bool = False,
        init_weights: Path | str | None = None,
    ) -> None:
        self.config = config
        self.run_name = run_name or Path(config.output_dir).name
        self.output_dir = Path(config.output_dir)
        self.resume_from = resume_from

        # 1. Deterministic RNG initialization
        seed_everything(config.seed)

        # 2. Directory collision check
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not overwrite and self.resume_from is None:
            metrics_exist = (self.output_dir / "metrics.jsonl").exists()
            checkpoints_exist = any(self.output_dir.glob("step-*.pt"))
            if metrics_exist or checkpoints_exist:
                raise FileExistsError(
                    f"Output directory '{self.output_dir}' already exists and contains training artifacts. "
                    "Use a distinct --run-name, specify resume_from=<checkpoint>, or specify overwrite=True."
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.jsonl"
        if overwrite and self.resume_from is None and self.metrics_file.exists():
            self.metrics_file.unlink()

        # 3. Device & Precision resolution
        self.device = resolve_device(config.device)

        if self.device.type == "cpu" and self.config.precision != "fp32":
            raise ValueError(
                f"CPU mixed precision is not supported in basikGPT. "
                f"Requested precision '{self.config.precision}' on device '{self.device}'. "
                "Use precision='fp32' on CPU or device='cuda'."
            )

        if self.device.type == "cuda" and self.config.precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                gpu_name = torch.cuda.get_device_name(self.device)
                raise RuntimeError(
                    f"Requested precision 'bf16' on GPU '{gpu_name}', but this GPU does not support bfloat16 natively."
                )

        if self.device.type == "cuda" and self.config.precision == "bf16":
            self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            self.scaler: torch.amp.GradScaler | None = None
        elif self.device.type == "cuda" and self.config.precision == "fp16":
            self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.autocast_ctx = contextlib.nullcontext()
            self.scaler = None

        # 4. Raw model ownership. torch.compile wraps a separate callable; checkpoints stay raw.
        self.raw_model = model.to(self.device)
        if init_weights is not None:
            self._load_init_weights(init_weights)
        self.optimizer = configure_optimizers(self.raw_model, self.config)

        if self.config.compile:
            if self.device.type != "cuda":
                raise ValueError(
                    "torch.compile is only supported on CUDA in this project. "
                    f"Requested compile=True on device '{self.device}'."
                )
            # Fail-fast: no silent uncompiled fallback if inductor compilation raises here.
            self.model = compile_model(self.raw_model, mode=self.config.compile_mode)
        else:
            self.model = self.raw_model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.global_step = 0
        self.tokens_seen = 0
        self.data_sample_index = 0
        self.best_val_loss: float | None = None
        self.last_val_loss: float | None = None
        self.time_to_first_optimizer_step: float | None = None
        self.train_elapsed_seconds = 0.0
        self.step_durations: list[float] = []
        self.cold_compile_seconds: float | None = None
        self.compile_recompile_info: dict[str, Any] | None = None
        self.parameter_count = self._count_parameters()
        self.tokens_per_optimizer_step = self._tokens_per_optimizer_step()

        # 5. Save run provenance. On resume this refreshes training_config.json
        # to match the continuing process (e.g. stop_at_step cleared).
        if hasattr(self.raw_model, "config"):
            save_run_metadata(
                output_dir=self.output_dir,
                run_name=self.run_name,
                model_config=self.raw_model.config,
                training_config=self.config,
                dataset_manifest=dataset_manifest,
                dataset_manifest_path=dataset_manifest_path,
                extra_metadata={
                    "parameter_count": self.parameter_count,
                    "tokens_per_optimizer_step": self.tokens_per_optimizer_step,
                },
            )

    def _infinite_loader(self, loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields batches infinitely from a DataLoader by looping across epochs."""
        while True:
            for batch in loader:
                yield batch

    def _sequential_batch_iterator(
        self,
        dataset: Any,
        batch_size: int,
        start_index: int = 0,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields sequential full batches, wrapping the dataset, from `start_index`.

        Used when `track_data_sample_index` is enabled so resume can fast-forward
        by integer sample index instead of replaying a shuffled DataLoader.
        """
        n = len(dataset)
        if n <= 0:
            raise ValueError("Cannot iterate an empty dataset")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        index = start_index
        while True:
            xs: list[torch.Tensor] = []
            ys: list[torch.Tensor] = []
            for _ in range(batch_size):
                item = dataset[index % n]
                x, y = item[0], item[1]
                xs.append(x)
                ys.append(y)
                index += 1
            yield torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _make_train_iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.config.track_data_sample_index:
            dataset = self.train_loader.dataset
            return self._sequential_batch_iterator(
                dataset,
                batch_size=self.config.batch_size,
                start_index=self.data_sample_index,
            )
        return self._infinite_loader(self.train_loader)

    def _load_init_weights(self, path: Path | str) -> None:
        """Loads a raw state_dict (or checkpoint payload) into `raw_model` before compile."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload
        self.raw_model.load_state_dict(state_dict)
        if hasattr(self.raw_model, "lm_head") and hasattr(self.raw_model, "wte"):
            if self.raw_model.lm_head.weight is not self.raw_model.wte.weight:
                self.raw_model.lm_head.weight = self.raw_model.wte.weight

    def _checkpoint_extra_state(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.config.track_data_sample_index:
            extra["data_sample_index"] = self.data_sample_index
            extra["resume_class"] = "exact-sample-index"
        else:
            extra["resume_class"] = "state-continuous"
        return extra

    def _save_training_checkpoint(self, checkpoint_path: Path) -> Path:
        return save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.raw_model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            training_config=self.config,
            model_config=getattr(self.raw_model, "config", None),
            scaler=self.scaler,
            extra_state=self._checkpoint_extra_state(),
        )

    def _snapshot_compile_counters(self) -> dict[str, Any] | None:
        try:
            from torch._dynamo.utils import counters

            frames = dict(counters["frames"]) if counters["frames"] else {}
            return {"frames": {str(key): int(value) for key, value in frames.items()}}
        except Exception:
            return None

    def _estimate_cold_compile_seconds(self) -> float | None:
        if not self.config.compile or self.time_to_first_optimizer_step is None:
            return None
        later = self.step_durations[1:]
        if not later:
            return self.time_to_first_optimizer_step
        later_sorted = sorted(later)
        median_later = later_sorted[len(later_sorted) // 2]
        return max(0.0, self.time_to_first_optimizer_step - median_later)

    def _sdpa_context(self):
        """Forces a single SDPA backend when configured; `auto` leaves PyTorch dispatch unchanged."""
        return sdpa_kernel_context(self.config.sdpa_kernel)

    def _count_parameters(self) -> int:
        if hasattr(self.raw_model, "num_parameters"):
            return int(self.raw_model.num_parameters())
        return sum(p.numel() for p in self.raw_model.parameters())

    def _tokens_per_optimizer_step(self) -> int | None:
        context_length = getattr(getattr(self.raw_model, "config", None), "context_length", None)
        if context_length is None:
            return None
        return (
            self.config.batch_size
            * int(context_length)
            * self.config.gradient_accumulation_steps
        )

    def _val_metrics_record(self, val_loss: float, elapsed_seconds: float) -> dict[str, Any]:
        return {
            "type": "val",
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "val_loss": val_loss,
            "val_perplexity": perplexity_from_loss(val_loss),
            "elapsed_seconds": elapsed_seconds,
        }

    def evaluate(self, num_batches: int | None = None) -> float:
        """Runs an evaluation loop over validation batches without mutating training state.

        Args:
            num_batches: Number of validation batches to evaluate (defaults to config.eval_batches).

        Returns:
            Average cross-entropy loss across evaluated batches.
        """
        if self.val_loader is None:
            return 0.0

        num_batches = num_batches or self.config.eval_batches
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        batches_evaluated = 0

        with torch.inference_mode(), self._sdpa_context():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= num_batches:
                    break
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with self.autocast_ctx:
                    logits = self.model(x)
                    loss = compute_cross_entropy_loss(logits, y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss detected during evaluation at batch {i}: {loss.item()}"
                    )

                total_loss += loss.item()
                batches_evaluated += 1

        if was_training:
            self.model.train()

        mean_val_loss = total_loss / max(1, batches_evaluated)
        if self.best_val_loss is None or mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss

        return mean_val_loss

    def train_step(self, data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        """Executes a single optimizer step comprising `gradient_accumulation_steps` micro-batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_tokens = 0
        accum_steps = self.config.gradient_accumulation_steps

        with self._sdpa_context():
            for _ in range(accum_steps):
                x, y = next(data_iter)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with self.autocast_ctx:
                    logits = self.model(x)
                    loss = compute_cross_entropy_loss(logits, y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss detected at global step {self.global_step}: {loss.item()}"
                    )

                # Scale loss for gradient accumulation
                loss_scaled = loss / accum_steps
                if self.scaler is not None:
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                step_loss += loss.item() / accum_steps
                step_tokens += y.numel()
                if self.config.track_data_sample_index:
                    self.data_sample_index += y.shape[0]

        # Unscale FP16 grads before measuring/clipping so the reported norm is in parameter space.
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # clip_grad_norm_ returns the pre-clipping Euclidean norm. max_norm=inf measures without clipping.
        clip_max = self.config.max_grad_norm if self.config.max_grad_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), clip_max).item()

        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at global step {self.global_step}: {grad_norm}"
            )

        lr = get_learning_rate_at_step(self.global_step, self.config)
        update_learning_rate(self.optimizer, lr)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.tokens_seen += step_tokens
        self.global_step += 1

        return {
            "loss": step_loss,
            "train_loss": step_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_tokens": step_tokens,
        }

    def train(self, resume_from: Path | str | None = None) -> list[dict[str, Any]]:
        """Runs the full training loop until `max_steps` with interruption safety and summary generation."""
        target_resume = resume_from or self.resume_from
        if target_resume is not None:
            meta = load_checkpoint(
                target_resume,
                self.raw_model,
                self.optimizer,
                scaler=self.scaler,
                device=self.device,
                expected_model_config=getattr(self.raw_model, "config", None),
            )
            self.global_step = meta.get("global_step", 0)
            self.tokens_seen = meta.get("tokens_seen", 0)
            extra = meta.get("extra_state") or {}
            if self.config.track_data_sample_index:
                saved_index = extra.get("data_sample_index")
                if saved_index is None:
                    saved_index = self.global_step * self.config.batch_size * self.config.gradient_accumulation_steps
                self.data_sample_index = int(saved_index)
            self._prune_metrics_after_resume(self.global_step)
            print(
                f"[Trainer] Resumed from {target_resume} at step {self.global_step}, "
                f"tokens {self.tokens_seen:,}"
                + (
                    f", data_sample_index {self.data_sample_index:,}"
                    if self.config.track_data_sample_index
                    else ""
                )
            )

        data_iter = self._make_train_iterator()
        history: list[dict[str, Any]] = []

        gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None

        print(f"[Trainer] Starting training for run '{self.run_name}' ...")
        print(f"  Device:             {self.device} (Precision: {self.config.precision.upper()})")
        if gpu_name:
            print(f"  GPU Model:          {gpu_name}")
        print(f"  Random Seed:        {self.config.seed}")
        if self.config.compile:
            print(f"  torch.compile:      enabled (backend=inductor, mode={self.config.compile_mode})")
        else:
            print("  torch.compile:      disabled")
        print(f"  SDPA kernel:        {self.config.sdpa_kernel}")
        print(f"  Target steps:       {self.config.max_steps:,}")
        print(f"  Batch size:         {self.config.batch_size}")
        print(f"  Grad accumulation:  {self.config.gradient_accumulation_steps}")
        print(f"  Warmup steps:       {self.config.warmup_steps:,}")
        print(f"  Peak LR:            {self.config.learning_rate:.2e}")
        print(f"  Min LR:             {self.config.min_learning_rate:.2e}")
        print(f"  Output dir:         {self.output_dir}")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()
        last_log_time = start_time
        last_tokens_seen = self.tokens_seen
        last_train_loss: float | None = None
        last_val_loss: float | None = None
        last_ckpt_path: Path | None = None
        compile_counters_start = self._snapshot_compile_counters() if self.config.compile else None
        stop_at = self.config.stop_at_step
        loop_limit = stop_at if stop_at is not None else self.config.max_steps

        try:
            if (
                self.config.eval_at_start
                and self.val_loader is not None
                and target_resume is None
            ):
                start_val = self.evaluate()
                last_val_loss = start_val
                self.last_val_loss = start_val
                self._append_metrics_record(self._val_metrics_record(start_val, elapsed_seconds=0.0))
                print(f"Step {self.global_step:06d}/{self.config.max_steps:06d} | Val: {start_val:.4f}")

            while self.global_step < loop_limit:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_t0 = time.perf_counter()
                step_metrics = self.train_step(data_iter)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_dt = time.perf_counter() - step_t0
                self.train_elapsed_seconds += step_dt
                self.step_durations.append(step_dt)
                if self.time_to_first_optimizer_step is None:
                    self.time_to_first_optimizer_step = step_dt
                last_train_loss = step_metrics["loss"]

                reached_end = self.global_step == self.config.max_steps
                reached_stop = stop_at is not None and self.global_step == stop_at
                should_eval = self.val_loader is not None and (
                    self.global_step % self.config.eval_interval == 0 or reached_end or reached_stop
                )
                should_log = (
                    self.global_step % self.config.log_interval == 0 or reached_end or reached_stop
                )
                if self.config.checkpoint_steps is not None:
                    should_ckpt = self.global_step in self.config.checkpoint_steps or reached_stop
                else:
                    should_ckpt = (
                        self.global_step % self.config.checkpoint_interval == 0
                        or reached_end
                        or reached_stop
                    )

                val_loss = None
                if should_eval:
                    val_loss = self.evaluate()
                    last_val_loss = val_loss
                    self.last_val_loss = val_loss

                if should_ckpt:
                    ckpt_path = self.output_dir / f"step-{self.global_step:08d}.pt"
                    self._save_training_checkpoint(ckpt_path)
                    last_ckpt_path = ckpt_path

                elapsed_total = time.perf_counter() - start_time
                training_only_tok_s = self.tokens_seen / max(1e-6, self.train_elapsed_seconds)
                e2e_tok_s = self.tokens_seen / max(1e-6, elapsed_total)

                if should_log:
                    now = time.perf_counter()
                    elapsed_since_log = now - last_log_time
                    tokens_delta = self.tokens_seen - last_tokens_seen
                    tok_per_sec = tokens_delta / max(1e-6, elapsed_since_log)
                    elapsed_total = now - start_time
                    e2e_tok_s = self.tokens_seen / max(1e-6, elapsed_total)

                    peak_allocated_vram_bytes = None
                    peak_reserved_vram_bytes = None
                    if self.device.type == "cuda":
                        peak_allocated_vram_bytes = int(torch.cuda.max_memory_allocated(self.device))
                        peak_reserved_vram_bytes = int(torch.cuda.max_memory_reserved(self.device))

                    train_record = {
                        "type": "train",
                        "step": self.global_step,
                        "tokens_seen": self.tokens_seen,
                        "loss": step_metrics["loss"],
                        "train_loss": step_metrics["loss"],
                        "learning_rate": step_metrics["lr"],
                        "grad_norm": step_metrics["grad_norm"],
                        "grad_clipped": gradient_was_clipped(
                            step_metrics["grad_norm"], self.config.max_grad_norm
                        ),
                        "estimated_flops": estimate_training_flops(
                            self.parameter_count, int(tokens_delta)
                        ),
                        "tokens_per_sec": tok_per_sec,
                        "training_only_tokens_per_sec": training_only_tok_s,
                        "end_to_end_tokens_per_sec": e2e_tok_s,
                        "elapsed_seconds": elapsed_total,
                        "train_elapsed_seconds": self.train_elapsed_seconds,
                        "compile": self.config.compile,
                        "compile_mode": self.config.compile_mode if self.config.compile else None,
                    }
                    if peak_allocated_vram_bytes is not None:
                        train_record["peak_allocated_vram_bytes"] = peak_allocated_vram_bytes
                        train_record["peak_reserved_vram_bytes"] = peak_reserved_vram_bytes
                        train_record["peak_allocated_vram_mib"] = peak_allocated_vram_bytes / (1024 * 1024)

                    history.append(train_record)
                    self._append_metrics_record(train_record)

                    val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
                    vram_str = ""
                    if peak_allocated_vram_bytes is not None:
                        allocated_mib = peak_allocated_vram_bytes / (1024 * 1024)
                        vram_str = f" | Allocated: {allocated_mib:.0f} MiB"
                    print(
                        f"Step {self.global_step:06d}/{self.config.max_steps:06d} | "
                        f"Loss: {step_metrics['loss']:.4f} | "
                        f"GradNorm: {step_metrics['grad_norm']:.3f} | "
                        f"LR: {step_metrics['lr']:.2e} | "
                        f"Tokens: {self.tokens_seen:,} | "
                        f"{tok_per_sec:,.0f} tok/s{vram_str}{val_str}"
                    )

                    last_log_time = now
                    last_tokens_seen = self.tokens_seen
                elif should_eval and val_loss is not None:
                    print(
                        f"Step {self.global_step:06d}/{self.config.max_steps:06d} | "
                        f"Val: {val_loss:.4f}"
                    )

                if val_loss is not None:
                    self._append_metrics_record(
                        self._val_metrics_record(val_loss, elapsed_seconds=elapsed_total)
                    )

            self.cold_compile_seconds = self._estimate_cold_compile_seconds()
            compile_counters_end = self._snapshot_compile_counters() if self.config.compile else None
            self.compile_recompile_info = {
                "counters_start": compile_counters_start,
                "counters_end": compile_counters_end,
                "time_to_first_optimizer_step": self.time_to_first_optimizer_step,
                "cold_compile_seconds": self.cold_compile_seconds,
                "n_optimizer_steps_timed": len(self.step_durations),
            }
            if self.config.compile and len(self.step_durations) > 2:
                later = self.step_durations[1:]
                median_later = sorted(later)[len(later) // 2]
                first = self.step_durations[0]
                spikes = [
                    i + 2
                    for i, dt in enumerate(later)
                    if median_later > 0 and dt > max(3.0 * median_later, 0.5 * first)
                ]
                self.compile_recompile_info["later_step_time_spikes"] = spikes
                self.compile_recompile_info["possible_repeated_recompile"] = bool(spikes)

            elapsed_total = time.perf_counter() - start_time
            paused = stop_at is not None and self.global_step < self.config.max_steps
            status = "paused" if paused else "completed"

            final_ckpt = last_ckpt_path
            if self.config.save_step_final and not paused:
                final_ckpt = self.output_dir / "step-final.pt"
                self._save_training_checkpoint(final_ckpt)
            elif final_ckpt is None:
                final_ckpt = self.output_dir / f"step-{self.global_step:08d}.pt"
                self._save_training_checkpoint(final_ckpt)

            training_only_tok_s = self.tokens_seen / max(1e-6, self.train_elapsed_seconds)
            e2e_tok_s = self.tokens_seen / max(1e-6, elapsed_total)
            extra_summary = {
                "training_only_tokens_per_sec": training_only_tok_s,
                "end_to_end_tokens_per_sec": e2e_tok_s,
                "train_elapsed_seconds": self.train_elapsed_seconds,
                "time_to_first_optimizer_step": self.time_to_first_optimizer_step,
                "cold_compile_seconds": self.cold_compile_seconds,
                "compile": self.config.compile,
                "compile_mode": self.config.compile_mode if self.config.compile else None,
                "compile_recompile_info": self.compile_recompile_info,
                "data_sample_index": self.data_sample_index if self.config.track_data_sample_index else None,
                "resume_class": (
                    "exact-sample-index" if self.config.track_data_sample_index else "state-continuous"
                ),
            }
            save_run_summary(
                output_dir=self.output_dir,
                status=status,
                final_step=self.global_step,
                tokens_seen=self.tokens_seen,
                elapsed_seconds=elapsed_total,
                final_train_loss=last_train_loss,
                final_val_loss=last_val_loss,
                best_val_loss=self.best_val_loss,
                checkpoint_path=final_ckpt,
                extra=extra_summary,
            )
            if paused:
                print(
                    f"[Trainer] Paused at step {self.global_step} "
                    f"(stop_at_step={stop_at}). Checkpoint saved to {final_ckpt}"
                )
            else:
                print(f"[Trainer] Training completed successfully. Final checkpoint saved to {final_ckpt}")
            return history

        except KeyboardInterrupt:
            elapsed_total = time.perf_counter() - start_time
            print(f"\n[Trainer] KeyboardInterrupt received. Saving emergency checkpoint at step {self.global_step} ...")
            interrupted_ckpt = self.output_dir / "step-interrupted.pt"
            self._save_training_checkpoint(interrupted_ckpt)
            save_run_summary(
                output_dir=self.output_dir,
                status="interrupted",
                final_step=self.global_step,
                tokens_seen=self.tokens_seen,
                elapsed_seconds=elapsed_total,
                final_train_loss=last_train_loss,
                final_val_loss=last_val_loss,
                best_val_loss=self.best_val_loss,
                checkpoint_path=interrupted_ckpt,
                error_message="Execution interrupted by user (KeyboardInterrupt)",
            )
            raise

        except Exception as exc:
            elapsed_total = time.perf_counter() - start_time
            save_run_summary(
                output_dir=self.output_dir,
                status="failed",
                final_step=self.global_step,
                tokens_seen=self.tokens_seen,
                elapsed_seconds=elapsed_total,
                final_train_loss=last_train_loss,
                final_val_loss=last_val_loss,
                best_val_loss=self.best_val_loss,
                error_message=str(exc),
            )
            raise

    def _append_metrics_record(self, record: dict[str, Any]) -> None:
        """Appends one JSONL metrics record and flushes immediately."""
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def _prune_metrics_after_resume(self, resume_step: int) -> None:
        """Drops metrics.jsonl records after `resume_step` so resumed runs do not duplicate steps."""
        if not self.metrics_file.exists():
            return

        kept: list[str] = []
        with open(self.metrics_file, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                step = record.get("step")
                if step is None or step <= resume_step:
                    kept.append(json.dumps(record))

        with open(self.metrics_file, "w", encoding="utf-8") as f:
            for serialized in kept:
                f.write(serialized + "\n")
