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

from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.compile import compile_model
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.metadata import save_run_metadata, save_run_summary
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
        self.best_val_loss: float | None = None
        self.last_val_loss: float | None = None

        # 5. Save Initial Run Provenance Metadata (always from the uncompiled module)
        if hasattr(self.raw_model, "config") and self.resume_from is None:
            save_run_metadata(
                output_dir=self.output_dir,
                run_name=self.run_name,
                model_config=self.raw_model.config,
                training_config=self.config,
                dataset_manifest=dataset_manifest,
                dataset_manifest_path=dataset_manifest_path,
            )

    def _infinite_loader(self, loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields batches infinitely from a DataLoader by looping across epochs."""
        while True:
            for batch in loader:
                yield batch

    def _sdpa_context(self):
        """Forces a single SDPA backend when configured; `auto` leaves PyTorch dispatch unchanged."""
        return sdpa_kernel_context(self.config.sdpa_kernel)

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
            self._prune_metrics_after_resume(self.global_step)
            print(f"[Trainer] Resumed from {target_resume} at step {self.global_step}, tokens {self.tokens_seen:,}")

        data_iter = self._infinite_loader(self.train_loader)
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

        try:
            while self.global_step < self.config.max_steps:
                step_metrics = self.train_step(data_iter)
                last_train_loss = step_metrics["loss"]

                should_eval = self.val_loader is not None and (
                    self.global_step % self.config.eval_interval == 0
                    or self.global_step == self.config.max_steps
                )
                should_log = (
                    self.global_step % self.config.log_interval == 0
                    or self.global_step == self.config.max_steps
                )
                should_ckpt = (
                    self.global_step % self.config.checkpoint_interval == 0
                    or self.global_step == self.config.max_steps
                )

                val_loss = None
                if should_eval:
                    val_loss = self.evaluate()
                    last_val_loss = val_loss
                    self.last_val_loss = val_loss

                if should_ckpt:
                    ckpt_path = self.output_dir / f"step-{self.global_step:08d}.pt"
                    save_checkpoint(
                        checkpoint_path=ckpt_path,
                        model=self.raw_model,
                        optimizer=self.optimizer,
                        global_step=self.global_step,
                        tokens_seen=self.tokens_seen,
                        training_config=self.config,
                        model_config=getattr(self.raw_model, "config", None),
                        scaler=self.scaler,
                    )

                elapsed_total = time.perf_counter() - start_time

                if should_log:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)

                    now = time.perf_counter()
                    elapsed_since_log = now - last_log_time
                    tokens_delta = self.tokens_seen - last_tokens_seen
                    tok_per_sec = tokens_delta / max(1e-6, elapsed_since_log)
                    elapsed_total = now - start_time

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
                        "tokens_per_sec": tok_per_sec,
                        "elapsed_seconds": elapsed_total,
                    }
                    if peak_allocated_vram_bytes is not None:
                        train_record["peak_allocated_vram_bytes"] = peak_allocated_vram_bytes
                        train_record["peak_reserved_vram_bytes"] = peak_reserved_vram_bytes
                        # Allocated bytes expressed in MiB for log readability. Not nvidia-smi usage.
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
                    val_record = {
                        "type": "val",
                        "step": self.global_step,
                        "tokens_seen": self.tokens_seen,
                        "val_loss": val_loss,
                        "elapsed_seconds": elapsed_total,
                    }
                    self._append_metrics_record(val_record)

            # Final checkpoint & completed summary
            final_ckpt = self.output_dir / "step-final.pt"
            save_checkpoint(
                checkpoint_path=final_ckpt,
                model=self.raw_model,
                optimizer=self.optimizer,
                global_step=self.global_step,
                tokens_seen=self.tokens_seen,
                training_config=self.config,
                model_config=getattr(self.raw_model, "config", None),
                scaler=self.scaler,
            )
            save_run_summary(
                output_dir=self.output_dir,
                status="completed",
                final_step=self.global_step,
                tokens_seen=self.tokens_seen,
                elapsed_seconds=time.perf_counter() - start_time,
                final_train_loss=last_train_loss,
                final_val_loss=last_val_loss,
                best_val_loss=self.best_val_loss,
                checkpoint_path=final_ckpt,
            )
            print(f"[Trainer] Training completed successfully. Final checkpoint saved to {final_ckpt}")
            return history

        except KeyboardInterrupt:
            elapsed_total = time.perf_counter() - start_time
            print(f"\n[Trainer] KeyboardInterrupt received. Saving emergency checkpoint at step {self.global_step} ...")
            interrupted_ckpt = self.output_dir / "step-interrupted.pt"
            save_checkpoint(
                checkpoint_path=interrupted_ckpt,
                model=self.raw_model,
                optimizer=self.optimizer,
                global_step=self.global_step,
                tokens_seen=self.tokens_seen,
                training_config=self.config,
                model_config=getattr(self.raw_model, "config", None),
                scaler=self.scaler,
            )
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
