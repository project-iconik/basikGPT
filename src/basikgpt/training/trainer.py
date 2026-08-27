"""Single-device baseline pretraining loop and orchestration for basikGPT."""

from collections.abc import Iterator
import json
import math
from pathlib import Path
import time
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.loss import compute_cross_entropy_loss
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.scheduler import get_learning_rate_at_step, update_learning_rate


def resolve_device(device_str: str) -> torch.device:
    """Resolves device string ('auto', 'cpu', 'cuda') into a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class Trainer:
    """Orchestrates single-device FP32 autoregressive pretraining for basikGPT."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)

        self.model = model.to(self.device)
        self.optimizer = configure_optimizers(self.model, self.config)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.global_step = 0
        self.tokens_seen = 0

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.jsonl"

    def _infinite_loader(self, loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yields batches infinitely from a DataLoader by looping across epochs."""
        while True:
            for batch in loader:
                yield batch

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

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                if i >= num_batches:
                    break
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.model(x)
                loss = compute_cross_entropy_loss(logits, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"Non-finite loss detected during evaluation at batch {i}: {loss.item()}")

                total_loss += loss.item()
                batches_evaluated += 1

        if was_training:
            self.model.train()

        return total_loss / max(1, batches_evaluated)

    def train_step(self, data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        """Executes a single optimizer step comprising `gradient_accumulation_steps` micro-batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_tokens = 0
        accum_steps = self.config.gradient_accumulation_steps

        for _ in range(accum_steps):
            x, y = next(data_iter)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = compute_cross_entropy_loss(logits, y)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"Non-finite training loss at global step {self.global_step}: {loss.item()}")

            # Scale loss for gradient accumulation
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            step_loss += loss.item() / accum_steps
            step_tokens += y.numel()

        # Gradient clipping
        if self.config.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            ).item()
        else:
            grad_norm = 0.0

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at global step {self.global_step}: {grad_norm}")

        # Update learning rate according to schedule
        lr = get_learning_rate_at_step(self.global_step, self.config)
        update_learning_rate(self.optimizer, lr)

        # Optimizer update
        self.optimizer.step()

        self.tokens_seen += step_tokens
        self.global_step += 1

        return {
            "loss": step_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "step_tokens": step_tokens,
        }

    def train(self, resume_from: Path | str | None = None) -> list[dict[str, Any]]:
        """Runs the full training loop until `max_steps` is reached."""
        if resume_from is not None:
            meta = load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                device=self.device,
            )
            self.global_step = meta.get("global_step", 0)
            self.tokens_seen = meta.get("tokens_seen", 0)
            print(f"[Trainer] Resumed from {resume_from} at step {self.global_step}, tokens {self.tokens_seen:,}")

        data_iter = self._infinite_loader(self.train_loader)
        history: list[dict[str, Any]] = []

        start_time = time.perf_counter()
        last_log_time = start_time
        last_tokens_seen = self.tokens_seen

        print(f"[Trainer] Starting training on device '{self.device}' (FP32 baseline) ...")
        print(f"  Target steps:       {self.config.max_steps:,}")
        print(f"  Batch size:         {self.config.batch_size}")
        print(f"  Grad accumulation:  {self.config.gradient_accumulation_steps}")
        print(f"  Warmup steps:       {self.config.warmup_steps:,}")
        print(f"  Peak LR:            {self.config.learning_rate:.2e}")
        print(f"  Min LR:             {self.config.min_learning_rate:.2e}")
        print(f"  Output dir:         {self.output_dir}")

        while self.global_step < self.config.max_steps:
            step_metrics = self.train_step(data_iter)

            # Evaluation
            val_loss = None
            if self.val_loader is not None and (
                self.global_step % self.config.eval_interval == 0
                or self.global_step == self.config.max_steps
            ):
                val_loss = self.evaluate()

            # Checkpoint
            if (
                self.global_step % self.config.checkpoint_interval == 0
                or self.global_step == self.config.max_steps
            ):
                ckpt_path = self.output_dir / f"step-{self.global_step:08d}.pt"
                save_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    global_step=self.global_step,
                    tokens_seen=self.tokens_seen,
                    training_config=self.config,
                    model_config=getattr(self.model, "config", None),
                )

            # Logging
            if (
                self.global_step % self.config.log_interval == 0
                or self.global_step == self.config.max_steps
            ):
                now = time.perf_counter()
                elapsed_since_log = now - last_log_time
                tokens_delta = self.tokens_seen - last_tokens_seen
                tok_per_sec = tokens_delta / max(1e-6, elapsed_since_log)

                log_entry = {
                    "step": self.global_step,
                    "tokens_seen": self.tokens_seen,
                    "train_loss": step_metrics["loss"],
                    "grad_norm": step_metrics["grad_norm"],
                    "lr": step_metrics["lr"],
                    "tokens_per_sec": tok_per_sec,
                    "elapsed_sec": now - start_time,
                }
                if val_loss is not None:
                    log_entry["val_loss"] = val_loss

                history.append(log_entry)

                val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
                print(
                    f"Step {self.global_step:06d}/{self.config.max_steps:06d} | "
                    f"Loss: {step_metrics['loss']:.4f} | "
                    f"GradNorm: {step_metrics['grad_norm']:.3f} | "
                    f"LR: {step_metrics['lr']:.2e} | "
                    f"Tokens: {self.tokens_seen:,} | "
                    f"{tok_per_sec:,.0f} tok/s{val_str}"
                )

                # Write metrics line
                with open(self.metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                last_log_time = now
                last_tokens_seen = self.tokens_seen

        # Final save
        final_ckpt = self.output_dir / "step-final.pt"
        save_checkpoint(
            checkpoint_path=final_ckpt,
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            training_config=self.config,
            model_config=getattr(self.model, "config", None),
        )
        print(f"[Trainer] Training completed successfully. Final checkpoint saved to {final_ckpt}")
        return history
