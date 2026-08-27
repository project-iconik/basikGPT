"""Training configuration dataclass for basikGPT pretraining."""

from dataclasses import dataclass
from typing import Literal

from basikgpt.training.compile import validate_compile_mode
from basikgpt.training.sdpa import validate_sdpa_kernel_name

Precision = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and runtime configuration for single-device baseline training.

    Maintains a strict boundary from GPTConfig: contains only optimization,
    scheduling, batching, precision, and environment parameters.
    """

    # Optimization
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0

    # Scheduling
    warmup_steps: int = 2000
    max_steps: int = 10000

    # Batching & Accumulation
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Evaluation & Checkpointing
    eval_interval: int = 500
    eval_batches: int = 20
    checkpoint_interval: int = 1000
    log_interval: int = 10
    eval_at_start: bool = False
    track_data_sample_index: bool = False
    save_step_final: bool = True
    stop_at_step: int | None = None
    checkpoint_steps: tuple[int, ...] | None = None

    # Runtime Environment & Precision
    device: str = "auto"
    precision: Precision = "fp32"
    output_dir: str = "runs/baseline"
    seed: int = 1337

    # Opt-in performance knobs. Defaults preserve the Milestone 14 uncompiled path.
    compile: bool = False
    compile_mode: str = "default"
    sdpa_kernel: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_learning_rate < 0 or self.min_learning_rate > self.learning_rate:
            raise ValueError(
                f"min_learning_rate must satisfy 0 <= min_lr <= lr, got {self.min_learning_rate} (lr={self.learning_rate})"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not (0.0 <= self.beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {self.beta1}")
        if not (0.0 <= self.beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {self.beta2}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.warmup_steps > self.max_steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) cannot exceed max_steps ({self.max_steps})")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive or None, got {self.max_grad_norm}")
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        if self.eval_batches <= 0:
            raise ValueError(f"eval_batches must be positive, got {self.eval_batches}")
        if self.checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {self.checkpoint_interval}")
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        if self.stop_at_step is not None:
            if self.stop_at_step <= 0:
                raise ValueError(f"stop_at_step must be positive or None, got {self.stop_at_step}")
            if self.stop_at_step > self.max_steps:
                raise ValueError(
                    f"stop_at_step ({self.stop_at_step}) cannot exceed max_steps ({self.max_steps})"
                )
        if self.checkpoint_steps is not None:
            if len(self.checkpoint_steps) == 0:
                raise ValueError("checkpoint_steps must be non-empty when provided")
            for step in self.checkpoint_steps:
                if step <= 0 or step > self.max_steps:
                    raise ValueError(
                        f"checkpoint_steps values must be in 1..max_steps, got {self.checkpoint_steps}"
                    )
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"precision must be one of 'fp32', 'bf16', 'fp16', got '{self.precision}'")
        validate_compile_mode(self.compile_mode)
        object.__setattr__(self, "sdpa_kernel", validate_sdpa_kernel_name(self.sdpa_kernel))
