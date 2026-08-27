"""CPU-safe tests for torch.compile helpers, checkpoint unwrap, and Trainer opt-in defaults."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.compile import (
    compile_model,
    state_dict_has_compile_wrapper_keys,
    unwrap_compiled_model,
)
from basikgpt.training.config import TrainingConfig
from basikgpt.training.optimizer import configure_optimizers
from basikgpt.training.trainer import Trainer


def _tiny_gpt() -> GPT:
    cfg = GPTConfig(
        vocab_size=32,
        context_length=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    return GPT(cfg)


def test_training_config_compile_defaults_disabled() -> None:
    cfg = TrainingConfig(warmup_steps=0)
    assert cfg.compile is False
    assert cfg.compile_mode == "default"
    assert cfg.sdpa_kernel == "auto"


def test_invalid_compile_mode_raises() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        TrainingConfig(warmup_steps=0, compile_mode="max-autotune")


def test_invalid_sdpa_kernel_raises() -> None:
    with pytest.raises(ValueError, match="sdpa_kernel"):
        TrainingConfig(warmup_steps=0, sdpa_kernel="flashattn2")


def test_cpu_compile_fail_fast() -> None:
    model = _tiny_gpt()
    ds = TensorDataset(torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
    loader = DataLoader(ds, batch_size=2)
    with pytest.raises(ValueError, match="only supported on CUDA"):
        Trainer(model, TrainingConfig(device="cpu", compile=True, warmup_steps=0), loader)


def test_unwrap_eager_compile_and_checkpoint_keys(tmp_path: Path) -> None:
    """Uses the eager compile backend so CPU tests do not require Inductor/GPU."""
    model = _tiny_gpt()
    compiled = torch.compile(model, backend="eager")
    unwrapped = unwrap_compiled_model(compiled)
    assert unwrapped is model
    assert not state_dict_has_compile_wrapper_keys(unwrapped.state_dict())

    train_cfg = TrainingConfig(warmup_steps=0)
    optimizer = configure_optimizers(unwrapped, train_cfg)
    ckpt = tmp_path / "unwrap.pt"
    save_checkpoint(
        ckpt,
        compiled,
        optimizer,
        global_step=1,
        tokens_seen=8,
        training_config=train_cfg,
        model_config=model.config,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert not state_dict_has_compile_wrapper_keys(payload["model_state_dict"])

    fresh = _tiny_gpt()
    fresh_opt = configure_optimizers(fresh, train_cfg)
    load_checkpoint(ckpt, fresh, fresh_opt, device="cpu")
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p1, p2)


def test_compile_model_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        compile_model(_tiny_gpt(), mode="max-autotune")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_compiled_forward_backward_and_checkpoint(tmp_path: Path) -> None:
    """Compiled tiny CUDA train_step is finite and checkpoints load into an uncompiled model."""
    torch.cuda.empty_cache()
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        attention_backend="sdpa",
    )
    raw_tokens = torch.randint(0, 64, (4, 17), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:]), batch_size=2)
    train_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        compile=True,
        compile_mode="default",
        output_dir=str(tmp_path / "compiled"),
        seed=1337,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=1,
    )
    trainer = Trainer(GPT(cfg), train_cfg, loader, overwrite=True)
    step = trainer.train_step(iter(trainer._infinite_loader(loader)))
    assert torch.isfinite(torch.tensor(step["loss"]))
    ckpt = tmp_path / "compiled.pt"
    save_checkpoint(
        ckpt,
        trainer.raw_model,
        trainer.optimizer,
        global_step=1,
        tokens_seen=step["step_tokens"],
        training_config=train_cfg,
        model_config=cfg,
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert not state_dict_has_compile_wrapper_keys(payload["model_state_dict"])

    uncompiled = GPT(cfg)
    opt = configure_optimizers(uncompiled, TrainingConfig(warmup_steps=0, device="cpu"))
    load_checkpoint(ckpt, uncompiled, opt, device="cpu")

    resume_cfg = TrainingConfig(
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        device="cuda",
        precision="fp32",
        compile=True,
        compile_mode="default",
        output_dir=str(tmp_path / "resume"),
        seed=1337,
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=10_000,
    )
    resumed = Trainer(GPT(cfg), resume_cfg, loader, overwrite=True)
    load_checkpoint(ckpt, resumed.raw_model, resumed.optimizer, device=resumed.device)
    resumed_step = resumed.train_step(iter(resumed._infinite_loader(loader)))
    assert torch.isfinite(torch.tensor(resumed_step["loss"]))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this machine")
def test_cuda_compiled_bf16_smoke() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this CUDA GPU")
    cfg = GPTConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        attention_backend="sdpa",
    )
    raw_tokens = torch.randint(0, 64, (4, 17), dtype=torch.long)
    loader = DataLoader(TensorDataset(raw_tokens[:, :-1], raw_tokens[:, 1:]), batch_size=2)
    train_cfg = TrainingConfig(
        warmup_steps=0,
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=2,
        device="cuda",
        precision="bf16",
        compile=True,
        compile_mode="default",
        log_interval=10_000,
        eval_interval=10_000,
        checkpoint_interval=10_000,
    )
    trainer = Trainer(GPT(cfg), train_cfg, loader)
    step = trainer.train_step(iter(trainer._infinite_loader(loader)))
    assert torch.isfinite(torch.tensor(step["loss"]))
    assert torch.isfinite(torch.tensor(step["grad_norm"]))
