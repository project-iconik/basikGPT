"""Unit tests for checkpoint save, load, and state restoration."""

from pathlib import Path
import torch
from basikgpt.config import GPTConfig
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import load_checkpoint, save_checkpoint
from basikgpt.training.config import TrainingConfig
from basikgpt.training.optimizer import configure_optimizers


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """Verifies that save_checkpoint and load_checkpoint preserve all weights, states, and weight tying."""
    cfg = GPTConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=128,
    )
    model1 = GPT(cfg)
    train_cfg = TrainingConfig(learning_rate=1e-3)
    optimizer1 = configure_optimizers(model1, train_cfg)

    # Perform a dummy step to populate optimizer state
    input_ids = torch.randint(0, 128, (2, 16))
    loss = model1(input_ids).sum()
    loss.backward()
    optimizer1.step()

    ckpt_file = tmp_path / "test_ckpt.pt"
    save_checkpoint(
        checkpoint_path=ckpt_file,
        model=model1,
        optimizer=optimizer1,
        global_step=42,
        tokens_seen=1337,
        training_config=train_cfg,
        model_config=cfg,
    )

    assert ckpt_file.exists()

    # Load into fresh model & optimizer
    model2 = GPT(cfg)
    optimizer2 = configure_optimizers(model2, train_cfg)

    meta = load_checkpoint(ckpt_file, model2, optimizer2)

    assert meta["global_step"] == 42
    assert meta["tokens_seen"] == 1337

    # Parameter equality
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)

    # Weight tying preserved
    assert model2.lm_head.weight is model2.wte.weight
