"""Integration test comparing basikGPT greedy generation token-for-token against Hugging Face GPT-2."""

from pathlib import Path
import pytest
import torch

from basikgpt.config import GPTConfig
from basikgpt.conversion.gpt2 import load_hf_gpt2_weights
from basikgpt.data.tokenizer import GPT2Tokenizer
from basikgpt.generation.config import GenerationConfig
from basikgpt.generation.generate import generate
from basikgpt.model.gpt import GPT
from basikgpt.training.checkpoint import (
    load_model_from_checkpoint,
    save_checkpoint,
)
from basikgpt.training.config import TrainingConfig

try:
    from transformers import GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers package required for reference parity")
def test_reference_gpt2_greedy_token_parity() -> None:
    """Verifies that basikGPT produces the EXACT same token ID sequence as HuggingFace GPT-2 in greedy mode."""
    # 1. Load HuggingFace Reference Model
    hf_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    hf_model.eval()

    # 2. Instantiate and convert basikGPT model
    cfg = GPTConfig.gpt2_small(dropout=0.0)
    basik_model = GPT(cfg)
    load_hf_gpt2_weights(basik_model, "openai-community/gpt2")
    basik_model.eval()

    # 3. Prompt setup
    prompt = "The history of artificial intelligence"
    tokenizer = GPT2Tokenizer()
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

    # 4. Generate with HuggingFace (Greedy, 10 new tokens)
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=50256,
        )

    # 5. Generate with basikGPT (Greedy, 10 new tokens)
    gen_cfg = GenerationConfig(max_new_tokens=10, do_sample=False)
    basik_out = generate(basik_model, input_ids, gen_cfg)

    # 6. Verify exact token parity
    assert hf_out.shape == basik_out.shape
    assert torch.equal(hf_out, basik_out), f"Token mismatch:\nHF:    {hf_out.tolist()}\nBasik: {basik_out.tolist()}"


def test_load_model_from_checkpoint_weight_tying(tmp_path: Path) -> None:
    """Verifies that load_model_from_checkpoint reconstructs GPT and preserves weight tying."""
    cfg = GPTConfig(vocab_size=64, context_length=16, n_layers=2, n_heads=2, d_model=32, d_ff=128)
    model = GPT(cfg)
    train_cfg = TrainingConfig()
    opt = torch.optim.AdamW(model.parameters())

    ckpt_file = tmp_path / "model_ckpt.pt"
    save_checkpoint(
        ckpt_file,
        model=model,
        optimizer=opt,
        global_step=42,
        tokens_seen=1024,
        training_config=train_cfg,
        model_config=cfg,
    )

    loaded_model, meta = load_model_from_checkpoint(ckpt_file, device="cpu")
    assert isinstance(loaded_model, GPT)
    assert meta["global_step"] == 42
    assert meta["tokens_seen"] == 1024
    assert meta["model_config"].vocab_size == 64
    assert loaded_model.lm_head.weight is loaded_model.wte.weight
    assert not loaded_model.training
