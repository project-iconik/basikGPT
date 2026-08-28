# basikGPT

> **Educational, Reproducible, and Open-Source GPT-2 Small Pretraining from Scratch in PyTorch**

`basikGPT` is an educational project aimed at building, training, and scaling a decoder-only Transformer from the ground up. The code is crafted to be thoroughly reverse-engineered, read line-by-line, reviewed, and studied by engineers learning LLM architecture and pretraining mechanics.

---

## Engineering Philosophy & Priorities

In every line of code, we enforce the following hierarchy:

$$\text{Correctness} > \text{Readability} > \text{Verifiability} > \text{Reproducibility} > \text{Performance Optimization} > \text{Code Conciseness}$$

- **No Premature Abstraction**: We avoid speculative base classes or deep inheritance hierarchies.
- **Traceable Tensor Operations**: Every shape transformation is explicitly annotated and documented.
- **Dual Attention Backends**: An explicit `eager` implementation for learning and verification alongside an optimized `sdpa` path for fast pretraining.
- **Step-by-Step Milestones**: Development is divided into 16 strict milestones (Milestone 0 to 15) to guarantee correctness at every stage.

---

## GPT-2 Small Canonical Architecture

The canonical architecture accurately reproduces the 124M-parameter GPT-2 Small configuration:

| Hyperparameter | Value | Description |
|---|---|---|
| `vocab_size` | `50,257` | GPT-2 standard vocabulary size |
| `context_length` | `1,024` | Maximum context sequence length |
| `n_layers` | `12` | Number of Transformer decoder blocks |
| `n_heads` | `12` | Number of attention heads |
| `d_model` | `768` | Model embedding / hidden dimension |
| `head_dim` | `64` | Dimension per head ($d_{\text{model}} / n_{\text{heads}}$) |
| `d_ff` | `3,072` | Feed-forward intermediate dimension ($4 \times d_{\text{model}}$) |
| `normalization` | `LayerNorm` | Pre-norm configuration ($\epsilon = 10^{-5}$) |
| `activation` | `GELU` | GPT-2 compatible GELU (tanh approximation) |
| `position_encoding` | `Learned` | Absolute learned positional embeddings |
| `weight_tying` | `True` | Token embedding weight tied to LM Head weight |
| `bias` | `True` | Biases included in Linear and LayerNorm layers |

---

## Repository Structure

```text
basikGPT/
├── pyproject.toml                  # Build system, dependencies, and test configuration
├── LICENSE                         # Apache-2.0 License
├── README.md                       # Project overview & quickstart
├── AGENTS.md                       # AI agent master guidelines and invariant specifications
│
├── src/basikgpt/
│   ├── config.py                   # GPTConfig dataclass, validation, and size presets
│   ├── model/                      # Attention, MLP, Block, full GPT assembly
│   ├── conversion/                 # HuggingFace GPT-2 checkpoint conversion
│   ├── data/                       # Tokenizer, sharding, FineWeb-Edu pipeline
│   ├── training/                   # Optimizer, scheduler, Trainer, checkpoints
│   ├── generation/                 # Sampling and KV-cache decoding
│   └── evaluation/                 # Validation loss / perplexity and HellaSwag
│
├── scripts/                        # CLI entrypoints (train, generate, evaluate, parity)
├── tests/                          # Unit and integration tests
└── docs/                           # Tensor conventions, recipe audit, pilot protocol
```

---

## Quickstart & Development

### 1. Installation

Requires Python $\ge 3.12$ and PyTorch $\ge 2.1.0$.

```bash
# Clone the repository
git clone https://github.com/project-iconik/basikGPT.git
cd basikGPT

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

Core model and training code depend only on `torch` and `numpy`. Tokenization and FineWeb download extras (`tiktoken`, `datasets`) are installed via `.[data]` or `.[dev]`.

### 2. Prepare Smoke Data

Token shards and `runs/` artifacts are gitignored. A fresh clone does **not** include `data/fineweb-edu-smoke`, so `scripts/train.py` will fail until shards exist.

```bash
# Small FineWeb-Edu smoke set (train/val uint16 shards + manifest.json)
python scripts/prepare_fineweb_edu.py --output-dir data/fineweb-edu-smoke
```

### 3. Running Unit Tests

```bash
pytest
```

### 4. Train a Tiny CPU Smoke Run

```bash
python scripts/train.py --model-preset tiny --max-steps 20 --device cpu
```

### 5. Basic Configuration Usage

```python
from basikgpt import GPTConfig

# Instantiate canonical GPT-2 Small configuration
config = GPTConfig.gpt2_small()

print(f"Model dimension: {config.d_model}")
print(f"Attention heads: {config.n_heads}")
print(f"Head dimension: {config.head_dim}")
print(f"Analytical parameter count (tied): {config.num_total_parameters():,}")
# Output: Analytical parameter count (tied): 124,439,808
```

`GPTConfig.dropout` defaults to `0.1` to match GPT-2 / HuggingFace. Pretraining CLIs (`scripts/train.py`, `scripts/run_pilot.py`) set `dropout=0.0` as a modernized training recipe.

---

## Milestone Roadmap

Status is tracked against the canonical numbering in `AGENTS.md`.

- [x] **Milestone 0**: Repository Foundation (Packaging, `GPTConfig`, validation, tests, docs)
- [x] **Milestone 1**: Eager Causal Self-Attention (pure PyTorch tensor ops)
- [x] **Milestone 2**: SDPA Backend & Numerical Parity Tests
- [x] **Milestone 3**: GPT-2 Components (MLP, Block, LayerNorm, Embeddings)
- [x] **Milestone 4**: Complete GPT-2 Small Assembly & Verification
- [x] **Milestone 5**: Reference GPT-2 Checkpoint Parity (Logits Match)
- [x] **Milestone 6**: English FineWeb Streaming & Token Pipeline
- [x] **Milestone 7**: Pretraining Engine (AdamW, Cosine Warmup, BF16)
- [ ] **Milestone 8**: Training Validation (1M $\to$ 10M $\to$ 100M $\to$ 500M tokens) — *partial: CPU Stage A/B pilots only*
- [ ] **Milestone 9**: Performance Benchmarking & Engineering — *partial: CPU tiny benchmark only*
- [x] **Milestone 10**: Canonical Pretraining (~2.5B FineWeb tokens)
- [ ] **Milestone 11**: Evaluation & Perplexity Analysis — *partial: evaluators implemented; trained-model eval pending*
- [ ] **Milestone 12**: Scaling Experiments
- [ ] **Milestone 13**: Distributed Training (DDP $\to$ FSDP)
- [ ] **Milestone 14**: 30B-ready Architectural Validation
- [ ] **Milestone 15**: Comprehensive Technical Whitepaper

Implemented outside the original numbered roadmap (used by later milestones): autoregressive generation with KV cache, a HellaSwag zero-shot evaluation engine, local CPU Stage A/B pilots, Milestone 14 RunPod GPU qualification (`docs/runpod.md`), Milestone 15 GPU performance engineering (`docs/performance.md`), Milestone 16 1M/10M configuration freeze (`docs/config_freeze.md`), and the 2.5B FineWeb-Edu main run (`docs/main_2p5b.md`).

On NVIDIA RTX PRO 4500 Blackwell, PyTorch 2.8.0+cu128, BF16, T=1024, `attention_backend=sdpa`:

- Uncompiled B=1 G=1: ≈ 30.1k tokens/sec, peak allocated ≈ 2.58 GiB
- Uncompiled B=16 G=1: ≈ 79.3k tokens/sec, peak allocated ≈ 16.5 GiB
- `torch.compile` inductor `default` B=16 G=1: ≈ 87.2k tokens/sec, peak allocated ≈ 16.0 GiB (opt-in)

Provisional frozen single-GPU pretraining baseline (1M/10M FineWeb-Edu pilots; not an optimal claim): **uncompiled BF16, SDPA auto, B=8, T=1024, G=8, 65,536 tokens/step**. Config: [`configs/gpt2_small_fineweb_edu_single_gpu.json`](configs/gpt2_small_fineweb_edu_single_gpu.json). The 2.5B-token FineWeb-Edu main run completed on this recipe: 38,147 steps, 2,500,001,792 tokens, ≈ 8.18 GPU-hours, training-only ≈ 85.1k tokens/sec, peak allocated ≈ 9.52 GiB, full-val PPL 25.92, HellaSwag `acc_norm` 29.33%. Logs and eval JSON: [`runs/main_2p5b/`](runs/main_2p5b/). Write-up: [`docs/main_2p5b.md`](docs/main_2p5b.md). Checkpoints (`.pt`) are local-only and are not in git.

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
