# basikGPT

> **Educational, Reproducible, and Open-Source GPT-2 Small Pretraining from Scratch in PyTorch**

`basikGPT` is an educational project aimed at building, training, and scaling a decoder-only Transformer from the ground up. The code is crafted to be thoroughly reverse-engineered, read line-by-line, reviewed, and studied by engineers learning LLM architecture and pretraining mechanics.

---

## 🎯 Engineering Philosophy & Priorities

In every line of code, we enforce the following hierarchy:

$$\text{Correctness} > \text{Readability} > \text{Verifiability} > \text{Reproducibility} > \text{Performance Optimization} > \text{Code Conciseness}$$

- **No Premature Abstraction**: We avoid speculative base classes or deep inheritance hierarchies.
- **Traceable Tensor Operations**: Every shape transformation is explicitly annotated and documented.
- **Dual Attention Backends**: An explicit `eager` implementation for learning and verification alongside an optimized `sdpa` path for fast pretraining.
- **Step-by-Step Milestones**: Development is divided into 16 strict milestones (Milestone 0 to 15) to guarantee correctness at every stage.

---

## 🏛️ GPT-2 Small Canonical Architecture

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

## 📁 Repository Structure

```text
basikGPT/
├── pyproject.toml              # Build system, dependencies, and test configuration
├── LICENSE                     # Apache-2.0 License
├── README.md                   # Project overview & quickstart
├── AGENTS.md                   # AI agent master guidelines and invariant specifications
│
├── src/
│   └── basikgpt/
│       ├── __init__.py         # Package root exporting core classes and version
│       └── config.py           # GPTConfig dataclass with strict validation & presets
│
├── docs/
│   └── tensor_conventions.md   # Tensor shape conventions and notation guide
│
└── tests/
    ├── __init__.py
    └── test_config.py          # Unit tests for GPTConfig and presets
```

---

## 🚀 Quickstart & Development

### 1. Installation

Requires Python $\ge 3.12$ and PyTorch $\ge 2.0.0$.

```bash
# Clone the repository
git clone https://github.com/your-username/basikGPT.git
cd basikGPT

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Running Unit Tests

```bash
pytest
```

### 3. Basic Configuration Usage

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

---

## 🗺️ Milestone Roadmap

- [x] **Milestone 0**: Repository Foundation (Packaging, `GPTConfig`, validation, tests, docs)
- [ ] **Milestone 1**: Eager Causal Self-Attention (pure PyTorch tensor ops)
- [ ] **Milestone 2**: SDPA Backend & Numerical Parity Tests
- [ ] **Milestone 3**: GPT-2 Components (MLP, Block, LayerNorm, Embeddings)
- [ ] **Milestone 4**: Complete GPT-2 Small Assembly & Verification
- [ ] **Milestone 5**: Reference GPT-2 Checkpoint Parity (Logits Match)
- [ ] **Milestone 6**: English FineWeb Streaming & Token Pipeline
- [ ] **Milestone 7**: Pretraining Engine (AdamW, Cosine Warmup, BF16)
- [ ] **Milestone 8**: Training Validation (1M $\to$ 10M $\to$ 100M $\to$ 500M tokens)
- [ ] **Milestone 9**: Performance Benchmarking & Engineering
- [ ] **Milestone 10**: Canonical Pretraining (~2.5B FineWeb tokens)
- [ ] **Milestone 11**: Evaluation & Perplexity Analysis
- [ ] **Milestone 12**: Scaling Experiments
- [ ] **Milestone 13**: Distributed Training (DDP $\to$ FSDP)
- [ ] **Milestone 14**: 30B-ready Architectural Validation
- [ ] **Milestone 15**: Comprehensive Technical Whitepaper

---

## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).
