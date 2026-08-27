# Pretraining Recipe & Fidelity Audit Guide (basikGPT)

This document presents a comprehensive audit and comparison between the historical 2019 OpenAI GPT-2 pretraining setup and the modernized **`basikGPT`** pretraining recipe.

---

## 1. Project Framing & Fidelity Classification

**Canonical Reproduction Statement**:
> **`basikGPT` is a GPT-2 architecture reproduction implemented with a modern PyTorch pretraining stack and the FineWeb-Edu corpus.**

To ensure complete transparency and rigorous engineering standards, `basikGPT` classifies reproduction aspects across four explicit tiers:

```text
1. Architecture Fidelity:
   Bitwise structural equivalence to GPT-2 Small (12 layers, 12 heads, 768 d_model, 3072 d_ff,
   Pre-LayerNorm topology, GELU-tanh activation, vocab=50,257, context_length=1,024,
   and parameter count strictly 124,439,808).

2. Checkpoint Fidelity:
   Bitwise parity in logit generation and candidate scoring when loading official public
   reference weights (openai-community/gpt2).

3. Initialization Fidelity:
   GPT-2-compatible modernized initialization implementing standard normal distributions
   (std=0.02) and depth-dependent residual projection scaling (std=0.02 / sqrt(2*L)).

4. Training Recipe Modernization:
   Adoption of modern, proven PyTorch optimization practices (AdamW decoupled weight decay,
   beta_1=0.9/beta_2=0.95, 2D/1D parameter grouping, PyTorch SDPA, Cosine Decay, FineWeb-Edu).
```

`basikGPT` does **not** claim byte-for-byte reproduction of the 2019 OpenAI training run (which utilized a closed, unreleased WebText dataset and legacy TensorFlow 1.x runtime).

---

## 2. Comprehensive Comparison Table

| Dimension / Hyperparameter | Original OpenAI GPT-2 (2019) | basikGPT Canonical Recipe | Status / Rationale |
|---|---|---|---|
| **Architecture** | GPT-2 Small (124M) | GPT-2 Small (124M) | **Faithful**: Identical 12 layers, 12 heads, 768 d_model, 3072 d_ff |
| **Parameter Count** | 124,439,808 unique | 124,439,808 unique | **Faithful**: Bitwise exact parameter count matching |
| **Vocabulary Size** | 50,257 (BPE) | 50,257 (BPE via tiktoken `gpt2`) | **Faithful**: Identical token vocabulary |
| **Context Length** | 1,024 tokens | 1,024 tokens | **Faithful**: Identical context window |
| **Weight Tying** | `lm_head.weight` is `wte.weight` | `lm_head.weight` is `wte.weight` | **Faithful**: Exact memory tensor sharing |
| **Activation Function** | GELU (tanh approx) | GELU (tanh approx, `approximate="tanh"`) | **Faithful**: Exact numerical formulation |
| **Layer Normalization** | Pre-Norm ($\epsilon = 10^{-5}$) | Pre-Norm ($\epsilon = 10^{-5}$) | **Faithful**: Exact Pre-LayerNorm placement |
| **Base Initialization Std** | $0.02$ | $0.02$ (`initializer_range`) | **GPT-2-compatible modernized init**: Normal with $\sigma = 0.02$ |
| **Residual Projection Scaling** | $\frac{0.02}{\sqrt{2L}} \approx 0.004082$ | $\frac{0.02}{\sqrt{2L}} \approx 0.004082$ | **GPT-2-compatible modernized init**: Radford et al., 2019 Section 2.3 |
| **Dropout Policy** | 0.1 uniform | Config default `0.1`; pretraining CLI uses `0.0` | **Split**: architecture default matches GPT-2/HF; training CLI is modernized (see §4) |
| **Optimizer** | Classic Adam with L2 decay | **AdamW** (Decoupled Weight Decay) | **Modernized**: Loshchilov & Hutter (2017) prevents gradient-scale corruption |
| **Optimizer $\beta_1, \beta_2$** | $\beta_1 = 0.9, \beta_2 = 0.98$ (or 0.999) | $\beta_1 = 0.9, \beta_2 = 0.95$ | **Modernized**: Industry standard for Transformer stability (GPT-3/nanoGPT) |
| **Weight Decay** | 0.01 (L2) | 0.1 (Decoupled) | **Modernized**: Applied strictly to 2D matrices; 0.0 for 1D biases/LayerNorms |
| **Gradient Clipping** | Not standard in early TF | `max_grad_norm = 1.0` | **Modernized**: Prevents gradient spikes in mixed precision / deep stacks |
| **Attention Backend** | Manual matrix multiplication | **PyTorch SDPA** (FlashAttention / Mem-Efficient) | **Modernized**: High-efficiency hardware acceleration without numerical divergence |
| **Learning Rate Schedule** | Linear Warmup + Cosine Decay | Linear Warmup (2,000 steps) + Cosine Decay to $0.1 \times \text{lr}$ | **Modernized**: Stable convergence curve |
| **Training Corpus** | WebText (~40GB private dataset) | **FineWeb-Edu** (10B sample / 2.5B target) | **Project-Specific**: Open, high educational quality web crawl |
| **Token Budget** | ~40 Billion tokens | ~2.5 Billion tokens (**Chinchilla-inspired**) | **Project-Specific**: ~20 tokens per parameter ratio for 124M model |

---

## 3. Parameter Grouping Policy

In `basikGPT`, the optimizer parameter grouping is explicitly configured via [`configure_optimizers`](../src/basikgpt/training/optimizer.py):

```python
# Decay Group (weight_decay = 0.1):
# All 2D+ tensors: Linear projection weights, Token & Positional embeddings
- attn.qkv_proj.weight
- attn.out_proj.weight
- mlp.fc_in.weight
- mlp.fc_out.weight
- wte.weight (and tied lm_head.weight registered once via id deduplication)
- wpe.weight

# No-Decay Group (weight_decay = 0.0):
# All 1D tensors: LayerNorm parameters and additive Linear biases
- ln_1.weight, ln_1.bias
- ln_2.weight, ln_2.bias
- ln_f.weight, ln_f.bias
- attn.qkv_proj.bias, attn.out_proj.bias
- mlp.fc_in.bias, mlp.fc_out.bias
```

---

## 4. Dropout Fidelity Audit

In the original GPT-2 specification:
- `embd_pdrop = 0.1` (applied to token and position embeddings)
- `attn_pdrop = 0.1` (applied to attention softmax matrix)
- `resid_pdrop = 0.1` (applied to attention output and MLP output projections)

Because all three rates are identical ($0.1$), `basikGPT` unifies them under a single hyperparameter `GPTConfig.dropout` (default `0.1`). This default preserves GPT-2 / HuggingFace inference and reference-parity behavior.

**Pretraining recipe (modernized):** `scripts/train.py` and `scripts/run_pilot.py` instantiate models with `dropout=0.0`. That is an intentional nanoGPT-style training choice, not a claim that the 2019 GPT-2 run used zero dropout. To train with the original 0.1 rates, pass a `GPTConfig` with `dropout=0.1` instead of the CLI default.
