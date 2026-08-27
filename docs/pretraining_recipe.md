# Pretraining Recipe & Fidelity Audit Guide (basikGPT)

This document presents a comprehensive audit and comparison between the historical 2019 OpenAI GPT-2 pretraining setup and the modernized **`basikGPT`** pretraining recipe.

---

## 1. Project Framing & Scope

**Reproduction Scope Statement**:
> `basikGPT` is a **GPT-2 architecture reproduction** implemented with a **modern PyTorch pretraining stack** and the **FineWeb-Edu corpus**.
> It preserves 100% architectural and mathematical equivalence to GPT-2 Small, while adopting industry-standard optimization improvements (such as AdamW, PyTorch SDPA, and decoupled parameter grouping) developed since 2019.

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
| **Base Initialization Std** | $0.02$ | $0.02$ (`initializer_range`) | **Faithful**: Standard normal with $\sigma = 0.02$ |
| **Residual Projection Scaling** | $\frac{0.02}{\sqrt{2L}} \approx 0.004082$ | $\frac{0.02}{\sqrt{2L}} \approx 0.004082$ | **Faithful**: Radford et al., 2019 Section 2.3 |
| **Dropout Policy** | 0.1 uniform | 0.1 uniform (`dropout = 0.1`) | **Faithful**: Uniform dropout across embeddings, attn, residual |
| **Optimizer** | Classic Adam with L2 decay | **AdamW** (Decoupled Weight Decay) | **Modernized**: Loshchilov & Hutter (2017) prevents gradient-scale corruption |
| **Optimizer $\beta_1, \beta_2$** | $\beta_1 = 0.9, \beta_2 = 0.98$ (or 0.999) | $\beta_1 = 0.9, \beta_2 = 0.95$ | **Modernized**: Industry standard for Transformer stability (GPT-3/nanoGPT) |
| **Weight Decay** | 0.01 (L2) | 0.1 (Decoupled) | **Modernized**: Applied strictly to 2D matrices; 0.0 for 1D biases/LayerNorms |
| **Gradient Clipping** | Not standard in early TF | `max_grad_norm = 1.0` | **Modernized**: Prevents gradient spikes in mixed precision / deep stacks |
| **Attention Backend** | Manual matrix multiplication | **PyTorch SDPA** (FlashAttention / Mem-Efficient) | **Modernized**: High-efficiency hardware acceleration without numerical divergence |
| **Learning Rate Schedule** | Linear Warmup + Cosine Decay | Linear Warmup (2,000 steps) + Cosine Decay to $0.1 \times \text{lr}$ | **Modernized**: Stable convergence curve |
| **Training Corpus** | WebText (~40GB private dataset) | **FineWeb-Edu** (10B sample / 2.5B target) | **Project-Specific**: Open, high educational quality web crawl |
| **Token Budget** | ~40 Billion tokens | ~2.5 Billion tokens (Chinchilla compute-optimal) | **Project-Specific**: 20 tokens per parameter ratio |

---

## 3. Parameter Grouping Policy

In `basikGPT`, the optimizer parameter grouping is explicitly configured via [`configure_optimizers`](file:///C:/Users/jmint/.gemini/antigravity/scratch/basikGPT/src/basikgpt/training/optimizer.py):

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

Because all three rates are identical ($0.1$) during pretraining, `basikGPT` unifies them under a single, cohesive hyperparameter `GPTConfig.dropout = 0.1`. This avoids unnecessary configuration sprawl while maintaining exact behavioral equivalence.
