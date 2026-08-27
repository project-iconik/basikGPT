# GPT-2-Compatible Modernized Initialization & Variance Control Guide (basikGPT)

This document details the mathematical theory, reference origins, fidelity classification, and explicit implementation of parameter initialization in **`basikGPT`** for from-scratch pretraining.

---

## 1. Project Framing & Fidelity Classification

`basikGPT` is defined as:
> **"A GPT-2 architecture reproduction with a modern PyTorch pretraining stack and FineWeb-Edu corpus."**

To maintain rigorous engineering fidelity without misrepresenting historical artifacts, `basikGPT` distinguishes across four explicit tiers of fidelity:

```text
1. Architecture Fidelity:
   Bitwise structural equivalence to GPT-2 Small (12 layers, 12 heads, d_model=768, d_ff=3072,
   Pre-LayerNorm topology, GELU-tanh activation, vocab=50,257, context_length=1,024,
   and parameter count strictly 124,439,808).

2. Checkpoint Fidelity:
   Logit and internal state numerical parity when loading official public weights
   (e.g., openai-community/gpt2 converted to basikGPT format).

3. Initialization Fidelity (This Document):
   GPT-2-compatible modernized initialization implementing standard normal distributions
   and depth-dependent residual projection scaling.

4. Training Recipe Modernization:
   Adoption of post-2019 PyTorch best practices (AdamW decoupled weight decay,
   beta_1=0.9/beta_2=0.95, parameter grouping, PyTorch SDPA, Cosine Annealing, FineWeb-Edu).
```

`basikGPT` does **not** claim byte-for-byte reproduction of the closed 2019 OpenAI training run, but rather provides a **GPT-2-compatible modernized initialization** that is mathematically verified.

---

## 2. Comparison Across Reference Implementations

| Parameter / Layer | OpenAI Released TF (`openai/gpt-2`) | GPT-2 Paper (Radford et al., 2019) | Hugging Face PyTorch (`transformers`) | basikGPT Modernized Policy |
|---|---|---|---|---|
| **Base Weight Initializer** | `tf.random_normal_initializer(stddev=0.02)` | Normal ($\sigma = 0.02$) | Normal ($\sigma = \text{initializer\_range} = 0.02$) | Normal ($\sigma = \text{initializer\_range} = 0.02$) |
| **Token Embeddings (`wte`)** | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) |
| **Positional Embeddings (`wpe`)** | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) | Normal ($\sigma = 0.02$) |
| **Linear Biases** | `tf.constant_initializer(0)` | Constant $0.0$ | Constant $0.0$ | Constant $0.0$ (`zeros_`) |
| **LayerNorm Scale ($g$) & Bias ($b$)** | $g=1.0, b=0.0$ | Constant $g=1.0, b=0.0$ | $g=1.0, b=0.0$ | $g=1.0, b=0.0$ (`ones_`, `zeros_`) |
| **Residual Projections (`out_proj`, `fc_out`)** | Scaled by $\frac{1}{\sqrt{2N}}$ ($N = \text{layers}$) | Scale by $\frac{1}{\sqrt{2N}}$ (Section 2.3) | $\sigma = \frac{\text{initializer\_range}}{\sqrt{2 \cdot n\_layer}}$ | $\sigma = \frac{\text{initializer\_range}}{\sqrt{2 \cdot n\_layers}}$ |
| **Weight Tying Execution** | Graph tensor sharing | Shared embedding & output weights | Executed via `self.tie_weights()` in `post_init()` | Executed via `self.tie_weights()` after init |

---

## 3. Theoretical Motivation & Residual Variance Control

In deep Pre-LayerNorm Transformer architectures (like GPT-2), signal variance naturally accumulates along the residual pathway:
$$x_{l} = x_{l-1} + f_l(\text{LayerNorm}(x_{l-1}))$$
If each sublayer $f_l$ adds outputs with variance $\sigma^2$, the variance of the residual stream after $M$ residual blocks would grow as $\sim M \sigma^2$. This variance accumulation can destabilize early training dynamics and cause exploding gradients.

To counteract this, OpenAI (Radford et al., 2019, Section 2.3) introduced a depth-dependent scaling factor:
> *"A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $\frac{1}{\sqrt{2N}}$ where $N$ is the number of residual layers."*

Because each Transformer block contains **2 residual additions** (one after Causal Multi-Head Self-Attention, one after the Position-Wise MLP), a stack of $L$ blocks contains $2L$ residual additions.

Canonical modernized residual projection **standard deviation** (not a linear `0.022 * n_layers` product):

$$\text{std}_{\text{resid}} = \frac{\text{initializer\_range}}{\sqrt{2 \cdot n\_layers}}$$

With the default `initializer_range = 0.02` and GPT-2 Small ($L = 12$):

$$\text{std}_{\text{resid}} = \frac{0.02}{\sqrt{2 \times 12}} = \frac{0.02}{\sqrt{24}} \approx 0.00408248$$

Positional embeddings (`wpe`) are **not** residual-scaled. They use the same base Normal as `wte`: $\mathcal{N}(0, 0.02^2)$.

---

## 4. Canonical Parameter Class Rules

| Parameter Class | Module / Attribute in basikGPT | Initialization Rule | GPT-2 Small ($L=12$) Target std |
|---|---|---|---|
| **Token Embedding** | `self.wte.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **Positional Embedding** | `self.wpe.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **QKV Projection** | `attn.qkv_proj.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **Attention Output Projection** | `attn.out_proj.weight` | $\mathcal{N}(0, \text{std}_{\text{resid}}^2)$ with $\text{std}_{\text{resid}} = \sigma / \sqrt{2L}$ | $\sigma / \sqrt{24} \approx 0.004082$ |
| **MLP Expansion** | `mlp.fc_in.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **MLP Contraction** | `mlp.fc_out.weight` | $\mathcal{N}(0, \text{std}_{\text{resid}}^2)$ with $\text{std}_{\text{resid}} = \sigma / \sqrt{2L}$ | $\sigma / \sqrt{24} \approx 0.004082$ |
| **Linear Biases** | `*.bias` | Constant $0$ (`zeros_`) | n/a (not sampled) |
| **LayerNorm Scales** | `ln_1`, `ln_2`, `ln_f` weight | Constant $1$ (`ones_`) | n/a (not sampled) |
| **LayerNorm Biases** | `ln_1`, `ln_2`, `ln_f` bias | Constant $0$ (`zeros_`) | n/a (not sampled) |
| **Language Model Head** | `self.lm_head.weight` | Tied to `self.wte.weight` | Identical memory tensor to `wte` |

---

## 5. Weight Tying and Initialization Order

In standard GPT-2, the output language model head (`lm_head.weight`) shares weights with the token embedding table (`wte.weight`).

To avoid double-sampling from the random number generator on the same memory buffer:
```python
# 1. Base initialization over all module primitives
self.apply(self._init_weights)

# 2. Depth-dependent residual projection scaling
residual_std = self.config.initializer_range / math.sqrt(2 * self.config.n_layers)
for pn, p in self.named_parameters():
    if pn.endswith("out_proj.weight") or pn.endswith("fc_out.weight"):
        torch.nn.init.normal_(p, mean=0.0, std=residual_std)

# 3. Explicit Weight Tying (lm_head.weight is wte.weight)
self.tie_weights()
```

### Invariants:
- `model.lm_head.weight is model.wte.weight` $\to$ `True`
- `id(model.lm_head.weight) == id(model.wte.weight)` $\to$ `True`
- `model.num_parameters()` $\to$ Exactly `124,439,808` for GPT-2 Small.

---

## 6. Theoretical Initial Loss Sanity

For a vocabulary of size $V = 50,257$, a uniformly random next-token prediction yields cross-entropy loss:
$$\mathcal{L}_{\text{uniform}} = -\ln\left(\frac{1}{V}\right) = \ln(V) = \ln(50,257) \approx 10.8249$$

With Gaussian initialization $\sigma = 0.02$, logits are small zero-centered values ($|z_i| \ll 1$), ensuring the initial training loss starts cleanly near $\approx 10.8 - 11.0$ without numerical instability, explosion, or NaN/Inf.
