# Weight Initialization & Variance Control Guide (basikGPT)

This document details the mathematical theory, reference origins, and explicit implementation of parameter initialization in **`basikGPT`** for training from scratch.

---

## 1. Theoretical Motivation & Reference Origins

In deep Pre-LayerNorm Transformer architectures (like GPT-2), signal variance naturally grows along the residual path:
$$x_{l} = x_{l-1} + f_l(\text{LayerNorm}(x_{l-1}))$$
If each sublayer $f_l$ adds outputs with variance $\sigma^2$, the variance of the residual stream after $M$ residual blocks would grow as $\sim M \sigma^2$. This variance accumulation can destabilize early training dynamics and cause exploding gradients.

To counteract this, OpenAI (Radford et al., 2019, Section 2.3) introduced a depth-dependent scaling factor:
> *"A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $\frac{1}{\sqrt{2N}}$ where $N$ is the number of residual layers."*

Because each Transformer block contains **2 residual additions** (one after Causal Multi-Head Self-Attention, one after the Position-Wise MLP), a stack of $L$ blocks contains $2L$ residual additions.

---

## 2. Canonical Initialization Specification

`basikGPT` parameterizes base initialization standard deviation via `GPTConfig.initializer_range = 0.02`.

### 2.1. Parameter Class Rules

| Parameter Class | Module / Attribute in basikGPT | Initialization Rule | GPT-2 Small ($L=12$) Target std |
|---|---|---|---|
| **Token Embedding** | `self.wte.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **Positional Embedding** | `self.wpe.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **QKV Projection** | `attn.qkv_proj.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **Attention Output Projection** | `attn.out_proj.weight` | $\mathcal{N}\left(0, \frac{\sigma^2}{2L}\right)$ | $\frac{0.02}{\sqrt{24}} \approx 0.004082$ |
| **MLP Expansion** | `mlp.fc_in.weight` | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.020000$ |
| **MLP Contraction** | `mlp.fc_out.weight` | $\mathcal{N}\left(0, \frac{\sigma^2}{2L}\right)$ | $\frac{0.02}{\sqrt{24}} \approx 0.004082$ |
| **Linear Biases** | `*.bias` | $\mathbf{0}$ (constant zero) | $0.000000$ |
| **LayerNorm Scales** | `ln_1`, `ln_2`, `ln_f` wt | $\mathbf{1}$ (constant one) | $0.000000$ |
| **LayerNorm Biases** | `ln_1`, `ln_2`, `ln_f` bias | $\mathbf{0}$ (constant zero) | $0.000000$ |
| **Language Model Head** | `self.lm_head.weight` | Tied to `self.wte.weight` | Identical memory tensor to `wte` |

---

## 3. Weight Tying and Initialization Order

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

## 4. Theoretical Initial Loss Sanity

For a vocabulary of size $V = 50,257$, a uniformly random next-token prediction yields cross-entropy loss:
$$\mathcal{L}_{\text{uniform}} = -\ln\left(\frac{1}{V}\right) = \ln(V) = \ln(50,257) \approx 10.8249$$

With Gaussian initialization $\sigma = 0.02$, logits are small zero-centered values ($|z_i| \ll 1$), ensuring the initial training loss starts cleanly near $\approx 10.8 - 11.0$ without numerical instability, explosion, or NaN/Inf.
