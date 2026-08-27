# Tensor Shape Conventions

This document establishes the canonical tensor shape notation and transformation rules for the `basikGPT` codebase.

Every neural-network module and tensor-manipulating function in `basikGPT` MUST document input and output tensor shapes using these exact conventions.

---

## 1. Primary Dimension Symbols

| Symbol | Name | Description | GPT-2 Small Canonical Value |
|:---:|:---|:---|:---:|
| **`B`** | Batch Size | Number of independent sequences in a forward batch | Dynamic (e.g., 8, 16, 64) |
| **`T`** | Sequence Length | Context / time steps in sequence | $\le 1024$ |
| **`C`** | Model Dimension | Embedding / hidden representation dimension (`d_model`) | `768` |
| **`H`** | Number of Heads | Number of parallel self-attention heads (`n_heads`) | `12` |
| **`D`** | Head Dimension | Dimension per attention head (`head_dim = C // H`) | `64` |
| **`V`** | Vocabulary Size | Total number of discrete tokens in tokenizer (`vocab_size`) | `50,257` |
| **`d_ff`** | Feed-Forward Dim | Hidden dimension in the MLP / feed-forward network | `3,072` ($4 \times C$) |

---

## 2. Component-by-Component Tensor Flows

### 2.1. Embeddings

```text
Input token IDs:
    idx: (B, T) [dtype: torch.long]

Token Embedding Lookup:
    tok_emb = wte(idx): (B, T, C)

Positional Embedding Lookup:
    pos = torch.arange(0, T, device=device): (T,)
    pos_emb = wpe(pos): (T, C)

Combined Input Representation:
    x = tok_emb + pos_emb: (B, T, C)
```

### 2.2. Pre-Norm Transformer Block

For a sequence of $L$ layers:

```text
x: (B, T, C)
  │
  ├──► LayerNorm 1 ──► (B, T, C)
  │         │
  │    Causal Multi-Head Attention ──► (B, T, C)
  │         │
  ▼         ▼
  + ◄───────┘ (Residual Connection 1: x = x + attn_out)
  │
  ├──► LayerNorm 2 ──► (B, T, C)
  │         │
  │        MLP (c_fc -> GELU -> c_proj) ──► (B, T, C)
  │         │
  ▼         ▼
  + ◄───────┘ (Residual Connection 2: x = x + mlp_out)
  │
  ▼
Output: (B, T, C)
```

### 2.3. Causal Multi-Head Self-Attention

Detailed tensor shape progression inside attention:

```text
1. Input tensor:
   x: (B, T, C)

2. Linear QKV projection (fused 3*C output):
   qkv = c_attn(x): (B, T, 3 * C)

3. Split into Q, K, V:
   q, k, v: each (B, T, C)

4. Unfold heads and transpose for parallel head computation:
   q: (B, T, H, D) -> (B, H, T, D)
   k: (B, T, H, D) -> (B, H, T, D)
   v: (B, T, H, D) -> (B, H, T, D)

5. Attention score matrix (Q @ K^T):
   scores = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(D)): (B, H, T, T)

6. Apply causal upper-triangular mask (mask future tokens to -inf):
   masked_scores: (B, H, T, T)

7. Softmax & Dropout:
   attn_weights = softmax(masked_scores, dim=-1): (B, H, T, T)
   attn_weights = dropout(attn_weights): (B, H, T, T)

8. Weighted Value Aggregation (attn_weights @ V):
   out = attn_weights @ v: (B, H, T, D)

9. Head Merge & Transpose back:
   out = out.transpose(1, 2).contiguous().view(B, T, C): (B, T, C)

10. Output Projection & Residual Dropout:
    out = out_proj(out): (B, T, C)
    out = resid_dropout(out): (B, T, C)
```

> **Dual Backend Note**:
> - **`eager` Backend (Educational Reference)**: Executes steps 5–8 explicitly via manual matrix multiplication, masking, softmax, and dropout.
> - **`sdpa` Backend (Optimized Pretraining Path)**: Replaces steps 5–8 with PyTorch's `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, while preserving identical tensor shapes and parameters across projections.

### 2.4. Feed-Forward Network (MLP)

```text
1. Input tensor:
   x: (B, T, C)

2. Expansion linear projection:
   h = c_fc(x): (B, T, d_ff)   # d_ff = 4 * C = 3072

3. Non-linear activation:
   h = gelu(h): (B, T, d_ff)   # GPT-2 tanh-approximation GELU

4. Compression linear projection:
   out = c_proj(h): (B, T, C)

5. Dropout:
   out = dropout(out): (B, T, C)
```

### 2.5. Final LayerNorm & Language Model Head

```text
1. Block stack output:
   x: (B, T, C)

2. Final LayerNorm:
   x = ln_f(x): (B, T, C)

3. Language Model Head (linear projection tied to wte.weight):
   logits = lm_head(x): (B, T, V)  # V = 50257
```

---

## 3. Mathematical Causality Invariant

For any sequence of length $T$, the causal attention mask ensures that for each position $i \in [0, T-1]$, the output token representation $x_i$ depends strictly on $\{x_0, x_1, \dots, x_i\}$ and is mathematically independent of $\{x_{i+1}, \dots, x_{T-1}\}$.

$$\text{Attention}(Q, K, V)_{i, j} = 0 \quad \forall j > i$$

This invariant will be tested directly in unit tests in subsequent milestones.

---

## 4. Parameter & Naming Mapping (OpenAI / HuggingFace $\iff$ basikGPT)

This reference mapping defines how OpenAI / HuggingFace `GPT2LMHeadModel` state dict keys and weight matrix layouts map to `basikGPT`.

### 4.1. Weight Layout Conventions
- **HuggingFace `Conv1D`**: Uses OpenAI's 1D convolutional linear layer format where 2D weight matrices are stored with shape `(in_features, out_features)`.
- **PyTorch `nn.Linear` (basikGPT)**: Follows standard PyTorch convention where weight matrices are stored with shape `(out_features, in_features)`.
- **Transformation Rule**: Every 2D projection weight matrix from HuggingFace MUST be transposed (`.t()`) when importing into `basikGPT`.

### 4.2. Complete State Dict Mapping Table

| Component | HuggingFace / OpenAI Key | HF Shape | basikGPT Parameter Key | basikGPT Shape | Transpose Required? |
|---|---|---|---|---|:---:|
| **Token Embedding** | `transformer.wte.weight` | `(50257, 768)` | `wte.weight` | `(50257, 768)` | No |
| **Positional Embedding** | `transformer.wpe.weight` | `(1024, 768)` | `wpe.weight` | `(1024, 768)` | No |
| **Block $l$ LayerNorm 1** | `transformer.h.{l}.ln_1.weight` | `(768,)` | `blocks.{l}.ln_1.weight` | `(768,)` | No |
| | `transformer.h.{l}.ln_1.bias` | `(768,)` | `blocks.{l}.ln_1.bias` | `(768,)` | No |
| **Block $l$ Attention QKV** | `transformer.h.{l}.attn.c_attn.weight` | `(768, 2304)` | `blocks.{l}.attn.qkv_proj.weight` | `(2304, 768)` | **Yes (`.t()`)** |
| | `transformer.h.{l}.attn.c_attn.bias` | `(2304,)` | `blocks.{l}.attn.qkv_proj.bias` | `(2304,)` | No |
| **Block $l$ Attention Out** | `transformer.h.{l}.attn.c_proj.weight` | `(768, 768)` | `blocks.{l}.attn.out_proj.weight` | `(768, 768)` | **Yes (`.t()`)** |
| | `transformer.h.{l}.attn.c_proj.bias` | `(768,)` | `blocks.{l}.attn.out_proj.bias` | `(768,)` | No |
| **Block $l$ LayerNorm 2** | `transformer.h.{l}.ln_2.weight` | `(768,)` | `blocks.{l}.ln_2.weight` | `(768,)` | No |
| | `transformer.h.{l}.ln_2.bias` | `(768,)` | `blocks.{l}.ln_2.bias` | `(768,)` | No |
| **Block $l$ MLP Expansion** | `transformer.h.{l}.mlp.c_fc.weight` | `(768, 3072)` | `blocks.{l}.mlp.fc_in.weight` | `(3072, 768)` | **Yes (`.t()`)** |
| | `transformer.h.{l}.mlp.c_fc.bias` | `(3072,)` | `blocks.{l}.mlp.fc_in.bias` | `(3072,)` | No |
| **Block $l$ MLP Contraction** | `transformer.h.{l}.mlp.c_proj.weight` | `(3072, 768)` | `blocks.{l}.mlp.fc_out.weight` | `(768, 3072)` | **Yes (`.t()`)** |
| | `transformer.h.{l}.mlp.c_proj.bias` | `(768,)` | `blocks.{l}.mlp.fc_out.bias` | `(768,)` | No |
| **Final LayerNorm** | `transformer.ln_f.weight` | `(768,)` | `ln_f.weight` | `(768,)` | No |
| | `transformer.ln_f.bias` | `(768,)` | `ln_f.bias` | `(768,)` | No |
| **LM Head (Tied)** | `lm_head.weight` | `(50257, 768)` | `lm_head.weight` (tied to `wte`) | `(50257, 768)` | Tied |

