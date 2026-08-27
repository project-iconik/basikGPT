# Autoregressive Generation & Intrinsic Evaluation Guide (basikGPT)

This document describes the design, implementation, and mathematical formulations of the autoregressive text decoding, Key-Value Caching, and language-model evaluation engines in **`basikGPT`**.

---

## 1. Autoregressive Generation Pipeline

In autoregressive language modeling, token generation is modeled as a discrete sequential sampling process factorized by the chain rule of probability:

$$P(x_1, x_2, \dots, x_N) = \prod_{t=1}^N P(x_t \mid x_1, \dots, x_{t-1})$$

```text
Prompt String ────────► GPT2Tokenizer.encode ──────► input_ids (1, T)
                                                        │
┌────────────────── Autoregressive Loop ────────────────┤
│                                                       ▼
│                                               GPT.forward / forward_cached
│                                                       │
│                                                       ▼
│                                              logits (1, T, vocab_size)
│                                                       │
│                                                       ▼
│                                            next_logits = logits[:, -1, :]
│                                                       │
│                                                       ▼
│                                        sample_next_token(next_logits, config)
│                                        [Temperature -> Top-k -> Top-p -> Multinomial]
│                                                       │
│                                                       ▼
│                                                next_token (1, 1)
│                                                       │
│   input_ids = [input_ids, next_token] ◄───────────────┤ (Check EOT & Context Limit)
└───────────────────────────────────────────────────────┘
                                                        │
Generated Token IDs ─► GPT2Tokenizer.decode ─────► Generated Text String
```

### Why Only the Last Position Logits?
In causal decoder-only transformers with masked self-attention:
- Position $0$ predicts token $1$ given $x_0$
- Position $1$ predicts token $2$ given $x_{0:1}$
- $\dots$
- Position $T-1$ predicts token $T$ (the next unseen token) given the entire prefix $x_{0:T-1}$.

Therefore, only `logits[:, -1, :]` is extracted and forwarded to token sampling.

---

## 2. Key-Value (KV) Caching Fast Decoding (Milestone 10)

In naive autoregressive decoding, generating $N$ tokens requires re-evaluating the accumulating prefix at every step, causing $O(N^2)$ cumulative linear projections and self-attention operations.

With **KV Caching**, past Key ($K$) and Value ($V$) projections are saved per layer and reused across decoding steps:

```text
Step 0 (Prompt Prefill, T tokens):
  input_ids: (B, T)
  Q, K, V: (B, H, T, D)
  Cache = [(K_0, V_0), (K_1, V_1), ..., (K_{L-1}, V_{L-1})]

Step 1 (Single Token Decode, 1 token):
  next_token: (B, 1)
  Q_new, K_new, V_new: (B, H, 1, D)
  K_updated = concat([K_cached, K_new], dim=-2) -> (B, H, T+1, D)
  V_updated = concat([V_cached, V_new], dim=-2) -> (B, H, T+1, D)
  Attention = Q_new @ K_updated^T @ V_updated   -> (B, H, 1, D)
```

### 2.1. Why Queries are Never Cached
In causal self-attention, token generation at step $t$ requires computing attention between the **current query $Q_t$** and **all past keys $K_{0:t}$**. Past queries $Q_{0:t-1}$ are never referenced in future decoding steps and are discarded immediately.

### 2.2. Position Offset in Learned Positional Embeddings
GPT-2 uses learned absolute positional embeddings (`wpe`) indexed $[0, 1, \dots, \text{context\_length}-1]$. When appending token $x_t$ to an existing cache of length $T_{\text{past}}$, its position index is strictly:
$$\text{pos} = T_{\text{past}}$$
Failing to offset position indices causes catastrophic output degradation by resetting the learned positional embedding to 0.

### 2.3. Causal Masking: Prefill vs. Decode
- **Prefill Phase ($T_q > 1$)**: Standard lower-triangular causal masking is enforced ($j \le i$) because queries cannot attend to future prompt tokens.
- **Decode Phase ($T_q = 1, T_k = T_{\text{past}} + 1$)**: The single query at position $T_{\text{past}}$ is allowed to attend to **all** keys $0 \dots T_{\text{past}}$. Thus, **no causal masking is required**, allowing unmasked SDPA (`is_causal=False`) and eager attention.

### 2.4. KV Cache Memory Footprint Accounting
The theoretical memory consumed by the KV Cache across all $L$ layers is:
$$\text{Memory}_{\text{KV}} = 2 \times L \times B \times H \times T \times D \times \text{sizeof(dtype)}$$

For canonical **GPT-2 Small** ($L=12, H=12, D=64$) at maximum context length $T=1024$ and batch size $B=1$:
$$\text{Total Elements} = 2 \times 12 \times 1 \times 12 \times 1024 \times 64 = 18,874,368 \text{ elements}$$
- In **FP32** (4 bytes/element): $18,874,368 \times 4 = 75,497,472\text{ bytes} \approx 72.0\text{ MB}$
- In **BF16/FP16** (2 bytes/element): $18,874,368 \times 2 = 37,748,736\text{ bytes} \approx 36.0\text{ MB}$

### 2.5. Computational Complexity Clarification
KV Caching eliminates the repeated linear projection $O(N \cdot d_{\text{model}}^2)$ for previous tokens. However, the attention dot-product $Q_{\text{new}} \times K_{\text{cached}}^T$ at step $t$ must still scan all $t$ cached keys. Thus, per-step attention cost grows linearly with sequence length ($O(t)$), reducing total decoding work from quadratic $O(N^2 \cdot d_{\text{model}} + N^3)$ to $O(N \cdot d_{\text{model}} + N^2)$.

---

## 3. Sampling & Filtering Mechanics

`basikGPT` supports both deterministic (greedy argmax) and stochastic sampling controlled by `GenerationConfig`.

### 3.1. Greedy Argmax Decoding (`do_sample=False`)
Selects the token with the highest predicted unnormalized logit:
$$x_{\text{next}} = \arg\max_{v \in \mathcal{V}} z_v$$

### 3.2. Temperature Scaling ($T$)
Adjusts distribution sharpness prior to probability normalization:
$$z'_v = \frac{z_v}{T}$$
- $T < 1.0$: Sharper distribution (more deterministic, higher probability on modes).
- $T = 1.0$: Original raw distribution.
- $T > 1.0$: Flatter distribution (higher entropy, more diversity).

### 3.3. Top-$k$ Filtering
Restricts the sampling vocabulary $\mathcal{V}$ to the top $k$ candidates $\mathcal{V}_k \subset \mathcal{V}$:
$$z''_v = \begin{cases} z'_v & \text{if } v \in \mathcal{V}_k \\ -\infty & \text{otherwise} \end{cases}$$

### 3.4. Top-$p$ (Nucleus) Filtering
Selects the smallest subset $\mathcal{V}_p$ such that the cumulative probability mass exceeds $p$:
$$\sum_{v \in \mathcal{V}_p} P(v) \ge p$$
Tokens outside $\mathcal{V}_p$ are assigned logits of $-\infty$. The highest-probability token is always preserved to avoid empty sets.

### 3.5. Combined Execution Order
$$\text{Logits} \xrightarrow{\text{Temperature}} z' \xrightarrow{\text{Top-}k} z'' \xrightarrow{\text{Top-}p} z''' \xrightarrow{\text{Softmax}} P(v) \xrightarrow{\text{torch.multinomial}} x_{\text{next}}$$

---

## 4. Context Length Policy (Learned Positional Embeddings)

GPT-2 employs learned absolute positional embeddings (`wpe`) bounded by `context_length = 1024`.
1. **Prompt Guardrail**: If $\text{len}(\text{prompt}) > \text{context\_length}$, generation fails fast with `ValueError`.
2. **Effective Budget**: The maximum new tokens generated is capped by available context:
   $$\text{effective\_max\_new\_tokens} = \min(\text{max\_new\_tokens}, \text{context\_length} - \text{prompt\_length})$$
3. **Boundary Condition**: If $\text{prompt\_length} == \text{context\_length}$, generation terminates immediately without error, returning the original prompt.

---

## 5. Intrinsic Language Model Evaluation & Perplexity

### Token-Weighted Cross-Entropy Loss
$$\mathcal{L} = \frac{\sum_{b=1}^B \sum_{i=1}^{T_b} \text{NLL}_{b, i}}{\sum_{b=1}^B T_b}$$

### Perplexity ($PPL$)
Perplexity measures the effective branching factor of the language model:
$$\text{PPL} = \exp(\mathcal{L})$$
- A perfect model achieves $\text{PPL} = 1.0$.
- A uniform random guess across vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 50,257$) yields:
  $$\text{PPL}_{\text{random}} = \exp(-\ln(1 / 50257)) = 50,257$$

---

## 6. CLI Execution Reference

### 6.1. Generate Text with KV Cache Enabled (Fast Default)
```bash
python scripts/generate.py \
    --hf-reference \
    --prompt "The history of artificial intelligence" \
    --max-new-tokens 30 \
    --use-cache \
    --device cpu
```

### 6.2. Generate Text with Naive Decoding (Reference Verification)
```bash
python scripts/generate.py \
    --hf-reference \
    --prompt "The history of artificial intelligence" \
    --max-new-tokens 30 \
    --no-use-cache \
    --device cpu
```

### 6.3. Generate Text with Stochastic Sampling
```bash
python scripts/generate.py \
    --hf-reference \
    --prompt "The history of artificial intelligence" \
    --max-new-tokens 30 \
    --do-sample \
    --temperature 0.8 \
    --top-k 50 \
    --seed 1337 \
    --device cpu
```

### 6.4. Evaluate Checkpoint on Validation Shards
```bash
python scripts/evaluate.py \
    --checkpoint runs/m8-local-smoke/step-final.pt \
    --data-dir data/fineweb-edu-smoke \
    --device cpu
```
