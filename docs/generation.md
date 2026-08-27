# Autoregressive Generation & Intrinsic Evaluation Guide (basikGPT)

This document describes the design, implementation, and mathematical formulations of the autoregressive text decoding and language-model evaluation engines in **`basikGPT`**.

---

## 1. Autoregressive Generation Pipeline

In autoregressive language modeling, token generation is modeled as a discrete sequential sampling process factorized by the chain rule of probability:

$$P(x_1, x_2, \dots, x_N) = \prod_{t=1}^N P(x_t \mid x_1, \dots, x_{t-1})$$

```text
Prompt String ────────► GPT2Tokenizer.encode ──────► input_ids (1, T)
                                                        │
┌────────────────── Autoregressive Loop ────────────────┤
│                                                       ▼
│                                               GPT.forward(input_ids)
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

## 2. Sampling & Filtering Mechanics

`basikGPT` supports both deterministic (greedy argmax) and stochastic sampling controlled by `GenerationConfig`.

### 2.1. Greedy Argmax Decoding (`do_sample=False`)
Selects the token with the highest predicted unnormalized logit:
$$x_{\text{next}} = \arg\max_{v \in \mathcal{V}} z_v$$

### 2.2. Temperature Scaling ($T$)
Adjusts distribution sharpness prior to probability normalization:
$$z'_v = \frac{z_v}{T}$$
- $T < 1.0$: Sharper distribution (more deterministic, higher probability on modes).
- $T = 1.0$: Original raw distribution.
- $T > 1.0$: Flatter distribution (higher entropy, more diversity).

### 2.3. Top-$k$ Filtering
Restricts the sampling vocabulary $\mathcal{V}$ to the top $k$ candidates $\mathcal{V}_k \subset \mathcal{V}$:
$$z''_v = \begin{cases} z'_v & \text{if } v \in \mathcal{V}_k \\ -\infty & \text{otherwise} \end{cases}$$

### 2.4. Top-$p$ (Nucleus) Filtering
Selects the smallest subset $\mathcal{V}_p$ such that the cumulative probability mass exceeds $p$:
$$\sum_{v \in \mathcal{V}_p} P(v) \ge p$$
Tokens outside $\mathcal{V}_p$ are assigned logits of $-\infty$. The highest-probability token is always preserved to avoid empty sets.

### 2.5. Combined Execution Order
$$\text{Logits} \xrightarrow{\text{Temperature}} z' \xrightarrow{\text{Top-}k} z'' \xrightarrow{\text{Top-}p} z''' \xrightarrow{\text{Softmax}} P(v) \xrightarrow{\text{torch.multinomial}} x_{\text{next}}$$

---

## 3. Context Length Policy (Learned Positional Embeddings)

GPT-2 employs learned absolute positional embeddings (`wpe`) bounded by `context_length = 1024`.
1. **Prompt Guardrail**: If $\text{len}(\text{prompt}) > \text{context\_length}$, generation fails fast with `ValueError`.
2. **Effective Budget**: The maximum new tokens generated is capped by available context:
   $$\text{effective\_max\_new\_tokens} = \min(\text{max\_new\_tokens}, \text{context\_length} - \text{prompt\_length})$$
3. **Boundary Condition**: If $\text{prompt\_length} == \text{context\_length}$, generation terminates immediately without error, returning the original prompt.

---

## 4. Why KV Cache is Omitted in Milestone 9

In this baseline milestone, each generated token triggers a full forward pass over the accumulating prefix ($O(N^2)$ aggregate attention complexity over sequence length $N$):
$$\sum_{t=1}^N t = \frac{N(N+1)}{2}$$

### Design Rationale
- **Simplicity & Readability**: Naive full-prefix forward provides an unambiguous, transparent reference implementation.
- **Parity Ground Truth**: Serves as the mathematical baseline against which future KV Cache implementations (Milestone 10) will be validated for bitwise equivalence.

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

### 6.1. Generate Text from Reference GPT-2 (Greedy)
```bash
python scripts/generate.py \
    --hf-reference \
    --prompt "The history of artificial intelligence" \
    --max-new-tokens 30 \
    --device cpu
```

### 6.2. Generate Text with Sampling
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

### 6.3. Evaluate Checkpoint on Validation Shards
```bash
python scripts/evaluate.py \
    --checkpoint runs/m8-local-smoke/step-final.pt \
    --data-dir data/fineweb-edu-smoke \
    --device cpu
```
