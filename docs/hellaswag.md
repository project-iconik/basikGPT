# Zero-Shot HellaSwag Benchmark Evaluation Guide (basikGPT)

This document describes the design, implementation, and mathematical formulations of the zero-shot downstream multiple-choice **HellaSwag** evaluation engine in **`basikGPT`**.

---

## 1. Benchmark Overview

**HellaSwag** ([Zellers et al., 2019](https://arxiv.org/abs/1905.07830)) is a multiple-choice commonsense reasoning dataset designed to evaluate how plausibly a language model finishes an unfolding story or activity.

Each example presents:
1. An opening **Context** (derived from ActivityNet or WikiHow).
2. Four alternative **Candidate Endings** ($e_0, e_1, e_2, e_3$).
3. A single **Gold Label** $y \in \{0, 1, 2, 3\}$.

Unlike generative tasks, HellaSwag evaluation does **not** generate free-form text. Instead, it computes the exact conditional log-probability:
$$P(\text{completion}_j \mid \text{context})$$
for each of the 4 candidate choices and selects the candidate with the highest likelihood.

```text
Context (T_prompt tokens)
    ├── Candidate 0 ──► model forward ──► Score S_0
    ├── Candidate 1 ──► model forward ──► Score S_1
    ├── Candidate 2 ──► model forward ──► Score S_2
    └── Candidate 3 ──► model forward ──► Score S_3
                              │
                              ▼
                  Predicted Index = argmax(S_j)
                              │
                              ▼
               Compare with Gold Label y (0, 1, 2, 3)
```

---

## 2. Dataset Split Selection & Schema

### 2.1. Why the Validation Split is Canonical
- **Hugging Face Repository**: `Rowan/hellaswag`
- **`train`**: 39,905 examples.
- **`validation`**: **10,042 examples** with complete ground truth gold labels (`label: "0".."3"`).
- **`test`**: 10,003 examples where `label: ""` (hidden for blind server leaderboard submission).

Because the `test` split lacks labels, zero-shot academic benchmarking is standardly performed on the **`validation` split**.

### 2.2. Example Data Structure
```python
{
    "ind": 24,
    "activity_label": "Roof shingle removal",
    "ctx_a": "A man is sitting on a roof.",
    "ctx_b": "he",
    "ctx": "A man is sitting on a roof. he",
    "endings": [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof."
    ],
    "source_id": "activitynet~v_-JhWjGDPHMY",
    "split": "val",
    "split_type": "indomain",
    "label": "3"
}
```

---

## 3. Benchmark Context Formatting

`basikGPT` supports two explicit prompt formatting styles:

### 3.1. Standard Format (`activity_ctx`, Default)
Following the standard convention in `lm-evaluation-harness` and the original paper:
$$\text{Prompt} = \text{activity\_label} + \text{": "} + \text{ctx\_a} + \text{" "} + \text{ctx\_b.capitalize()}$$
*(If `ctx_b` is empty, `activity_label + ": " + ctx_a`)*

### 3.2. Raw Context Format (`ctx_only`)
Following the convention used in `nanoGPT` / `build-nanogpt`:
$$\text{Prompt} = \text{ctx}$$

---

## 4. GPT-2 Token Boundary Invariant

In Byte-Pair Encoding (BPE), tokenization of `"A"` + `"B"` can differ from `"A B"`.

In HellaSwag:
- Endings never begin with a leading space.
- Contexts never end with a trailing space.
- By prepending `" "` to `ending`, we establish the exact invariant:
  $$\text{encode}(\text{context}) + \text{encode}(\text{" "} + \text{ending}) == \text{encode}(\text{context} + \text{" "} + \text{ending})$$
  *(Verified with 0 discrepancies across all 40,168 validation sequences).*

---

## 5. Conditional Log-Likelihood & Shift Alignment

For a sequence of $T$ tokens $X = [x_0, \dots, x_{P-1}, x_P, \dots, x_{T-1}]$, where $x_{0:P-1}$ is the context ($P$ tokens) and $x_{P:T-1}$ is the completion ($M = T - P$ tokens):

In causal language models, logit $Z[t]$ parameterizes $P(x_{t+1} \mid x_{\le t})$.

```text
Tokens:  [ x_0,   x_1,  ..., x_{P-1}, | x_P,  x_{P+1}, ..., x_{T-1} ]
Logits:  [ Z_0,   Z_1,  ..., Z_{P-1}, | Z_P,  Z_{P+1}, ..., Z_{T-2} ]
           │       │            │         │      │             │
Predicts: [ x_1,   x_2,  ...,   x_P,  |  x_{P+1},x_{P+2},..., x_{T-1} ]
```

### Slice Alignment
- `shift_logits = logits[:, P-1 : T-1, :]` (shape: $(1, M, V)$)
- `shift_targets = tokens[:, P : T]` (shape: $(1, M)$)

### Token Log-Probability
$$\log P(x_t \mid x_{<t}) = \text{log\_softmax}(Z[t-1])[x_t]$$

---

## 6. Scoring Metrics: Raw vs. Length-Normalized

### 6.1. Raw Total Log-Likelihood ($S_{\text{raw}}$)
$$S_{\text{raw}} = \sum_{t=P}^{T-1} \log P(x_t \mid x_{<t})$$
$$\text{pred}_{\text{raw}} = \arg\max_{j \in \{0,1,2,3\}} S_{\text{raw}}^{(j)}$$

### 6.2. Length-Normalized Log-Likelihood ($S_{\text{norm}}$)
Because raw likelihood accumulates negative log-probabilities, longer completions are inherently penalized. The length-normalized score measures the average log-probability per completion token:
$$S_{\text{norm}} = \frac{1}{M} \sum_{t=P}^{T-1} \log P(x_t \mid x_{<t}) = \frac{S_{\text{raw}}}{M}$$
$$\text{pred}_{\text{norm}} = \arg\max_{j \in \{0,1,2,3\}} S_{\text{norm}}^{(j)}$$

`acc_norm` (using $S_{\text{norm}}$) is the standard canonical metric reported across LLM literature and leaderboards.

---

## 7. Context Overflow & Truncation Policy

- The maximum sequence length across all 10,042 validation examples is **166 tokens**, well within GPT-2's 1024 context window.
- **Guardrail Policy**: If a sequence exceeds 1024 tokens:
  1. Completion tokens $M$ are **strictly preserved**.
  2. Context tokens are left-truncated to $1024 - M$ tokens:
     $$\text{context\_tokens} = \text{context\_tokens}[-(1024 - M):]$$

---

## 8. CLI Reference

### 8.1. Evaluate Reference GPT-2 on Validation Split (Smoke 20 Examples)
```bash
python scripts/evaluate_hellaswag.py \
    --hf-reference \
    --device cpu \
    --max-examples 20 \
    --output-json runs/hellaswag-smoke.json
```

### 8.2. Evaluate Trained basikGPT Checkpoint
```bash
python scripts/evaluate_hellaswag.py \
    --checkpoint runs/m8-local-smoke/step-final.pt \
    --device cpu \
    --max-examples 50 \
    --output-json runs/eval-hellaswag.json \
    --output-jsonl runs/eval-hellaswag-records.jsonl
```

### 8.3. Evaluate Full Validation Set (10,042 Examples)
```bash
python scripts/evaluate_hellaswag.py \
    --hf-reference \
    --device cpu \
    --split validation
```

---

## 9. Baseline Calibration & Expected Accuracy

| Model | Parameters | Training Tokens | Zero-Shot HellaSwag Acc (`acc_norm`) |
|---|---|---|---|
| Random Guess | - | - | 25.00% |
| **GPT-2 Small (Reference 124M)** | 124M | ~40B | **~29.4% – 31.0%** |
| GPT-3 Small | 124M | 300B | ~33.7% |
| GPT-2 XL | 1.5B | ~40B | ~40.0% |

> [!NOTE]
> **Calibrated Statement**:
> HellaSwag zero-shot evaluation measures relative log-likelihood over continuation candidates. It demonstrates that the architecture, tokenization, and causal likelihood extraction are correctly aligned, rather than asserting general artificial intelligence.
