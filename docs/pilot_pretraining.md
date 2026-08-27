# Local Pilot Pretraining Protocol & Token Accounting Guide (basikGPT)

This document establishes the official **Pilot Pretraining Protocol**, canonical **Token Accounting Formulations**, numerical health guardrails, and readiness verification procedures for **`basikGPT`** before scaling up to large-scale GPU pretraining.

---

## 1. Executive Purpose

Large-scale autoregressive Transformer pretraining is computationally demanding. Before launching long-running pretraining campaigns (e.g. Chinchilla-inspired ~2.5B tokens), all orchestration mechanics, token accounting arithmetic, learning rate schedules, gradient dynamics, validation isolation, and failure detection systems must be verified deterministically on small-scale pilot runs.

> **Pilot Scope Statement**:
> Local CPU pilots validate **correctness, numerical health monitoring, gradient accumulation mechanics, checkpoint/resume continuity, and logging schemas**.
> They do not validate GPU-specific hardware metrics (VRAM ceiling, kernel throughput, or CUDA/BF16 stability).

---

## 2. Canonical Token Accounting Formulations

Pretraining accounting is formulated through rigorous arithmetic relations:

### 2.1. Basic Variables
- $B$: Micro-batch size per forward/backward pass.
- $T$: Context sequence length (block size).
- $G$: Gradient accumulation steps per optimizer update.
- $W$: World size (number of parallel distributed devices; $W=1$ for single-device CPU).

### 2.2. Mathematical Formulations
1. **Tokens per Micro-Batch**:
   $$\text{tokens\_per\_micro\_batch} = B \times T$$
2. **Tokens per Optimizer Step**:
   $$\text{tokens\_per\_optimizer\_step} = B \times T \times G \times W$$
3. **Step Calculation (Ceiling Policy)**:
   $$\text{optimizer\_steps} = \left\lceil \frac{\text{target\_token\_budget}}{\text{tokens\_per\_optimizer\_step}} \right\rceil$$
4. **Actual Processed Token Budget**:
   $$\text{actual\_token\_budget} = \text{optimizer\_steps} \times \text{tokens\_per\_optimizer\_step}$$
5. **Budget Overshoot**:
   $$\text{overshoot\_tokens} = \text{actual\_token\_budget} - \text{requested\_token\_budget} \ge 0$$

### 2.3. Canonical Token Class Distinctions
- **Input Tokens Processed**: $B \times T$ token IDs passed into the Transformer embeddings.
- **Target Tokens Contributing to Loss**: $B \times T$ next-token prediction targets evaluated in cross-entropy loss.
- **Nominal Training Tokens**: $B \times T \times G \times W$ tokens committed per optimizer parameter update.
- **Validation Tokens**: Tokens evaluated in `evaluate()`. **Invariant**: Validation tokens are strictly excluded from the `tokens_seen` training counter.

---

## 3. Pilot Pretraining Stages

`basikGPT` defines three standardized pilot stages:

| Stage | Target Budget | Typical Config ($B, T, G, W$) | Tokens / Step | Steps | Actual Tokens | Overshoot | Primary Verification Objective |
|---|---|---|---|---|---|---|---|
| **Stage A (Smoke)** | $10,000$ | $2, 64, 2, 1$ | $256$ | $40$ | $10,240$ | $+240$ | Pipeline mechanics, finite loss/grad, LR warmup, checkpointing, validation execution. |
| **Stage B (Short Pilot)** | $100,000$ | $4, 128, 2, 1$ | $1,024$ | $98$ | $100,352$ | $+352$ | Convergence trajectory, gradient norm stability, validation cadence, resume continuity. |
| **Stage C (Extended)** | $1,000,000$ | $4, 256, 4, 1$ | $4,096$ | $245$ | $1,003,520$ | $+3,520$ | Multi-thousand step regime, schedule boundary compliance, extended throughput tracking. |

---

## 4. Training Health Monitoring & Failure Guardrails

Pilot runs enforce fail-fast guardrails to abort immediately on numerical divergence:

1. **Finite Loss Guardrail**:
   If $\text{loss} \in \{\text{NaN}, +\infty, -\infty\}$ is encountered during training or evaluation, `FloatingPointError` is raised immediately.
2. **Finite Gradient Guardrail**:
   If the total unscaled gradient norm $\|\mathbf{g}\|_2 \in \{\text{NaN}, +\infty\}$ after `clip_grad_norm_`, `FloatingPointError` is raised immediately.
3. **Pre-Clipping Total Gradient Norm**:
   Gradient norms recorded in `metrics.jsonl` represent the true pre-clipping total Euclidean norm $\|\mathbf{g}\|_2 = \sqrt{\sum_i \|\mathbf{g}_i\|_2^2}$.
4. **Learning Rate Alignment**:
   Scheduler advances strictly once per global optimizer step (after all $G$ micro-batches), maintaining exact warmup and cosine decay boundaries.

---

## 5. Checkpoint & Resume Continuity Audit

### 5.1. Restored State Components
- Model parameter weights (`state_dict`).
- Optimizer first ($m_t$) and second ($v_t$) momentum buffers and step counters.
- Mixed precision GradScaler state (if FP16).
- Global optimizer step (`global_step`) and cumulative `tokens_seen`.
- Python `random`, PyTorch CPU, and PyTorch CUDA RNG state vectors.

### 5.2. Data Stream Continuity Classification
- **Classification**: *"State-continuous (model, optimizer, scheduler, RNG restored), but dataset iterator resets to shard start if not fast-forwarded."*
- `DataLoader` streams initialize from the beginning of shard files upon resumption. In full-scale distributed pretraining, shard offset tracking or deterministic index positioning is used.

---

## 6. Long-Term Chinchilla-Inspired Pretraining Planning (~2.5B Tokens)

Analytical planning for GPT-2 Small ($124\text{M parameters}$, context $T=1,024$) targeting $\approx 2.5\text{B tokens}$ ($\approx 20\text{ tokens/param}$):

| Distributed Setup | Micro-Batch ($B$) | Context ($T$) | Grad Accum ($G$) | World Size ($W$) | Tokens / Step | Optimizer Steps | Actual Tokens | Overshoot |
|---|---|---|---|---|---|---|---|---|
| **Single-Device (W=1)** | 4 | 1,024 | 8 | 1 | 32,768 | 76,294 | 2,500,001,792 | +1,792 |
| **2x GPU DDP (W=2)** | 4 | 1,024 | 8 | 2 | 65,536 | 38,147 | 2,500,001,792 | +1,792 |
| **4x GPU DDP (W=4)** | 4 | 1,024 | 8 | 4 | 131,072 | 19,074 | 2,500,067,328 | +67,328 |
| **8x GPU DDP (W=8)** | 4 | 1,024 | 8 | 8 | 262,144 | 9,537 | 2,500,067,328 | +67,328 |

---

## 7. Main-Run Readiness Checklist

Before transitioning to GPU qualification and main pretraining:

### 7.1. Validated on Local CPU (Milestone 13)
- [x] **Analytical Token Accounting**: Exact $B \times T \times G \times W$ step and token calculations.
- [x] **LR Scheduler Alignment**: Warmup and cosine decay synchronized strictly with optimizer steps.
- [x] **Finite Loss & Gradient Guardrails**: Fail-fast on NaN/Inf with `FloatingPointError`.
- [x] **Validation Isolation**: Validation tokens strictly excluded from `tokens_seen`.
- [x] **Checkpoint Serialization**: Atomic save/load of model, optimizer, scheduler, and RNG states.
- [x] **Resume Continuity**: Verifiable state resumption without step regression.
- [x] **Structured Summary Logging**: Complete machine-readable `pilot_summary.json` output.

### 7.2. GPU Qualification Required (Upcoming Milestone 14)
- [ ] **CUDA Device Execution**: `NOT VALIDATED` (requires NVIDIA GPU environment).
- [ ] **BF16 Mixed Precision Stability**: `NOT VALIDATED` (Ampere+ native hardware required).
- [ ] **PyTorch SDPA Hardware Kernels**: `NOT VALIDATED` (FlashAttention-2 / Cutlass).
- [ ] **VRAM Footprint & Micro-Batch Tuning**: `NOT VALIDATED`.
- [ ] **Hardware Throughput (tok/sec)**: `NOT VALIDATED`.
