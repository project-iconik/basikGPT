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
| **Stage B (Short Pilot)** | $100,000$ | $4, 128, 2, 1$ | $1,024$ | $98$ | $100,352$ | $+352$ | Overall downward loss trend (not per-step monotone), gradient-norm stability, validation cadence, state-continuous resume. |
| **Stage C (Extended)** | $1,000,000$ | $4, 256, 4, 1$ | $4,096$ | $245$ | $1,003,520$ | $+3,520$ | Multi-thousand step regime, schedule boundary compliance, extended throughput tracking. |

---

## 4. Training Health Monitoring & Failure Guardrails

Pilot runs enforce fail-fast guardrails to abort immediately on numerical divergence:

1. **Finite Loss Guardrail**:
   If $\text{loss} \in \{\text{NaN}, +\infty, -\infty\}$ is encountered during training or evaluation, `FloatingPointError` is raised immediately.
2. **Finite Gradient Guardrail**:
   After unscaling (FP16) and before or during clipping, the total Euclidean gradient norm $\|\mathbf{g}\|_2$ is measured. If $\|\mathbf{g}\|_2 \in \{\text{NaN}, +\infty\}$, `FloatingPointError` is raised immediately. This check applies to FP32, BF16, and FP16 (`GradScaler`) paths.
3. **Logged Gradient Norm**:
   Gradient norms recorded in `metrics.jsonl` are the true pre-clipping total Euclidean norm $\|\mathbf{g}\|_2 = \sqrt{\sum_i \|\mathbf{g}_i\|_2^2}$, including when `max_grad_norm` is `None` (no clipping). They are never reported as `0.0` merely because clipping is disabled. Train records also store `grad_clipped` (`true` iff clipping is enabled and the pre-clip norm exceeded `max_grad_norm`).
4. **Whitepaper convenience fields**:
   `run.json` `extra` records `parameter_count` and `tokens_per_optimizer_step`. Train records store Kaplan-style `estimated_flops = 6ND` over the logging interval (not MFU). Val records store `val_perplexity = exp(val_loss)`.
5. **Learning Rate Alignment**:
   Scheduler advances strictly once per global optimizer step (after all $G$ micro-batches), maintaining exact warmup and cosine decay boundaries.

---

## 5. Checkpoint & Resume Continuity Audit

### 5.1. Restored State Components
- Model parameter weights (`state_dict`).
- Optimizer first ($m_t$) and second ($v_t$) momentum buffers and step counters.
- Mixed precision GradScaler state (if FP16).
- Global optimizer step (`global_step`) and cumulative `tokens_seen`.
- Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA RNG state vectors. PyTorch RNG tensors are restored on CPU even when the checkpoint is mapped onto a CUDA device.

### 5.2. Data Stream Continuity Classification
- **Classification**: *"State-continuous (model, optimizer, scheduler, RNG restored), but dataset iterator resets to shard start if not fast-forwarded."*
- `DataLoader` streams initialize from the beginning of shard files upon resumption. Shuffle order at epoch start is seeded via `torch.Generator`, but the exact in-epoch batch offset is not restored. In full-scale distributed pretraining, shard offset tracking or deterministic index positioning is used.
- On resume, `metrics.jsonl` records with `step` greater than the restored `global_step` are dropped so later appends do not duplicate steps.

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

### 7.1. Validated on Local CPU (Milestone 8)
- [x] **Analytical Token Accounting**: Exact $B \times T \times G \times W$ step and token calculations.
- [x] **LR Scheduler Alignment**: Warmup and cosine decay synchronized strictly with optimizer steps.
- [x] **Finite Loss & Gradient Guardrails**: Fail-fast on NaN/Inf with `FloatingPointError` on FP32/BF16/FP16 paths.
- [x] **Validation Isolation**: Validation tokens strictly excluded from `tokens_seen`.
- [x] **Checkpoint Serialization**: Atomic save/load of model, optimizer, scaler, and RNG states.
- [x] **Resume Continuity**: Verifiable state resumption without step regression (DataLoader shuffle position is not restored; see §5.2).
- [x] **Structured Summary Logging**: Complete machine-readable `pilot_summary.json` output.

### 7.2. GPU Qualification (Milestone 14) — NVIDIA RTX PRO 4500 Blackwell, 32 GiB, RunPod

Measured on git `4ddcd7900d53aa92550bd20c32887434a838150d` (working tree dirty with Milestone 14 tooling). Attention is reported as **PyTorch SDPA**; kernel dispatch (Flash / Memory-Efficient / Math) was not inspected.

- [x] **CUDA FP32**: `VALIDATED` (tiny 10 steps; GPT-2 Small 8 steps, B=1, T=1024, G=1, finite loss/grad, token accounting exact)
- [x] **BF16**: `VALIDATED` (same GPT-2 Small smoke; `torch.cuda.is_bf16_supported()=True`; no GradScaler)
- [x] **PyTorch SDPA on GPU**: `VALIDATED` as the training backend (`attention_backend=sdpa`)
- [x] **GPU checkpoint / state-continuous resume**: `VALIDATED` (CPU→GPU load, GPU→GPU resume of `global_step`/`tokens_seen`, GPU→CPU inspection)
- [x] **Peak VRAM (allocated / reserved)**: `VALIDATED` (see capacity table; names are `max_memory_allocated` / `max_memory_reserved`)
- [x] **GPU throughput (tokens/sec)**: `VALIDATED` at B=1, T=1024, G=1, 5 timing-warmup + 20 measured steps: FP32 ≈ 17.4k tok/s, BF16 ≈ 29.7k tok/s
- [x] **Micro-batch capacity (T=1024)**: `VALIDATED` — B=1..16 PASS for FP32 and BF16; B=32 OOM for both

Not in Milestone 14 scope:

- [ ] **torch.compile**: see §7.3 below
- [ ] **DDP / FSDP**: `NOT VALIDATED`
- [ ] **2.5B-token main training**: `NOT EXECUTED`

Provisional single-device main-run sketch using the largest PASS micro-batch on this GPU (not a final hyperparameter):

$$B=16,\ T=1024,\ G=8,\ W=1 \Rightarrow \text{tokens/step} = 131{,}072$$

Micro-batch $B$ is not the same as the token batch $B \times T \times G \times W$.

### 7.3. GPU Performance Candidates (Milestone 15) — same GPU / PyTorch 2.8.0

Measured with synthetic timing (uncompiled vs one controlled change) plus short FineWeb-Edu compiled runs (~196K tokens). Not a hyperparameter freeze. Numbers below are the published summary; detailed lab tables are not in git.

Uncompiled BF16 T=1024 G=1: B=1 ≈ 30.1k tok/s; B=8 ≈ 75.2k; B=16 ≈ 79.3k. Gain flattens after B=8. AUTO SDPA matched FLASH_ATTENTION on B=8; MATH/eager ≈ 31k tok/s.

Compiled Inductor `default` B=16: ≈ 87.2k tok/s synthetic, FineWeb short run finite + resume. `reduce-overhead` B=8: ≈ 79.2k tok/s, lower VRAM. Uncompiled B=8 G=8 remains a conservative 65,536 tokens/step sketch.

2.5B analytical steps (ceiling): B=16 G=1 → 152,588; B=8 G=8 → 38,147; B=16 G=8 → 19,074.

DDP / FSDP / 2.5B main training were not started.

### 7.4. Configuration Freeze (Milestone 16) — same GPU

1M and 10M FineWeb-Edu pilots compared uncompiled `B=8 G=8` vs compiled `B=16 G=4` at **65,536 tokens/step**. Both stayed finite, validation declined overall, and process-level resume used sequential `data_sample_index`. Detailed A/B tables are not in git.

Provisional canonical single-GPU config (not claimed optimal): BF16, SDPA auto, `compile=false`, `B=8`, `T=1024`, `G=8`, 65,536 tokens/step. Artifact: [`configs/gpt2_small_fineweb_edu_single_gpu.json`](../configs/gpt2_small_fineweb_edu_single_gpu.json). 2.5B main training was not started.
