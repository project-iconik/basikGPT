# GPU Performance Engineering (Milestone 15)

This guide documents **Milestone 15**: controlled `torch.compile`, SDPA backend, micro-batch, and gradient-accumulation measurements on the existing unified `Trainer`.

Canonical training default remains:

```text
torch.compile disabled
BF16
PyTorch SDPA (auto dispatch)
```

Results below were measured on **NVIDIA RTX PRO 4500 Blackwell / PyTorch 2.8.0+cu128 / CUDA 12.8**. They are not claimed to be optimal, production-ready, or the fastest possible GPT-2.

DDP, FSDP, custom kernels, activation checkpointing, FineWeb format changes, and 2.5B-token main training are **out of scope**. Milestone 14 uncompiled qualification remains in [`docs/runpod.md`](runpod.md).

---

## 1. How to opt in

```bash
python scripts/train.py --precision bf16 --compile --compile-mode default
python scripts/benchmark_performance.py --output-dir runs/m15_performance
```

`--compile` defaults to off. `--compile-mode` is `default` or `reduce-overhead` (`max-autotune` is not enabled). `--sdpa-kernel auto` leaves PyTorch dispatch unchanged; other values force one backend exclusively.

The GPT module is compiled. Optimizer, loss, checkpoints, and `GPTConfig` stay on the uncompiled `raw_model`. Checkpoint keys must not contain `_orig_mod.`.

---

## 2. Timing definitions

- Training LR warmup ≠ benchmark timing warmup.
- `time_to_first_optimizer_step` includes tracing/compilation.
- `steady_state_tokens_per_second` is measured-window tokens / wall-clock after warmup, with `torch.cuda.synchronize()`.
- `end_to_end_tokens_per_second_including_compile` includes compile/warmup.
- Throughput is **training target tokens / wall-clock**, not optimizer steps/sec.
- Peak memory: `max_memory_allocated` / `max_memory_reserved`. Do not equate those with `nvidia-smi`.

Compile break-even tokens (compiled faster only):

```text
N = C * R0 * R1 / (R1 - R0)
```

If `R1 <= R0`, there is no break-even.

---

## 3. Baseline re-measurement (uncompiled BF16, B=1, T=1024, G=1, 5+20 steps)

| Item | Value |
|---|---|
| tokens/sec | 30,123 |
| peak allocated | 2.58 GiB |
| Milestone 14 (same shape) | 29,683 tok/s, ≈2.58 GiB |

Same GPU/software family as Milestone 14. Small run-to-run scatter is expected.

---

## 4. torch.compile (Inductor)

Cold first-process compile overhead (GPT-2 Small, this GPU):

| Mode | B | Steady tok/s | Speedup vs B=1 baseline | Peak allocated | Cold compile s |
|---|---|---|---|---|---|
| default | 8 | 80,953 | 2.69 | 8.85 GiB | 19.3 |
| default | 16 | 87,164 | 2.89 | 16.01 GiB | 18.0 |
| reduce-overhead | 8 | 79,166 | 2.63 | 7.89 GiB | 14.9 |
| reduce-overhead | 16 | 85,162 | 2.83 | 15.07 GiB | 15.0 |

Warm inductor cache on a rerun dropped compile time to ~1 s. Short jobs should use the cold number.

Same-B uncompiled BF16 T=1024 G=1: B=8 ≈ 75,191 tok/s; B=16 ≈ 79,257 tok/s. Compiled `default` B=16 is about **1.10×** that uncompiled B=16 rate. Compile is not always faster than a larger uncompiled batch; here it is a modest same-B gain plus a large gain versus B=1.

Tiny FP32 compiled vs uncompiled logits: max abs diff ≈ 1.2e-7. Compiled checkpoints load into an uncompiled `GPT`.

---

## 5. SDPA backends (uncompiled BF16, B=8, T=1024, G=1)

Runtime `SDPBackend` members present: MATH, FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION.

| Backend | Status | tok/s | Peak allocated |
|---|---|---|---|
| AUTO | PASS | 75,207 | 9.31 GiB |
| MATH | PASS | 30,605 | 14.22 GiB |
| FLASH_ATTENTION | PASS | 75,264 | 9.31 GiB |
| EFFICIENT_ATTENTION | PASS | 70,613 | 9.31 GiB |
| CUDNN_ATTENTION | PASS | 73,837 | 9.31 GiB |
| manual eager | PASS | 30,942 | 16.06 GiB |

AUTO matched FLASH_ATTENTION within run noise on this shape. Exclusive force did not silently count a fallback as success. Different backends are not bitwise equal; this milestone does not claim Flash/Efficient training is deterministic.

---

## 6. Micro-batch sweep (uncompiled BF16, T=1024, G=1)

| B | tok/s | Peak allocated | Status |
|---|---|---|---|
| 1 | 30,123 | 2.58 GiB | PASS |
| 2 | 47,257 | 3.93 GiB | PASS |
| 4 | 64,000 | 5.71 GiB | PASS |
| 8 | 75,191 | 9.31 GiB | PASS |
| 16 | 79,257 | 16.47 GiB | PASS |

Throughput gain flattens after B=8. Maximum stable micro-batch (B=16) is not the same as the most VRAM-efficient B.

---

## 7. Gradient accumulation / token batch (uncompiled BF16, T=1024)

| B | G | Tokens/step | tok/s | Peak allocated |
|---|---|---|---|---|
| 8 | 2 | 16,384 | 79,787 | 9.78 GiB |
| 8 | 4 | 32,768 | 82,259 | 9.78 GiB |
| 8 | 8 | 65,536 | 83,622 | 9.78 GiB |
| 16 | 4 | 65,536 | 82,922 | 16.94 GiB |
| 16 | 8 | 131,072 | 83,430 | 16.94 GiB |

Larger G does not make the GPU compute faster; it changes optimizer-step frequency and the global token batch. B=8 G=8 and B=16 G=4 share 65,536 tokens/step at similar tok/s; B=8 uses less VRAM.

---

## 8. Candidates (not a frozen recipe)

| Role | Compile | B | G | Tokens/step | Synthetic tok/s | FineWeb short run |
|---|---|---|---|---|---|---|
| Best raw throughput | default | 16 | 1 | 16,384 | 87,164 | 196,608 tokens, loss 11.0→7.86, resume OK |
| Best VRAM-efficient (B≥8) | reduce-overhead | 8 | 1 | 8,192 | 79,166 | 196,608 tokens, loss 11.0→7.40, resume OK |
| Conservative uncompiled | off | 8 | 8 | 65,536 | 83,622 | synthetic G-sweep only |

2.5B-token analytical step counts (`calculate_training_steps`, ceiling policy):

| Candidate | Optimizer steps | Actual tokens |
|---|---|---|
| B=16 G=1 | 152,588 | 2,500,001,792 |
| B=8 G=1 | 305,176 | 2,500,001,792 |
| B=8 G=8 | 38,147 | 2,500,001,792 |
| B=16 G=8 (M14 provisional) | 19,074 | 2,500,067,328 |

These are planning numbers, not a hyperparameter freeze.

Spawning FineWeb `train.py` from a process that still holds a large CUDA cache can OOM a compiled B=16 run that PASSes in a clean process. The orchestrator now calls `empty_cuda()` before stability subprocesses.

---

## 9. What this milestone does not claim

Do not write: optimal configuration; fastest possible GPT-2; FlashAttention guaranteed; torch.compile always faster; production optimized.

Measured here: RTX PRO 4500 Blackwell / PyTorch 2.8.0 baseline vs compile/SDPA/batch candidates for single-GPU GPT-2 Small pretraining.
