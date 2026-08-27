# Single-GPU Pretraining Configuration Freeze (Milestone 16)

This guide records **Milestone 16**: 1M- and 10M-token FineWeb-Edu GPU pilots that compare two execution strategies at an identical token batch, then freeze one **provisional** single-GPU baseline.

RTX PRO 4500 Blackwell 단일 GPU에서 1M/10M FineWeb-Edu pilot을 통해 동일 token-batch 후보의 안정성, validation behavior, sustained throughput, checkpoint/resume 및 VRAM headroom을 비교하고 main-run baseline configuration을 잠정 freeze했다.

This is **not** an optimal-hyperparameter claim. Compile is not always better. 10M tokens does not prove final model quality. 2.5B training is not started and is not guaranteed.

Canonical artifact: [`configs/gpt2_small_fineweb_edu_single_gpu.json`](../configs/gpt2_small_fineweb_edu_single_gpu.json). Machine-readable pilots: `runs/m16_pilot/`.

---

## 1. Repository and hardware

| Item | Value |
|---|---|
| git SHA | `a9b7ecc425e55a33513199f90324e72c9c8361b4` |
| git_dirty | `false` (pilots ran on this clean revision) |
| GPU | NVIDIA RTX PRO 4500 Blackwell |
| VRAM | 31.37 GiB |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| BF16 | supported |

---

## 2. Dataset

Canonical corpus unchanged: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, tokenizer GPT-2 BPE (`tiktoken` `gpt2`). Prepared with the Milestone 6 **streaming** pipeline (`sample-10BT` was not downloaded in full).

| Split | Tokens | Shards | Notes |
|---|---|---|---|
| train | 11,998,973 | 12 | uint16, ~26MB total directory |
| validation | 493,489 | 1 | shared by Candidate A and B |

2.5B-scale shards were not generated.

---

## 3. Candidates (identical token batch)

Both candidates use from-scratch GPT-2 Small (`dropout=0.0`), seed `1337`, shared CPU `state_dict` (`runs/m16_pilot/shared_init.pt`), peak LR `6e-4`, min LR `6e-5`, AdamW, SDPA auto, `T=1024`, `W=1`.

| | Candidate A (Conservative) | Candidate B (Compiled) |
|---|---|---|
| compile | false | true, inductor `default` |
| B | 8 | 16 |
| G | 8 | 4 |
| tokens/step | 65,536 | 65,536 |

65,536 tokens/step is a **controlled comparison** choice, not a claim that this global batch is optimal. Scheduler is optimizer-step based (linear warmup + cosine), so equal tokens/step keeps A/B LR progression aligned.

Warmup policy: `max(1, round(0.10 * max_steps))`. 1M (16 steps) → 2. 10M (153 steps) → 15. The 2.5B recipe default remains **2000 warmup steps** (provisional; not copied from the 10% pilot fraction). Linear LR scaling was not applied.

Eval: 131,072 validation tokens per eval (A: 16 batches, B: 8 batches). Same shards and cadence.

---

## 4. 1M results

Actual: 16 × 65,536 = **1,048,576** tokens (+48,576 overshoot). Checkpoint: step 8 + step 16. Process-level resume at step 8. Sequential `data_sample_index` restored (`resume_class=exact-sample-index`).

| Candidate | Compile | B | G | Actual tokens | Train loss init→final | Val final | Grad min/max | Peak allocated | Resume | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| A | no | 8 | 8 | 1,048,576 | 10.953 → 7.742 | 7.694 | 0.47 / 14.67 | 9.31 GiB | exact-sample-index | PASS |
| B | yes | 16 | 4 | 1,048,576 | 10.953 → 7.743 | 7.695 | 0.47 / 14.67 | 16.48 GiB | exact-sample-index | PASS |

Val curve (shared cadence): start ≈ 10.956, mid (524,288 tok) ≈ 7.935, end ≈ 7.694.

1M `summary.json` training-only tok/s is **not** used for freeze: the run is split across two processes, so cumulative `tokens_seen` is divided by the second process's train timer. Logged per-step rates after warmup were about **84k tok/s (A)** and **90–91k tok/s (B)**. Sustained numbers below come from the uninterrupted 10M runs.

B 1M first-step wall-clock included compile (~5.9k tok/s on step 1). No unexplained repeated recompilation (`later_step_time_spikes=[]`).

---

## 5. 10M results

Actual: 153 × 65,536 = **10,027,008** tokens (+27,008 overshoot). Checkpoints at 38 / 76 / 115 / 153. Single-process training for throughput. Separate process-level resume probe from step 76 → 77 (`resume_ok=true`); probe `.pt` files deleted.

| Candidate | Compile | B | G | Actual tokens | Train init→final | Val final | Grad min/median/max | Train tok/s | E2E tok/s | Peak alloc / reserved | Cold compile s | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | no | 8 | 8 | 10,027,008 | 10.953 → 6.421 | 6.547 | 0.30 / 0.60 / 14.67 | 82,941 | 74,373 | 9.30 / 10.29 GiB | — | PASS |
| B | yes | 16 | 4 | 10,027,008 | 10.953 → 6.422 | 6.548 | 0.33 / 0.59 / 14.67 | 88,995 | 78,226 | 16.48 / 18.06 GiB | 1.16 | PASS |

Validation vs tokens (A / B):

| Tokens | A val | B val |
|---|---|---|
| 0 | 10.956 | 10.956 |
| 2,490,368 | 7.234 | 7.230 |
| 4,980,736 | 6.797 | 6.803 |
| 7,471,104 | 6.622 | 6.625 |
| 10,027,008 | 6.547 | 6.548 |

Loss/validation declined overall (not required to be monotone per step). All recorded train/val losses and gradient norms were finite. B showed no repeated recompile spikes on fixed `B=16, T=1024`.

Compile break-even from 10M training-only rates and B cold compile 1.16s: **≈ 1.41M tokens**. 10M recovers that overhead; a 1M job is near the break-even region once process-split compile is counted.

Peak allocated for B stayed ~16.5 GiB of 31.37 GiB (headroom remains, but less than A). B=32 remains an OOM-adjacent setting from Milestone 15 and was not chosen.

---

## 6. Canonical freeze

**Canonical:** Candidate A — BF16, PyTorch SDPA auto, `compile=false`, `B=8`, `T=1024`, `G=8`, `W=1`, 65,536 tokens/step.

**Fastest (this GPU, 10M window):** Candidate B, about **1.07×** A's training-only throughput, with compile dependency and ~7 GiB extra allocated VRAM.

Canonical is not Fastest. Selection used correctness, matching validation curves, resume, VRAM headroom, and operational simplicity. A ~7% compile speedup did not justify inductor version-sensitivity or the larger memory footprint for the main-run baseline.

Freeze means: later changes to this baseline need experimental evidence. It does not mean the file is immutable or that LR/warmup/token-batch are proven optimal.

Provisional 2.5B extras recorded with the freeze (not tuned here): peak LR `6e-4`, min LR `6e-5`, warmup **2000** steps.

---

## 7. 2.5B analytical plan (canonical B/T/G/W)

Ceiling arithmetic via `calculate_training_steps`:

| Item | Value |
|---|---|
| tokens/step | 65,536 |
| optimizer steps | 38,147 |
| actual tokens | 2,500,001,792 |
| overshoot | +1,792 |

Rough runtime from 10M **training-only** 82,941 tok/s: 2.5e9 / 82941 ≈ **8.4 GPU-hours**. From observed **end-to-end** 74,373 tok/s (includes eval/checkpoint): ≈ **9.3 GPU-hours**. These are not exact ETAs. Cost is not estimated (no trusted hourly price in run metadata).

---

## 8. Checkpoint / resume decision

Implemented in this milestone: sequential shard iteration stores `data_sample_index` in checkpoint `extra_state`. Resume fast-forwards by integer sample index (`index % len(dataset)` across epochs). Shuffle remains the default for `scripts/train.py` unless `--no-shuffle --track-data-index` is set.

Classification for M16 pilots: **exact-sample-index** (state-continuous **and** data order). Distributed samplers were not added. This should be kept for a 2.5B main run on one GPU.

---

## 9. Main-run readiness

| Gate | Status |
|---|---|
| 10M finite train/val/grad | PASS (A and B) |
| checkpoint / process-level resume | PASS (exact-sample-index) |
| token accounting | PASS (1,048,576 and 10,027,008) |
| unexplained compile failure | none |
| unexplained OOM | none |
| VRAM headroom on canonical | PASS (~9.3 GiB allocated of 31.4 GiB) |
| artifacts preserved | `runs/m16_pilot/` |
| 2.5B main training | **not started** |

---

## 10. Intentionally deferred

DDP, FSDP, tensor parallelism, DeepSpeed/ZeRO, activation checkpointing, custom Triton/FlashAttention, new tokenizer/dataset, broad LR/batch sweeps, 2.5B main training.

Recommended next: **Milestone 17 — Main Pretraining Readiness Audit & 100M Token Scale Test**.
