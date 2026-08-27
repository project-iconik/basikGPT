# RunPod GPU Setup, Qualification & Benchmark Guide (basikGPT)

This guide documents **Milestone 14**: baseline GPU qualification of the existing unified `Trainer` on NVIDIA GPU cloud instances (e.g. RunPod).

Milestone 14 is **not** performance optimization. It verifies that the CPU-validated training stack runs on CUDA FP32 and BF16, preserves checkpoint/resume and token accounting, and records GPT-2 Small 124M baseline VRAM and throughput.

`torch.compile`, DDP, FSDP, Tensor/Pipeline parallelism, DeepSpeed, ZeRO, activation checkpointing, custom kernels, and 2.5B-token main training are **out of scope**. Performance engineering after this baseline is documented in [`docs/performance.md`](performance.md) (Milestone 15).

Do **not** create a separate `GPUTrainer` / `BF16Trainer`. Use `device` and `precision` on the existing `Trainer`.

---

## 1. Credential Safety

Never write the following into the repository, `run.json`, metrics, checkpoints, or qualification JSON:

- RunPod API keys
- SSH private keys
- passwords
- Hugging Face tokens
- other secret environment variables

`.env` files must not be committed.

---

## 2. Persistent Storage

RunPod container local disks can be deleted when a pod is terminated. Keep shards, checkpoints, and `gpu_qualification.json` on a persistent volume (this project uses `/workspace`). Do not add a new cloud-storage integration.

---

## 3. Required Execution Order

Do not skip steps. Do not use `torch.compile`.

```text
1.  git rev-parse HEAD  (+ working-tree status)
2.  nvidia-smi
3.  PyTorch CUDA metadata
4.  pytest tests/training/test_cuda.py -v
5.  pytest -v
6.  tiny CUDA FP32 smoke
7.  GPT-2 Small CUDA FP32 smoke
8.  BF16 capability check (fail-fast; no silent FP32/FP16 fallback)
9.  GPT-2 Small CUDA BF16 smoke
10. CPU FP32 ↔ CUDA FP32 numerical sanity
11. FP32 ↔ BF16 loss sanity
12. CPU checkpoint → GPU load
13. GPU checkpoint → GPU resume (state-continuous, not bitwise data-order replay)
14. GPU checkpoint → CPU inspection
15. FP32 throughput / VRAM benchmark
16. BF16 throughput / VRAM benchmark
17. micro-batch capacity probe (OOM is a measurement, not a hidden failure)
18. short BF16 GPT-2 Small FineWeb-Edu dry-run
19. gpu_qualification.json
```

Orchestrator:

```bash
python scripts/collect_gpu_env.py --output-json runs/gpu_env.json
python scripts/qualify_gpu.py --output-dir runs/m14_gpu_qualification
```

---

## 4. Environment Qualification

```bash
nvidia-smi

python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('BF16:', torch.cuda.is_bf16_supported())
print('VRAM:', torch.cuda.get_device_properties(0).total_memory)
"
```

GPU model names are **metadata only**. Do not branch on `"RTX 4090"` / `"A100"` in training code. Capability checks (`torch.cuda.is_available()`, `torch.cuda.is_bf16_supported()`) are the supported control path.

If the user requests BF16 and the GPU does not support it, **fail-fast**.

---

## 5. Smoke, Benchmark, and Capacity Defaults

First GPT-2 Small correctness runs use a conservative micro-batch so VRAM is not the variable under test:

| Run | Model | Precision | Backend | B | T | G | Steps |
|---|---|---|---|---|---|---|---|
| Tiny smoke | tiny | fp32 | sdpa | 2 | 64 | 2 | 10 |
| GPT-2 Small FP32 smoke | gpt2_small | fp32 | sdpa | 1 | 1024 | 1 | 8 |
| GPT-2 Small BF16 smoke | gpt2_small | bf16 | sdpa | 1 | 1024 | 1 | 8 |
| FP32/BF16 benchmark | gpt2_small | fp32 then bf16 | sdpa | 1 | 1024 | 1 | 5 warmup + 20 measured |
| Capacity probe | gpt2_small | bf16 (then fp32 if cheap) | sdpa | 1,2,4,8,16 | 1024 | 1 | 1–2 |

Micro-batch $B$ is the samples in one forward/backward. Gradient accumulation $G$ is how many micro-batches are summed before `optimizer.step()`. Token batch is $B \times T \times G \times W$. They are not the same quantity.

Do **not** treat $B=4$, $G=8$ as a validated main-run setting until the capacity probe records a PASS for that $B$. After probe, a **provisional** main-run plan may use a measured PASS $B$; it is not a final hyperparameter.

The Trainer must **not** auto-reduce batch size on OOM.

---

## 6. Throughput and Peak VRAM

- Training LR warmup ≠ benchmark timing warmup.
- Time with `torch.cuda.synchronize()` before and after the measured window.
- Canonical throughput: `tokens_per_second = actual training target tokens / wall-clock seconds`.
- Primary memory metric: `torch.cuda.max_memory_allocated()` → `peak_allocated_vram_bytes`.
- Also record `torch.cuda.max_memory_reserved()` → `peak_reserved_vram_bytes`.
- Do not equate allocated, reserved, and `nvidia-smi` process usage.
- Do not claim “BF16 halves VRAM”. AdamW optimizer state may remain FP32; report measured values and distinguish parameters, gradients, activations, optimizer state, and temporaries.

Compare FP32 vs BF16 with identical model, initial weights seed, dataset, $T$, $B$, $G$, attention backend, warmup steps, and measured steps. Change **precision only**.

---

## 7. FineWeb-Edu Short Dry Run

Full 2.5B tokens are not required. Use the Milestone 6 shard format:

```bash
python scripts/prepare_fineweb_edu.py \
    --output data/fineweb-edu-smoke \
    --dataset-config sample-10BT \
    --max-train-tokens 100000 \
    --max-validation-tokens 10000 \
    --shard-token-target 50000 \
    --overwrite
```

Primary dry-run path: `device=cuda`, `precision=bf16`, `attention_backend=sdpa`, a micro-batch that PASSed the probe, **≤ 40 optimizer steps** or **≤ ~200K training tokens**.

Resume, if verified, is **state-continuous** (model, optimizer, scheduler, RNG, `global_step`, `tokens_seen`). It is not bitwise deterministic data-order replay.

---

## 8. Cost Boundary

Do not run full HellaSwag, 1M+ unnecessary pilots, full 2.5B preprocessing, or long hyperparameter sweeps on this milestone.

---

## 9. Reporting Language

Do not write: GPU training is perfect; BF16 is fully stable; all NVIDIA GPUs are supported; optimal batch size; production-ready; 2.5B training is guaranteed.

Write the actual scope, for example: this RunPod GPU ran GPT-2 Small CUDA FP32/BF16 training, state-continuous checkpoint/resume, token accounting, peak VRAM, and baseline throughput.
