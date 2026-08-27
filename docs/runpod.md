# RunPod GPU Setup, Verification & Benchmark Guide (basikGPT)

This guide documents the exact procedure to set up, verify, and benchmark **`basikGPT`** on NVIDIA GPU cloud instances (e.g. RunPod).

---

## 1. RunPod Instance Provisioning

### Recommended GPU Specifications
- **Primary Recommendation**: NVIDIA RTX 4090 (24GB VRAM), A100 (40GB/80GB), or H100 (80GB).
- **Minimum Requirement**: Any NVIDIA GPU with $\ge 16\text{GB}$ VRAM and compute capability $\ge 8.0$ (Ampere architecture or newer for native BF16 support).
- **Template**: PyTorch 2.x with CUDA 12.1+ / Ubuntu 22.04 LTS.

> [!WARNING]
> **Ephemeral Storage Notice**: RunPod container local disks (`/workspace` or root) are deleted when terminating pods. Always use a RunPod Persistent Network Volume for dataset shards and checkpoint storage.

> [!CAUTION]
> **Security Policy**: NEVER commit RunPod API keys, SSH private keys, or cloud access tokens to the repository.

---

## 2. Environment Setup

Inside the RunPod Web Terminal or SSH session:

```bash
# 1. Clone the repository
git clone https://github.com/your-org/basikGPT.git
cd basikGPT

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install basikGPT in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev,data]"

# 4. Verify CUDA runtime availability
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)} | BF16 Supported: {torch.cuda.is_bf16_supported()}')"
```

Expected output:
```text
PyTorch: 2.x.x+cu12x | CUDA: 12.x | GPU: NVIDIA RTX 4090 (or A100) | BF16 Supported: True
```

---

## 3. Prepare Smoke Dataset on GPU Pod

Generate the educational token shards directly on the pod:

```bash
python scripts/prepare_fineweb_edu.py \
    --output data/fineweb-edu-smoke \
    --dataset-config sample-10BT \
    --max-train-tokens 50000 \
    --max-validation-tokens 5000 \
    --shard-token-target 25000 \
    --overwrite
```

---

## 4. Run Automated CUDA Test Suite

Execute all tests, including GPU-specific tests:

```bash
# Run CUDA-specific test suite
pytest tests/training/test_cuda.py -v

# Run full project test suite on GPU
pytest -v
```

All tests (including `@pytest.mark.cuda`) must pass on the GPU node.

---

## 5. GPT-2 Small Smoke Training on CUDA

### 5.1. Baseline CUDA FP32 Run
```bash
python scripts/train.py \
    --model-preset gpt2_small \
    --data-dir data/fineweb-edu-smoke \
    --device cuda \
    --precision fp32 \
    --batch-size 4 \
    --context-length 1024 \
    --grad-accum-steps 8 \
    --max-steps 20 \
    --warmup-steps 5 \
    --eval-interval 10 \
    --checkpoint-interval 10 \
    --log-interval 5 \
    --output-dir runs/runpod_gpt2_fp32
```

### 5.2. Primary CUDA BF16 Mixed-Precision Run
```bash
python scripts/train.py \
    --model-preset gpt2_small \
    --data-dir data/fineweb-edu-smoke \
    --device cuda \
    --precision bf16 \
    --batch-size 4 \
    --context-length 1024 \
    --grad-accum-steps 8 \
    --max-steps 20 \
    --warmup-steps 5 \
    --eval-interval 10 \
    --checkpoint-interval 10 \
    --log-interval 5 \
    --output-dir runs/runpod_gpt2_bf16
```

### 5.3. Checkpoint Resume Verification on GPU
```bash
python scripts/train.py \
    --model-preset gpt2_small \
    --data-dir data/fineweb-edu-smoke \
    --device cuda \
    --precision bf16 \
    --batch-size 4 \
    --context-length 1024 \
    --grad-accum-steps 8 \
    --max-steps 30 \
    --warmup-steps 5 \
    --output-dir runs/runpod_gpt2_bf16 \
    --resume runs/runpod_gpt2_bf16/step-00000020.pt
```

---

## 6. Throughput & Peak VRAM Benchmark

Run comparative throughput and memory profiling between FP32 and BF16:

### 6.1. Benchmark CUDA FP32
```bash
python scripts/benchmark_training.py \
    --model-preset gpt2_small \
    --context-length 1024 \
    --attention-backend sdpa \
    --device cuda \
    --precision fp32 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --warmup-steps 5 \
    --measured-steps 20 \
    --output-json runs/benchmark_cuda_fp32.json
```

### 6.2. Benchmark CUDA BF16
```bash
python scripts/benchmark_training.py \
    --model-preset gpt2_small \
    --context-length 1024 \
    --attention-backend sdpa \
    --device cuda \
    --precision bf16 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --warmup-steps 5 \
    --measured-steps 20 \
    --output-json runs/benchmark_cuda_bf16.json
```

### 6.3. Benchmark Summary Format
Results are saved to structured JSON files containing:
- `tokens_per_second` (computed as $\frac{\text{measured target tokens}}{\text{wall-clock elapsed time}}$ with `torch.cuda.synchronize()`)
- `peak_vram_mb` (measured via `torch.cuda.max_memory_allocated()`)
- `gpu_name`, `compute_capability`, `cuda_version`, `pytorch_version`.
