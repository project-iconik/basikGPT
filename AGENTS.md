# basikGPT — Coding Agent Master Guidelines

This repository contains **basikGPT**, an educational, reproducible, and open-source project building and pretraining GPT-2 Small from scratch in PyTorch.

The codebase is explicitly designed for human reverse engineering, code review, and deep learning engineering education.

---

## 1. Priority Hierarchy

Every contributor and coding agent MUST adhere to this strict priority ranking:

```text
Correctness
>
Readability
>
Verifiability
>
Reproducibility
>
Performance Optimization
>
Code Conciseness
```

Do not implement entire subsystems at once. Work on one clearly scoped milestone or concept at a time.

---

## 2. Canonical Model: GPT-2 Small

The initial reproduction target is standard GPT-2 Small with the following configuration:

- `vocab_size = 50,257`
- `context_length = 1,024`
- `n_layers = 12`
- `n_heads = 12`
- `d_model = 768`
- `head_dim = 64` (`d_model // n_heads`)
- `d_ff = 3,072` (`4 * d_model`)
- `architecture = decoder-only Transformer`
- `attention = causal multi-head self-attention`
- `position encoding = learned absolute positional embedding`
- `normalization = LayerNorm (eps=1e-5)`
- `activation = GPT-2-compatible GELU` (tanh approximation)
- `weight tying = token embedding <-> language-model head`
- `bias = True` (in linear layers and LayerNorms)

### Topology: Pre-LayerNorm
```python
x = x + attention(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```
Followed by a final `LayerNorm` after the 12th Transformer block before `lm_head`.

---

## 3. Strict Restrictions During GPT-2 Reproduction

Until canonical GPT-2 Small reproduction and reference parity (Milestones 0–5) are fully verified, the following are **STRICTLY PROHIBITED**:

- RoPE (Rotary Position Embedding)
- RMSNorm
- SwiGLU / GeGLU
- GQA (Grouped-Query Attention) / MQA (Multi-Query Attention)
- ALiBi / Sliding Window Attention
- Mixture of Experts (MoE)
- Custom tokenizers (must use GPT-2 standard tokenizer)
- Extended context length (> 1024)
- Removing GPT-2 biases
- Replacing learned positional embeddings
- Altering Transformer block topology

---

## 4. Tensor Shape Conventions

All code docstrings and explanations MUST follow these standard symbols:

- `B`: Batch size
- `T`: Sequence length / time dimension
- `C`: Model / embedding dimension (`d_model`)
- `H`: Number of attention heads (`n_heads`)
- `D`: Head dimension (`head_dim = C // H`)
- `V`: Vocabulary size (`vocab_size`)

Standard transformations:
- Hidden state: `(B, T, C)`
- Attention Q/K/V split: `(B, T, C) -> (B, H, T, D)`
- Attention score matrix ($Q K^T$): `(B, H, T, T)`
- Attention output merged: `(B, H, T, D) -> (B, T, C)`

---

## 5. Milestone Roadmap

- **Milestone 0**: Repository Foundation (Packaging, `GPTConfig`, validation, tests, docs)
- **Milestone 1**: Eager Causal Self-Attention (pure PyTorch tensor ops, step-by-step)
- **Milestone 2**: SDPA Backend (`F.scaled_dot_product_attention` & parity tests)
- **Milestone 3**: GPT-2 Components (MLP, LayerNorm, Block, Embeddings, Weight Tying)
- **Milestone 4**: Complete GPT-2 Small Assembly (forward/backward, causality, parameter count)
- **Milestone 5**: Reference Parity (import OpenAI/HF weights, verify logits parity)
- **Milestone 6**: English FineWeb Data Pipeline (streaming, uint16 token shards)
- **Milestone 7**: Pretraining Engine (AdamW, cosine schedule, warmup, grad accum/clip, BF16)
- **Milestone 8**: Training Validation (single-batch overfit, 1M -> 10M -> 100M -> 500M tokens)
- **Milestone 9**: Performance Engineering (Eager vs SDPA, FP32 vs BF16, torch.compile)
- **Milestone 10**: Canonical Pretraining (~2.5B FineWeb tokens)
- **Milestone 11**: Evaluation & Analysis
- **Milestone 12**: Scaling Experiments
- **Milestone 13**: Distributed Training (DDP -> FSDP)
- **Milestone 14**: 30B-ready Validation (meta-device, sharded initialization)
- **Milestone 15**: Technical Whitepaper
