---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - gpt2
  - basikgpt
  - fineweb-edu
  - text-generation
---

# basikGPT-1 v1.0

**124M parameters** (GPT-2 Small). **v1.0 = 2.5B training tokens**, not 2.5B parameters.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("project-iconik/basikGPT-1-v1.0")
model = AutoModelForCausalLM.from_pretrained("project-iconik/basikGPT-1-v1.0")
```

Code and training recipe: [github.com/project-iconik/basikGPT](https://github.com/project-iconik/basikGPT).  
Next release: [basikGPT-1 v1.1](https://huggingface.co/project-iconik/basikGPT-1-v1.1) (same 124M, 5B tokens).

## Model details

| | |
|---|---|
| Architecture | GPT-2 Small decoder-only |
| Parameters | 124,439,808 |
| Context | 1024 |
| Vocab | 50,257 (GPT-2 BPE / tiktoken `gpt2`) |
| Training tokens | 2,500,001,792 (FineWeb-Edu) |
| Checkpoint | `runs/main_2p5b/step-00038147.pt` (step 38,147) |
| License (weights) | Apache-2.0 |

This Hub snapshot is a `GPT2LMHeadModel` export (safetensors + official GPT-2 tokenizer files). Optimizer state is not included.

## Training data

- [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), first 2.5B tokens, sequential (no shuffle).
- FineWeb-Edu is released under **ODC-By 1.0**. Downstream use should respect that license and the original page licenses.

## How it was trained

Single-GPU educational pretrain from random init. BF16, SDPA, batch 8 × 1024 × 8 grad accum (65,536 tokens/step), peak LR 6e-4 → min 6e-5, 2000-step warmup, cosine decay. Recipe and logs: the GitHub repo (`docs/main_2p5b.md`, `runs/main_2p5b/`).

## Evaluation (same protocol as the repo)

Zero-shot English suite measured in-repo (not mixed with published paper numbers). Token counts and architectures are **not** matched across rows.

| Model | Tokens / corpus | HS acc_norm | LAMBADA | PIQA | WG | ARC-E |
|---|---|---|---|---|---|---|
| **basikGPT-1 v1.0** | FineWeb-Edu 2.5B | **29.40%** | **19.58%** | **61.37%** | **50.51%** | **43.01%** |
| basikGPT-1 v1.1 | Edu 2.5B + FineWeb 2.25B + OpenWebMath 0.25B | 28.75% | 23.05% | 61.75% | 50.83% | 38.51% |
| openai-community/gpt2 | WebText | 30.37% | 30.93% | 62.57% | 51.62% | 38.13% |

WG is acc_raw; other tasks use the suite primary metric. n: HS 10,042 · LAMBADA 5,153 · PIQA 1,838 · WG 1,267 · ARC-E 2,376. Full protocol: [`benchmarks/REPORT.md`](https://github.com/project-iconik/basikGPT/blob/master/benchmarks/REPORT.md).

v1.0 is the only checkpoint in this pair that beats official GPT-2 Small on ARC-Easy (+4.88pp). LAMBADA is still well below gpt2 (−11.35pp). WinoGrande is chance-level.

## Intended use

Research and education: a small, reproducible GPT-2 Small trained on a documented public corpus. Not an instruction model. Not safe for open-ended production chat.

## Limitations

- 124M / 2.5B tokens: weak on long-range last-word prediction and pronoun resolution.
- English-centric FineWeb-Edu; no instruction or preference tuning.
- May reproduce biases and factual errors from web text.

## Citation

```
@software{basikgpt,
  title = {basikGPT},
  author = {basikGPT Contributors},
  url = {https://github.com/project-iconik/basikGPT},
  year = {2026}
}
```
