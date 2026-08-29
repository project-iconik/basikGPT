---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - gpt2
  - basikgpt
  - fineweb
  - openwebmath
  - text-generation
---

# basikGPT-1 v1.1

**124M parameters** (GPT-2 Small). **v1.1 = 5B training tokens**, not 5B parameters.

Continues [basikGPT-1 v1.0](https://huggingface.co/project-iconik/basikGPT-1-v1.0) (FineWeb-Edu 2.5B) for another 2.5B tokens on FineWeb + OpenWebMath.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("project-iconik/basikGPT-1-v1.1")
model = AutoModelForCausalLM.from_pretrained("project-iconik/basikGPT-1-v1.1")
```

Code and training recipe: [github.com/project-iconik/basikGPT](https://github.com/project-iconik/basikGPT).

## Model details

| | |
|---|---|
| Architecture | GPT-2 Small decoder-only |
| Parameters | 124,439,808 |
| Context | 1024 |
| Vocab | 50,257 (GPT-2 BPE / tiktoken `gpt2`) |
| Lifetime tokens | 5,000,003,584 |
| This stage | FineWeb 2.25B + OpenWebMath 0.25B (no shuffle; 9:1 shard cycle) |
| Resume from | v1.0 / `runs/main_2p5b/step-00038147.pt` |
| Checkpoint | `runs/cont_5b_mix/step-00076294.pt` (step 76,294) |
| License (weights) | Apache-2.0 |

This Hub snapshot is a `GPT2LMHeadModel` export (safetensors + official GPT-2 tokenizer files). Optimizer state is not included.

## Training data

Lifetime mix: FineWeb-Edu 50% + FineWeb 45% + OpenWebMath 5%.

- [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — first 2.5B tokens (v1.0).
- [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT`, revision `9bb295ddab0e05d785b879661af7260fed5140fc`) — 2.25B tokens.
- [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) (revision `fde8ef8de2300f5e778f56261843dab89f230815`) — 0.25B tokens.

FineWeb / FineWeb-Edu: **ODC-By 1.0**. OpenWebMath: see the dataset card. Downstream use should respect those licenses and original page licenses.

## How it was trained

Resumed from v1.0 (did not retrain from scratch). Continuation LR: rewarm 6e-5 → 3e-4 over 1000 steps relative to origin step 38,147, then cosine to 6e-5 at 5B. Same microbatch recipe as v1.0 (BF16, SDPA, 65,536 tokens/step). `--no-shuffle --track-data-index --reset-data-index` on first resume. Details: GitHub `runs/cont_5b_mix/` and `docs/english_lm_suite_analysis.md`.

FineWeb-Edu validation CE rose during this stage (about 3.32 → 3.47). That is expected: the val set is still FineWeb-Edu while train leaves Edu.

## Evaluation (same protocol as the repo)

Zero-shot English suite measured in-repo (not mixed with published paper numbers). Token counts and architectures are **not** matched across rows.

| Model | Tokens / corpus | HS acc_norm | LAMBADA | PIQA | WG | ARC-E |
|---|---|---|---|---|---|---|
| basikGPT-1 v1.0 | FineWeb-Edu 2.5B | 29.40% | 19.58% | 61.37% | 50.51% | 43.01% |
| **basikGPT-1 v1.1** | Edu + FineWeb + math, 5B | **28.75%** | **23.05%** | **61.75%** | **50.83%** | **38.51%** |
| openai-community/gpt2 | WebText | 30.37% | 30.93% | 62.57% | 51.62% | 38.13% |

WG is acc_raw; other tasks use the suite primary metric. n: HS 10,042 · LAMBADA 5,153 · PIQA 1,838 · WG 1,267 · ARC-E 2,376. Full protocol: [`benchmarks/REPORT.md`](https://github.com/project-iconik/basikGPT/blob/master/benchmarks/REPORT.md).

Versus v1.0: LAMBADA **+3.47pp**, ARC-Easy **−4.50pp**, HellaSwag −0.65pp. The Edu-aligned ARC-Easy lead over official gpt2 is essentially gone. WinoGrande remains chance-level. This release does **not** claim GSM8K gains from 0.25B math tokens (not in the protocol).

## Intended use

Research and education: a small, reproducible GPT-2 Small continued on a documented public mix. Not an instruction model. Not safe for open-ended production chat.

## Limitations

- 124M / 5B tokens: still far from modern 135M models trained on much larger mixes.
- English-centric web text plus a small math slice; no instruction or preference tuning.
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
