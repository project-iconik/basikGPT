# basikGPT-1 Whitepaper

**English** · [日本語](whitepaper.ja.md) · [한국어](whitepaper.ko.md)

| | |
| --- | --- |
| Authors | basikGPT Contributors |
| Document version | 1.0 |
| Date | 2026-08-29 |
| Package | `basikgpt` 0.1.0 |
| Production runs | `main_2p5b` (38,147 steps) → `cont_5b_mix` (76,294 steps) |
| Weights | [`project-iconik/basikGPT-1-v1.0`](https://huggingface.co/project-iconik/basikGPT-1-v1.0), [`project-iconik/basikGPT-1-v1.1`](https://huggingface.co/project-iconik/basikGPT-1-v1.1) |
| Code | [`project-iconik/basikGPT`](https://github.com/project-iconik/basikGPT) |

This technical whitepaper records the architecture, tokenizer, data mix, two completed production pretraining runs, language-model metrics, and a zero-shot English LM suite comparison for **basikGPT-1**.

Machine-readable tables for the 2.5B run live in [`runs/main_2p5b/WHITEPAPER.md`](../runs/main_2p5b/WHITEPAPER.md). This document is the narrative record of both checkpoints.

---

## Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Related work](#3-related-work)
4. [Model](#4-model)
5. [Tokenizer](#5-tokenizer)
6. [Data](#6-data)
7. [Training](#7-training)
8. [Compute](#8-compute)
9. [Language-model results](#9-language-model-results)
10. [English LM suite](#10-english-lm-suite)
11. [Intended use, limitations, and licenses](#11-intended-use-limitations-and-licenses)
12. [Reproducibility](#12-reproducibility)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)
15. [Appendix](#appendix)

---

## 1. Abstract

basikGPT-1 is a **124,439,808**-parameter GPT-2 Small decoder-only Transformer trained from scratch in PyTorch. Two production stages ran on a single NVIDIA RTX PRO 4500 Blackwell:

- **v1.0** (`main_2p5b`): **2,500,001,792** FineWeb-Edu tokens (about **20.09** tokens per parameter) in **29,462.59 s** (8.18 GPU hours). Post-training full validation cross-entropy / perplexity is **3.2548 / 25.9151**. Protocol HellaSwag `acc_norm` is **29.40%**.
- **v1.1** (`cont_5b_mix`): continues v1.0 for another 2.5B tokens on FineWeb 2.25B + OpenWebMath 0.25B. Lifetime tokens **5,000,003,584** (about **40.18** tokens per parameter) in **29,593.20 s** (8.22 GPU hours) for this stage. LAMBADA rises **+3.47 pp**; ARC-Easy falls **−4.50 pp**.

Weights are on Hugging Face Hub as `GPT2LMHeadModel` exports: [`project-iconik/basikGPT-1-v1.0`](https://huggingface.co/project-iconik/basikGPT-1-v1.0) and [`project-iconik/basikGPT-1-v1.1`](https://huggingface.co/project-iconik/basikGPT-1-v1.1).

The model is a pretrained **base**, not an instruction-tuned chatbot. The comparison table is a shared-protocol baseline, not a compute-matched claim that GPT-2 Small was beaten.

---

## 2. Introduction

Public GPT-2 Small checkpoints exist, but a complete, inspectable pretraining stack—architecture with reference-logit parity, a documented public corpus pipeline, packed uint16 shards, a frozen single-GPU recipe, and an in-repo evaluation suite—is still useful for education and reverse engineering.

basikGPT-1 was trained from scratch with three constraints:

- **Architecture fidelity.** The decoder matches GPT-2 Small: 12 Pre-Norm blocks, 12 heads, `d_model` 768, learned absolute positions, LayerNorm, GPT-2 GELU, tied embeddings, and biases. Official `openai-community/gpt2` weights load through the conversion path and match logits within the documented tolerance.
- **Chinchilla-near unique data for the first stage.** Hoffmann et al. suggest on the order of 20 tokens per parameter. v1.0 used 2.50B tokens for 124.4M unique parameters, seen once over FineWeb-Edu (`sample-10BT`).
- **Single 24–32 GB GPU.** Sequence length 1024, micro-batch 8, gradient accumulation 8, BF16, and SDPA keep measured allocation near 9.5 GiB.

The second stage is a documented continuation, not a second from-scratch run. An earlier mix sketch that included SmolLM `python-edu` was **not** executed; v1.1 used FineWeb + OpenWebMath only.

---

## 3. Related work

The backbone follows GPT-2 [Radford et al., 2019]: Pre-Norm residual blocks, learned absolute positional embeddings, causal multi-head self-attention, and a tied embedding / LM head. The training stack is modernized (AdamW, cosine decay, BF16, PyTorch SDPA) rather than a TensorFlow 1.x / WebText replica. Fidelity tiers are spelled out in [`docs/pretraining_recipe.md`](pretraining_recipe.md).

The first-stage token budget follows the Chinchilla compute-optimal ratio of roughly 20 tokens per parameter [Hoffmann et al., 2022]. The executed v1.0 ratio is 20.09 tokens per parameter for one pass over 2.5B FineWeb-Edu tokens.

Pretraining data is FineWeb-Edu [Penedo et al. / HuggingFaceFW] for v1.0, then FineWeb and OpenWebMath [Paster et al.] for the continuation. Details and licenses are in [§6](#6-data) and [§11](#11-intended-use-limitations-and-licenses).

On the evaluation side, official GPT-2 Small is the same-architecture reference. Pythia [Biderman et al., 2023], SmolLM2, and Qwen2.5-0.5B are included under one in-repo protocol (`english-lm-suite-v1`) so size and data language can be compared without mixing published paper numbers. Token counts and architectures are **not** matched. This is a baseline under a shared scoring rule, not a fair compute bake-off.

---

## 4. Model

Preset `gpt2_small` in `src/basikgpt/config.py`. GPT-2 causal decoder: token embedding, learned positional embedding, twelve Pre-Norm Transformer blocks, final LayerNorm, then an LM head. The LM-head weight is the embedding matrix (`tie_word_embeddings=true`), so the 50,257 × 768 table is counted once (**38,597,376** unique parameters). A parameter breakdown is in the [appendix](#a1-unique-parameter-breakdown).

| Field | Value |
| --- | --- |
| Unique parameters | 124,439,808 |
| `vocab_size` | 50,257 |
| `d_model` (`hidden_size`) | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `head_dim` | 64 |
| `d_ff` (`intermediate_size`) | 3,072 |
| `context_length` | 1,024 |
| LayerNorm `eps` | 1e-5 |
| Attention / MLP / LayerNorm bias | true (`lm_head` bias false) |
| `tie_word_embeddings` | true |
| Activation | GELU tanh approximation |
| Position encoding | learned absolute |
| Attention | causal multi-head self-attention (not GQA) |
| Training sequence length | **1024** |
| Training dropout | **0.0** (`GPTConfig` default is 0.1; pretraining CLIs override) |

**Why these choices.** The project freezes the 2019 GPT-2 Small topology until reference parity is verified. Residual projections use the GPT-2 scaled init `std = 0.02 / sqrt(2 * n_layers)`. Attention is scaled dot-product with scale `1/sqrt(64)`. Training uses the SDPA backend; an eager path exists for verification.

**Context length.** Positions are allocated and trained only at 1024. There is no RoPE table and no unused longer-context reservation.

---

## 5. Tokenizer

GPT-2 byte-level BPE via `tiktoken.get_encoding("gpt2")`. Vocabulary size 50,257. End-of-text id **50,256**. No custom tokenizer.

Training ingest uses `encode_ordinary()` on document text (so a literal `<|endoftext|>` in the page is ordinary bytes) and appends one EOT as the document boundary. Manifests record this as `special_token_policy: encode_ordinary + appended EOT`. The training-machine tiktoken version was `0.14.0`.

Hub exports ship the official GPT-2 tokenizer files next to the `GPT2LMHeadModel` safetensors, so `transformers.AutoTokenizer` loads the same BPE.

---

## 6. Data

Hub streams use the repository pipeline (`scripts/prepare_fineweb_edu.py`, `scripts/prepare_hf_corpus.py`) with a token budget. Documents are tokenized, packed into **uint16** `.npy` shards targeting 1,000,000 tokens each, and checksummed with SHA-256. The train/validation split is `sha256-hash-bucket-v1` (salt `basikgpt-fineweb-edu-v1`). Shards are read sequentially (`--no-shuffle`) so the executed prefix is reproducible.

### 6.1 v1.0 mix (`main_2p5b`)

Built from `HuggingFaceFW/fineweb-edu` `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, license **ODC-By 1.0**. Local shards: `data/fineweb-edu-2p5b/`.

| Manifest field | Value |
| --- | --- |
| Train / validation documents | 2,421,794 / 5,007 |
| Train / validation tokens | 2,499,999,466 / 4,986,319 |
| Train / validation shards | 2,500 / 5 |
| Packed train sequences (T=1024) | 2,440,000 |
| Discarded train tail tokens | 1,436,966 |
| Validation fraction | 0.005 |

v1.0 requested 2,500,000,000 tokens and executed **2,500,001,792** (+1,792 overshoot; 38,147 × 65,536).

### 6.2 v1.1 continuation mix (`cont_5b_mix`)

Lifetime mix after this stage: FineWeb-Edu **50%** + FineWeb **45%** + OpenWebMath **5%**. The continuation itself is FineWeb 2.25B + OpenWebMath 0.25B, interleaved offline as **1 OpenWebMath shard + 9 FineWeb shards per cycle**, then a FineWeb tail (`math1_fineweb9`). Validation remains the FineWeb-Edu holdout from v1.0.

| Source | Hub | Revision | Tokens in this stage |
| --- | --- | --- | --- |
| FineWeb | `HuggingFaceFW/fineweb` `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` | 2,249,995,296 (2,250 shards) |
| OpenWebMath | `open-web-math/open-web-math` | `fde8ef8de2300f5e778f56261843dab89f230815` | 249,999,979 (250 shards) |
| **Stage train total** | | | **2,499,995,275** |

A draft plan that added SmolLM `python-edu` at 10% was **not** run. There is no code slice in v1.1.

The 0.25B math slice exists so equations are not unseen. It is not enough data to claim a math model, and GSM8K is not in the evaluation protocol.

---

## 7. Training

Config freeze: [`configs/gpt2_small_fineweb_edu_single_gpu.json`](../configs/gpt2_small_fineweb_edu_single_gpu.json) (provisional). `scripts/train.py` does not load that JSON; production used equivalent CLI flags. Candidate A (`compile=false`, B=8, G=8) is the canonical recipe.

Tokens per optimizer step stay `8 × 8 × 1024` = **65,536**.

### 7.1 Stage v1.0

| Item | Value |
| --- | --- |
| `max_steps` | 38,147 |
| Token budget (executed) | 2,500,001,792 |
| `sequence_length` | 1024 |
| `micro_batch_size` × `gradient_accumulation_steps` | 8 × 8 |
| Optimizer | AdamW |
| `learning_rate` / `min_lr` | 6e-4 / 6e-5 |
| Warmup / schedule | 2,000 linear warmup, then cosine |
| `betas` / `eps` | [0.9, 0.95] / 1e-8 |
| `weight_decay` | 0.1 on rank-2 matrices (124,318,464 params); 0 on 1D (121,344) |
| `max_grad_norm` | 1.0 |
| `precision` | bf16 |
| `sdpa_kernel` | auto |
| `compile` | false |
| `seed` | 1337 |
| `eval_interval` / `eval_tokens` | 1,526 / 131,072 |
| Checkpoint steps | 1,526, 7,630, 15,259, 38,147 |

AdamW groups skip tied-parameter duplicates so `wte` and `lm_head` are not decayed twice.

### 7.2 Stage v1.1

Resumed from `runs/main_2p5b/step-00038147.pt` (weights and optimizer). `schedule_origin_step` 38,147. First resume used `--reset-data-index` so the new mix starts at sample 0.

| Item | Value |
| --- | --- |
| Final step | 76,294 |
| Lifetime tokens | 5,000,003,584 (+3,584 overshoot) |
| This-stage steps / tokens | 38,147 / 2,500,001,792 |
| LR | rewarm 6e-5 → 3e-4 over 1,000 steps, then cosine to 6e-5 |
| Other optimizer / batch fields | same as v1.0 |
| `seed` | 1337 |

**One FineWeb-Edu epoch, then a different mix.** The continuation leaves the Edu distribution on purpose. FineWeb-Edu validation CE rising during v1.1 is expected, not a silent regression.

---

## 8. Compute

| Field | v1.0 | v1.1 (this stage) |
| --- | --- | --- |
| GPU | NVIDIA RTX PRO 4500 Blackwell | same |
| VRAM | 33,685,569,536 bytes (~31.37 GiB) | same |
| PyTorch / CUDA (torch) / driver | 2.8.0+cu128 / 12.8 / 580.159.04 | same |
| Cloud | RunPod | RunPod |
| Wall-clock (s) | 29,462.59 | 29,593.20 |
| GPU hours | 8.1841 | 8.2203 |
| Training-only tok/s | 85,076 | ~84,700 |
| Peak CUDA allocated (MiB) | 9,523.61 | 9,528.69 |

The v1.1 `summary.json` field `training_only_tokens_per_sec` is **169,416**. That divides **lifetime** 5.0B tokens by this-stage train time and is not a throughput measurement. Stage tokens 2,500,001,792 / `train_elapsed_seconds` 29,513.24 s ≈ **84,708** tok/s, in line with v1.0.

Peak allocation stayed near 9.5 GiB, so a 24 GB card has headroom at this batch shape. Hub ingest and shard packing added wall-clock before step 1 and are not in the GPU-hour totals.

---

## 9. Language-model results

After `train.py`, `scripts/write_whitepaper_snapshot.py` wrote copy-ready tables for v1.0 from `training_log` / `metrics.jsonl`. Numbers below come from that snapshot and from post-training evaluation JSON. **In-loop validation uses 131,072 tokens**, a subset of packed validation, not the full val split.

Uniform-over-vocab reference: ln(50,257) ≈ **10.8249**. Train loss starts near that line (10.9094 at step 1) and falls to 3.28.

### 9.1 v1.0 training curve

| Metric | Value | Step |
| --- | --- | --- |
| first train loss | 10.9094 | 1 |
| last train loss | 3.2830 | 38,147 |
| min in-loop val CE / PPL | 3.3052 / 27.2551 | 36,624 |
| full val CE / PPL | 3.2548 / 25.9151 | 38,147 (post-hoc) |
| tokens processed | 2,500,001,792 | |
| wall time (s) | 29,462.59 | |

### 9.2 v1.0 checkpoint ladder (post-hoc)

Full FineWeb-Edu validation and HellaSwag validation were measured **after** training on the four numbered checkpoints, not inside the training loop. `step-final.pt` matches step 38,147.

| Step | Tokens (approx.) | Full val CE / PPL | HellaSwag acc_raw | HellaSwag acc_norm |
| --- | ---: | ---: | ---: | ---: |
| 1,526 | 100M | 4.701 / 110.05 | 26.38% | 25.05% |
| 7,630 | 500M | 3.660 / 38.86 | 26.76% | 27.16% |
| 15,259 | 1B | 3.479 / 32.43 | 27.20% | 27.71% |
| 38,147 | 2.5B | 3.255 / 25.92 | 28.10% | **29.33%** |

The later `english-lm-suite-v1` re-score of the same 2.5B checkpoint reports HellaSwag `acc_norm` **29.40%** (2,952 / 10,042). The 29.33% figure is the earlier standalone HellaSwag dump (`hellaswag-step-00038147.json`). Section 10 uses the suite number as the protocol official score.

### 9.3 v1.1 training curve

Validation is still FineWeb-Edu while train leaves Edu.

| Metric | Value | Step |
| --- | --- | --- |
| first train loss (this stage) | 3.8090 | 38,150 |
| last train loss | 3.5349 | 76,294 |
| min in-loop val CE / PPL | 3.3214 / 27.6990 | 38,150 |
| final in-loop val CE | 3.4710 | 76,294 |
| lifetime tokens | 5,000,003,584 | |

The Edu val CE rise 3.32 → 3.47 is the expected distribution shift. v1.1 has no post-hoc full-val / HellaSwag-step JSON in `runs/cont_5b_mix/`; downstream scores for that checkpoint live only in [`benchmarks/`](../benchmarks/).

---

## 10. English LM suite

After pretraining, both checkpoints were scored zero-shot with protocol **`english-lm-suite-v1`**. Splits, prompts, and scoring formulas are frozen in [`benchmarks/REPORT.md`](../benchmarks/REPORT.md) and `src/basikgpt/evaluation/`. lm-eval-harness is not a dependency. Published numbers from other papers are not mixed in.

Tokenizers and pretraining data differ across models. **This is a baseline under a shared protocol.**

| Task | Split | Primary metric | n | Chance (not subtracted) |
| --- | --- | --- | ---: | --- |
| HellaSwag | validation | acc_norm (mean completion LL) | 10,042 | 25% |
| LAMBADA (OpenAI) | test | last-word greedy accuracy | 5,153 | open-vocab |
| PIQA | validation (`baber/piqa`) | acc_norm | 1,838 | 50% |
| WinoGrande | validation (`winogrande_xl`) | acc_raw | 1,267 | 50% |
| ARC-Easy | test | acc_norm | 2,376 | 1/N (typically 25%) |

Multiple-choice scoring encodes context and `" " + ending` separately, concatenates, left-truncates context if needed, and scores **choice tokens only**. `acc_raw` is sum log-likelihood; `acc_norm` is mean log-likelihood. LAMBADA splits on the last space and requires a greedy token match on the whole last word.

Two forward paths share prompts and argmax rules: the GPT-2 path (basikGPT `.pt` and official `gpt2`, tiktoken `gpt2`) and `AutoModelForCausalLM` for SmolLM2 / Pythia / Qwen. Tokenization is not matched, so scores are protocol-comparable, not matched-token perplexity.

Checkpoints: v1.0 `runs/main_2p5b/step-00038147.pt`; v1.1 `runs/cont_5b_mix/step-00076294.pt`. Intermediate 100M / 500M / 1B checkpoints are not in this suite.

| Model | Params | Corpus | HS acc_norm | LAMBADA | PIQA | WG | ARC-E |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| **basikGPT-1 v1.0** | 124M | FineWeb-Edu 2.5B | **29.40%** | **19.58%** | **61.37%** | **50.51%** | **43.01%** |
| **basikGPT-1 v1.1** | 124M | Edu 2.5B + FineWeb 2.25B + OpenWebMath 0.25B | **28.75%** | **23.05%** | **61.75%** | **50.83%** | **38.51%** |
| `openai-community/gpt2` | 124M | WebText | 30.37% | 30.93% | 62.57% | 51.62% | 38.13% |
| SmolLM2-135M | 135M | SmolLM2 mix | 42.67% | 42.97% | 67.57% | 51.93% | 59.43% |
| SmolLM2-360M | 362M | SmolLM2 mix | 55.23% | 53.25% | 71.71% | 54.14% | 66.75% |
| Pythia-160M | 162M | The Pile | 29.26% | 11.57% | 58.32% | 49.49% | 34.22% |
| Pythia-410M | 405M | The Pile | 39.18% | 47.33% | 67.68% | 51.14% | 45.12% |
| Qwen2.5-0.5B | 494M | Qwen2.5 mix | 51.26% | 51.99% | 70.18% | 55.64% | 57.83% |

WG is acc_raw; other columns are the suite primary metric.

![english-lm-suite-v1 grouped comparison](whitepaper/figures/grouped.png)

![HellaSwag acc_norm vs parameter count](whitepaper/figures/hellaswag_vs_size.png)

**Reading the scores.**

- **v1.1 vs v1.0.** LAMBADA **+3.47 pp** (19.58 → 23.05): FineWeb prose moved last-word prediction in the intended direction. ARC-Easy **−4.50 pp** (43.01 → 38.51): the Edu-aligned science-question lead faded. HellaSwag −0.65 pp. PIQA and WinoGrande moved by less than 0.4 pp.
- **vs official GPT-2 Small.** Same decoder, same tiktoken, same completion NLL. v1.0 is the only checkpoint that beats gpt2 on a primary metric: ARC-Easy **+4.88 pp**. LAMBADA stays well below gpt2 (v1.0 −11.35 pp, v1.1 −7.88 pp). HellaSwag is slightly below (v1.0 −0.97 pp, v1.1 −1.62 pp).
- **HellaSwag ~29%.** Above chance 25%, in the same band as gpt2 and Pythia-160M, and far below SmolLM2-135M at 42.67%. Nearby parameter counts do not imply nearby data budgets.
- **WinoGrande.** All eight models sit in 49.5–55.6%. At n=1,267 the standard error of a 50% rate is about 1.4 pp, so 50.51% and 50.83% are not distinguishable from chance.
- **Size ladder.** SmolLM2-360M and Qwen2.5-0.5B sit clearly above the 124M GPT-2 class. That is the expected mix-and-scale gap, not a surprise.

**What these numbers do not claim.** The hyperparameter search is not finished. The table is not a token-matched or compute-matched bake-off. “We beat GPT-2” is false as a headline. 2.5B and 5B are **training tokens**, not parameters. 0.25B OpenWebMath does not imply GSM8K (not in the protocol). KoBEST, MMLU, HumanEval, and WikiText PPL are not in the suite.

Published scores: [`benchmarks/REPORT.md`](../benchmarks/REPORT.md) and [`benchmarks/summary.json`](../benchmarks/summary.json). Plots regenerate with `python scripts/plot_lm_suite_compare.py`.

---

## 11. Intended use, limitations, and licenses

### Intended use

basikGPT-1 is a pretrained English **base** checkpoint for research, education, further pretraining, and fine-tuning. It is not a chatbot and was not instruction-tuned. It is not safe for open-ended production chat.

Architecture and tokenizer are GPT-2 compatible. `transformers.AutoModelForCausalLM.from_pretrained` **does** load the Hub export:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("project-iconik/basikGPT-1-v1.1")
model = AutoModelForCausalLM.from_pretrained("project-iconik/basikGPT-1-v1.1")
```

Native `.pt` checkpoints load through the `basikgpt` package. Hub snapshots are `GPT2LMHeadModel` safetensors plus official GPT-2 tokenizer files. Optimizer state is not included.

### Limitations

- 124M / 2.5B–5B tokens cannot match modern 135M models trained on much larger mixes.
- The mix is English-centric web text plus a small math slice. No books-only corpus, dialogue, instruction data, or preference tuning.
- In-loop validation CE/PPL is a 131,072-token subset, not the full packed validation split.
- FineWeb-Edu / FineWeb streaming used a sequential prefix (`--no-shuffle`), not a random sample of the full crawl.
- PII handling is whatever the upstream FineWeb / FineWeb-Edu / OpenWebMath pipelines already applied.
- Training context is 1024 tokens.
- v1.1 FineWeb-Edu val CE is worse than v1.0 because the val set stayed Edu.
- Free generation was not archived for this document.

### Licenses

Code and exported weights are **Apache-2.0**. Dataset cards still apply to training data. Check each card before redistribution:

| Source | License note |
| --- | --- |
| FineWeb-Edu | ODC-By 1.0 |
| FineWeb | ODC-By 1.0 |
| OpenWebMath | see Hub dataset card |
| GPT-2 tokenizer / architecture | follows the public GPT-2 artifacts |

This document does not choose a new license.

---

## 12. Reproducibility

| Step | Path |
| --- | --- |
| Architecture / config | `src/basikgpt/config.py` (`gpt2_small`) |
| Frozen single-GPU JSON | `configs/gpt2_small_fineweb_edu_single_gpu.json` |
| FineWeb-Edu ingest | `scripts/prepare_fineweb_edu.py` → `data/fineweb-edu-2p5b/` |
| HF corpus ingest | `scripts/prepare_hf_corpus.py` |
| Mix interleave | `scripts/combine_shards.py` → `data/mix_5b_cont/` |
| v1.0 train | `python scripts/train.py` with the CLI in [`docs/main_2p5b.md`](main_2p5b.md) |
| v1.1 train | resume from `runs/main_2p5b/step-00038147.pt`; see `runs/cont_5b_mix/run.json` |
| Full val CE/PPL | `scripts/evaluate.py` |
| HellaSwag (standalone) | `scripts/evaluate_hellaswag.py` |
| English suite | `scripts/evaluate_lm_suite.py` |
| HF export | `scripts/export_hf.py` |
| Snapshot tables | `scripts/write_whitepaper_snapshot.py` |
| Suite figures | `scripts/plot_lm_suite_compare.py` |

v1.0 train CLI:

```bash
python scripts/train.py \
  --model-preset gpt2_small --device cuda --precision bf16 \
  --batch-size 8 --grad-accum-steps 8 \
  --target-tokens 2500000000 \
  --warmup-steps 2000 --lr 6e-4 --min-lr 6e-5 \
  --eval-at-start --eval-tokens 131072 --eval-interval 1526 \
  --log-interval 10 \
  --checkpoint-steps 1526,7630,15259,38147 \
  --no-shuffle --track-data-index \
  --data-dir data/fineweb-edu-2p5b \
  --output-dir runs/main_2p5b
```

Suite:

```bash
python scripts/evaluate_lm_suite.py --checkpoint runs/main_2p5b/step-00038147.pt
python scripts/evaluate_lm_suite.py --checkpoint runs/cont_5b_mix/step-00076294.pt --model-id basikgpt-5b
python scripts/evaluate_lm_suite.py --protocol-all --device cuda
python scripts/plot_lm_suite_compare.py
```

Large shards under `data/` and `.pt` checkpoints under `runs/` are gitignored. Production 2.5B ingest needs disk space and a Hub stream; it is not a laptop default.

Recorded git SHAs (both dirty):

| Artifact | Commit |
| --- | --- |
| v1.0 train / post-hoc val | `95e63c325591a96c1a71a288f03742049a589d04` |
| v1.1 train / english-lm-suite-v1 | `ff8b2c0284668c3333d268b27864460e2b1db5f7` |

A dirty tree means the SHA is provenance, not a bitwise recipe lock.

---

## 13. Conclusion

basikGPT-1 is a complete GPT-2 Small pretraining run: a verified 124,439,808-parameter decoder, GPT-2 BPE, a documented FineWeb-Edu 2.5B stage (20.09 tokens/parameter, 8.18 GPU hours, full-val PPL 25.92), a documented FineWeb+OpenWebMath continuation to 5B, and an in-repo zero-shot English suite.

v1.0 sits next to official gpt2 on HellaSwag and above it on ARC-Easy; LAMBADA remains the largest gap. v1.1 closes part of that LAMBADA gap and gives back the ARC-Easy lead. Same-size public decoders in this protocol are close; 135M–0.5B models trained on larger modern mixes sit higher. That is the expected size-and-data ladder, not a surprise.

---

## 14. References

- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
- Biderman, S., et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* arXiv:2304.01373.
- HuggingFaceFW. *FineWeb-Edu* (`sample-10BT`). https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- HuggingFaceFW. *FineWeb.* https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Paster, K., et al. *OpenWebMath.* https://huggingface.co/datasets/open-web-math/open-web-math
- OpenAI. *GPT-2 Small.* https://huggingface.co/openai-community/gpt2
- HuggingFaceTB. *SmolLM2.* https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Alibaba. *Qwen2.5-0.5B.* https://huggingface.co/Qwen/Qwen2.5-0.5B
- Zellers, R., et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* ACL.
- Paperno, D., et al. (2016). *The LAMBADA dataset.*
- Bisk, Y., et al. (2020). *PIQA: Reasoning about Physical Commonsense in Natural Language.*
- Sakaguchi, K., et al. (2020). *WinoGrande.*
- Clark, P., et al. (2018). *Think you have Solved Question Answering? Try ARC.*
- basikGPT-1 v1.0 weights. https://huggingface.co/project-iconik/basikGPT-1-v1.0
- basikGPT-1 v1.1 weights. https://huggingface.co/project-iconik/basikGPT-1-v1.1
- basikGPT code. https://github.com/project-iconik/basikGPT

Suite protocol: [`benchmarks/REPORT.md`](../benchmarks/REPORT.md). Machine-readable rollup: [`benchmarks/summary.json`](../benchmarks/summary.json).

```
@software{basikgpt,
  title = {basikGPT},
  author = {basikGPT Contributors},
  url = {https://github.com/project-iconik/basikGPT},
  year = {2026}
}
```

---

## Appendix

### A.1 Unique parameter breakdown

Tied input embedding / LM head is counted once. Linear and LayerNorm biases are included except `lm_head` (bias false).

| Block | Count |
| --- | ---: |
| Token embedding 50,257 × 768 (tied with LM head) | 38,597,376 |
| Position embedding 1,024 × 768 | 786,432 |
| 12 × attention (Q/K/V/O 768×768 + bias) | 28,348,416 |
| 12 × MLP (768↔3072 + bias) | 56,669,184 |
| 12 × 2 LayerNorm (768+768) + final LayerNorm | 38,400 |
| **Unique total** | **124,439,808** |

Untied total would be 163,037,184. Measured unique parameter count matches this table.
