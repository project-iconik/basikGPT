# English LM suite (zero-shot)

This file is the protocol. Scores below were measured in this repository with the **same
splits, prompts, and scoring formulas**. Published numbers from other papers are not mixed in.

## Tasks

| Task | Split | Primary metric | Also reported |
|---|---|---|---|
| HellaSwag | validation | acc_norm (mean completion log-likelihood) | acc_raw (sum LL) |
| LAMBADA (OpenAI) | test | last-word accuracy (greedy match) | — |
| PIQA | validation (`baber/piqa`) | acc_norm | acc_raw |
| WinoGrande | validation (`winogrande_xl`) | accuracy = acc_raw | acc_norm |
| ARC-Easy | test | acc_norm | acc_raw |

Not in this suite: KoBEST, MMLU, GSM8K, HumanEval, WikiText perplexity.

Chance rates (for calibration only, not subtracted from scores): HellaSwag 25%; PIQA and
WinoGrande 50%; ARC-Easy 1/N choices (typically 4 → 25%). LAMBADA is open-vocab.

## Scoring rules (shared by both forward paths)

Multiple-choice (HellaSwag, PIQA, WinoGrande, ARC-Easy):

1. Encode **context** and each **choice** separately. Choice tokens are `" " + ending`.
2. Concatenate `[context || choice]`. Left-truncate context if the pair exceeds the model's
   context length, keeping at least one context token.
3. Score **choice tokens only**. Logits at position k-1 parameterize token k.
4. **acc_raw** = argmax of sum log-likelihood. **acc_norm** = argmax of mean log-likelihood.

LAMBADA:

1. Split on the last space: prefix / last word (OpenAI / lm-eval convention).
2. Target continuation is `" " + last_word`.
3. Accuracy is 1 iff greedy argmax tokens equal the target token ids on every position.

WinoGrande blank: context is the text **left of `_`**; each completion is
`option + remainder after the blank`.

ARC-Easy context: `Question: {question}\nAnswer:`.

## Comparison set

Token counts and architectures are **not** matched. The table lists parameter size, family,
and training corpus only. All external models are **base** (not Instruct), 0.1B–0.5B.

**Ours:** two basikGPT GPT-2 Small checkpoints (124M parameters; token counts are
tokens seen, not parameters):

- `basikgpt-2p5b` — FineWeb-Edu 2.5B (`runs/main_2p5b/step-00038147.pt`)
- `basikgpt-5b` — same run continued to 5B on FineWeb 2.25B + OpenWebMath 0.25B
  (`runs/cont_5b_mix/step-00076294.pt`)

Intermediate 100M / 500M / 1B / 3.5B checkpoints are not in this suite.

External Hugging Face bases:

- `openai-community/gpt2` — 124M (GPT-2 forward path: tiktoken + existing converter)
- `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolLM2-360M`
- `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m`
- `Qwen/Qwen2.5-0.5B`

Not included: GPT-2 Medium/Large/XL, OPT, Gemma, Llama, OpenELM, TinyLlama, Qwen2.5-1.5B.
lm-eval-harness is not a dependency.

## Two forward paths

| Path | Models | Tokenizer | Forward |
|---|---|---|---|
| GPT-2 | basikGPT `.pt`, official `gpt2` | tiktoken `gpt2` | basikGPT `GPT` logits tensor |
| HF CausalLM | SmolLM2, Pythia, Qwen2.5-0.5B | each model's tokenizer | `AutoModelForCausalLM` `.logits` |

Prompts, length normalization, and argmax rules are the same. Tokenization is not: scores
are comparable as a protocol, not as matched-token perplexity.

## Results

| Model | Params | Family | Corpus | HS acc_norm | HS acc_raw | LAMBADA | PIQA acc_norm | WG acc | ARC-E acc_norm | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| `basikgpt-2p5b` | 124M | GPT-2 Small (basikGPT) | FineWeb-Edu 2.5B tokens | 29.40% | 28.07% | 19.58% | 61.37% | 50.51% | 43.01% | 40.77% |
| `basikgpt-5b` | 124M | GPT-2 Small (basikGPT) | FineWeb-Edu 2.5B + FineWeb 2.25B + OpenWebMath 0.25B | 28.75% | 27.95% | 23.05% | 61.75% | 50.83% | 38.51% | 40.58% |
| `gpt2` | 124M | GPT-2 Small | WebText | 30.37% | 28.95% | 30.93% | 62.57% | 51.62% | 38.13% | 42.72% |
| `SmolLM2-135M` | 135M | SmolLM2 | SmolLM2 (HuggingFaceTB) | 42.67% | 34.97% | 42.97% | 67.57% | 51.93% | 59.43% | 52.91% |
| `SmolLM2-360M` | 360M | SmolLM2 | SmolLM2 (HuggingFaceTB) | 55.23% | 42.14% | 53.25% | 71.71% | 54.14% | 66.75% | 60.22% |
| `pythia-160m` | 160M | Pythia | The Pile | 29.26% | 28.30% | 11.57% | 58.32% | 49.49% | 34.22% | 36.57% |
| `pythia-410m` | 410M | Pythia | The Pile | 39.18% | 33.53% | 47.33% | 67.68% | 51.14% | 45.12% | 50.09% |
| `Qwen2.5-0.5B` | 0.5B | Qwen2.5 | Qwen2.5 mix (Alibaba) | 51.26% | 40.09% | 51.99% | 70.18% | 55.64% | 57.83% | 57.38% |

Avg is the unweighted mean of the five primaries (HS acc_norm, LAMBADA, PIQA acc_norm, WG acc, ARC-E acc_norm). Token counts and architectures are not matched. WinoGrande sits near chance, so it pulls every 124M row toward 50.

![english-lm-suite-v1 unweighted average](../docs/whitepaper/figures/average.png)

Per-task JSON is written locally under `benchmarks/models/` (gitignored). Machine-readable rollup: `summary.json`.

## How to reproduce

```text
python scripts/evaluate_lm_suite.py --checkpoint runs/main_2p5b/step-00038147.pt
python scripts/evaluate_lm_suite.py --checkpoint runs/cont_5b_mix/step-00076294.pt --model-id basikgpt-5b
python scripts/evaluate_lm_suite.py --hf-model openai-community/gpt2
python scripts/evaluate_lm_suite.py --protocol-all --device cuda
```

`--output-dir` defaults to `benchmarks/`. After each model the suite rewrites `summary.json`
and this report so a crash does not drop finished scores.
