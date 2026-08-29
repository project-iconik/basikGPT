# FineWeb-Edu 2.5B main run

This records the single-GPU 2.5B-token FineWeb-Edu pretraining run on the Milestone 16 frozen recipe. It is **not** an optimal-hyperparameter claim.

Machine-readable artifacts (`run.json`, `summary.json`, evaluation JSON): [`runs/main_2p5b/`](../runs/main_2p5b/). Copy-ready tables: [`WHITEPAPER.md`](../runs/main_2p5b/WHITEPAPER.md). Step logs (`metrics.jsonl`) stay on the training machine and are not in git.

Checkpoints (`.pt`, ~1.4 GiB each) stay on the training machine and are **not** in git.

## Recipe

| Item | Value |
|---|---|
| Model | GPT-2 Small, 124,439,808 unique parameters, dropout 0.0 |
| Precision | BF16, SDPA auto, `compile=false` |
| Batch | B=8, T=1024, G=8, W=1 → 65,536 tokens/step |
| LR | peak `6e-4`, min `6e-5`, warmup 2,000, cosine |
| Data | `HuggingFaceFW/fineweb-edu` `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Train / val tokens | 2,499,999,466 / 4,986,319 (uint16 shards under `data/fineweb-edu-2p5b`, local) |
| Requested budget | 2,500,000,000 |
| Actual tokens | 2,500,001,792 (+1,792 overshoot; 38,147 optimizer steps) |
| GPU | NVIDIA RTX PRO 4500 Blackwell |
| Wall-clock | 29,462.59 s (8.18 GPU-hours) |
| Training-only tok/s | 85,076 |
| Peak allocated | 9.52 GiB |

`scripts/train.py` does not load the canonical JSON. The run used the freeze `logging_protocol` CLI:

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

In-loop `val_loss` covers 131,072 tokens only. Full validation perplexity and HellaSwag `acc_norm` were measured **after** training, on the four numbered checkpoints (not concurrently with the training loop). `step-final.pt` matches step 38,147 and was not evaluated twice.

## Downstream eval (full val + HellaSwag validation)

| Step | Tokens (approx.) | Full val CE / PPL | HellaSwag raw | HellaSwag acc_norm |
|---|---|---|---|---|
| 1,526 | 100M | 4.701 / 110.05 | 26.38% | 25.05% |
| 7,630 | 500M | 3.660 / 38.86 | 26.76% | 27.16% |
| 15,259 | 1B | 3.479 / 32.43 | 27.20% | 27.71% |
| 38,147 | 2.5B | 3.255 / 25.92 | 28.10% | 29.33% |

In-loop min val CE was 3.305 (PPL 27.26) at step 36,624. First train loss at step 1 was 10.909; last train loss 3.283.

JSON per checkpoint: `evaluation-step-XXXXXXXX.json`, `hellaswag-step-XXXXXXXX.json`.
