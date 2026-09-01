# Whitepaper snapshot

Run: `runs/main_2p5b`
Status: completed
Git: `95e63c325591a96c1a71a288f03742049a589d04` (dirty=True)

## Abstract

| Field | Value |
|---|---|
| Unique parameters | 124,439,808 |
| Tokens processed | 2,500,001,792 |
| Tokens per parameter | 20.0900 |
| Wall-clock (s) | 29462.59 |
| GPU hours | 8.1841 |
| GPU | NVIDIA RTX PRO 4500 Blackwell |
| Min val CE / PPL | 3.3052 / 27.2551 |
| Min val step | 36,624 |
| In-loop eval tokens | 131,072 |
| HellaSwag acc_norm | 0.2933 |
| Full val CE / PPL | 3.2548 / 25.9151 |

## Model

| Field | Value |
|---|---|
| Unique parameters | 124,439,808 |
| vocab_size | 50,257 |
| hidden_size (d_model) | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| head_dim | 64 |
| intermediate_size (d_ff) | 3,072 |
| max_position_embeddings | 1,024 |
| Training sequence length | 1,024 |
| layer_norm_eps | 1e-05 |
| bias | True |
| tie_word_embeddings | True |
| Token embedding params | 38,597,376 |
| Position embedding params | 786,432 |
| Transformer + final LN | 85,056,000 |

## Training

| Field | Value |
|---|---|
| max_steps | 38,147 |
| Planned executed token count | 2,500,001,792 |
| Token budget (requested) | 2,500,000,000 |
| Overshoot tokens | 1,792 |
| Tokens / optimizer step | 65,536 |
| micro_batch_size × grad_accum | 8 × 8 |
| learning_rate | 0.0006 |
| min_learning_rate | 6e-05 |
| warmup_steps | 2,000 |
| betas / eps | [0.9, 0.95] / 1e-08 |
| weight_decay | 0.1 (decay params 124,318,464; no-decay 121,344) |
| max_grad_norm | 1.0 |
| precision | bf16 |
| seed | 1,337 |
| eval_interval | 1,526 |
| eval_tokens / eval_batches | 131,072 / 16 |
| Uniform-over-vocab CE ln(V) | 10.8249 |

## Compute

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell |
| VRAM (bytes) | 33,685,569,536 |
| PyTorch | 2.8.0+cu128 |
| CUDA (torch) | 12.8 |
| Driver | 580.159.04 |
| bf16 hardware | True |
| Wall-clock (s) | 29462.59 |
| GPU hours | 8.1841 |
| Training-only tok/s | 85076.24 |
| End-to-end tok/s | 84853.42 |
| Peak CUDA allocated (MiB) | 9523.61 |

## Language-model results

| Metric | Value | Step |
|---|---|---|
| first train loss | 10.9094 | 1 |
| last train loss | 3.2830 | 38,147 |
| min val loss | 3.3052 | 36,624 |
| min val perplexity | 27.2551 | 36,624 |
| tokens processed | 2,500,001,792 |  |
| wall time (s) | 29462.59 |  |
| mean tokens/sec | 84853.42 |  |
| peak CUDA allocated (MiB) | 9523.61 |  |
| Uniform-over-vocab reference | 10.8249 |  |

## Data

| Field | Value |
|---|---|
| Repository | HuggingFaceFW/fineweb-edu |
| Config | sample-10BT |
| Revision | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 |
| License | ODC-By 1.0 |
| Tokenizer | gpt2 (vocab 50,257) |
| Train documents | 2,421,794 |
| Validation documents | 5,007 |
| Train tokens | 2,499,999,466 |
| Validation tokens | 4,986,319 |
| Train / val shards | 2,500 / 5 |
| Packed train sequences | 2,440,000 |
| Packed train tokens | 2,499,999,466 |
| Discarded train tail tokens | 1,436,966 |
| Packed val sequences | 4,867 |

