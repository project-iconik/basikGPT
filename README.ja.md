# basikGPT

[English](README.md) · **日本語** · [한국어](README.ko.md)

basikGPT は **事前学習済み GPT-2 Small decoder-only Transformer**（124,439,808 ユニークパラメータ）と、それを学習した PyTorch コードである。**base** チェックポイントであり、instruction-tuned なチャットボットではない。

- 重み: [`project-iconik/basikGPT-1-v1.0`](https://huggingface.co/project-iconik/basikGPT-1-v1.0)（2.5B トークン）、[`project-iconik/basikGPT-1-v1.1`](https://huggingface.co/project-iconik/basikGPT-1-v1.1)（5B トークン）
- ホワイトペーパー: [`docs/whitepaper.md`](docs/whitepaper.md)（[JA](docs/whitepaper.ja.md)、[KO](docs/whitepaper.ko.md)）
- English LM suite: [`benchmarks/REPORT.md`](benchmarks/REPORT.md)

本番ラン `main_2p5b`: 38,147 steps、**2,500,001,792** トークン（約 20.09 tokens/parameter）、系列長 **1024**。継続 `cont_5b_mix` はそのチェックポイントを FineWeb 2.25B + OpenWebMath 0.25B で生涯 **5,000,003,584** トークン（step 76,294）まで進める。

## Quick start

Python 3.12+ と PyTorch 2.1+。CUDA は先に [pytorch.org](https://pytorch.org/get-started/locally/) から PyTorch を入れる。

```bash
git clone https://github.com/project-iconik/basikGPT.git
cd basikGPT
pip install -e ".[dev]"
```

アーキテクチャとトークナイザは GPT-2 互換である。`transformers.AutoModelForCausalLM.from_pretrained` は Hub 出力を **読み込める**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("project-iconik/basikGPT-1-v1.1")
model = AutoModelForCausalLM.from_pretrained("project-iconik/basikGPT-1-v1.1")
```

ネイティブ `.pt` は `basikgpt` パッケージで読む。`scripts/generate.py` はローカルチェックポイント（または公式 `openai-community/gpt2` 向け `--hf-reference`）を取り、Hub id は取らない。

```bash
python scripts/generate.py --checkpoint runs/main_2p5b/step-00038147.pt --prompt "The history of artificial intelligence"
```

| Extra | Install | Use |
| --- | --- | --- |
| `data` | `pip install -e ".[data]"` | tiktoken、FineWeb 取り込み（`datasets`、`pyarrow`） |
| `dev` | `pip install -e ".[dev]"` | テスト、Hub 出力/読込、および `data` |

コアのモデルと学習コードの依存は `torch` と `numpy` のみ。

## Results

ゼロショット English LM suite（`english-lm-suite-v1`）: 全行で同じ split・プロンプト・採点。トークン数とアーキテクチャは **揃えていない**。全表: [`benchmarks/REPORT.md`](benchmarks/REPORT.md)。

| Model | size | HS | LAMBADA | PIQA | WG | ARC-E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **basikGPT-1 v1.0** | 124M | 29.40 | 19.58 | 61.37 | 50.51 | **43.01** |
| **basikGPT-1 v1.1** | 124M | 28.75 | 23.05 | 61.75 | 50.83 | 38.51 |
| openai-community/gpt2 | 124M | 30.37 | 30.93 | 62.57 | 51.62 | 38.13 |
| HuggingFaceTB/SmolLM2-135M | 135M | 42.67 | 42.97 | 67.57 | 51.93 | 59.43 |
| EleutherAI/pythia-160m | 162M | 29.26 | 11.57 | 58.32 | 49.49 | 34.22 |
| chance | | 25 | — | 50 | 50 | ~25 |

WG は acc_raw、他列はスイートの primary metric。本プロトコルの同規模公開デコーダは近く、現代的な 135M ミックスは上。WinoGrande は偶然水準。手法と全比較: [ホワイトペーパー](docs/whitepaper.ja.md)。

v1.0 言語モデル指標（ループ内 val は 131,072 トークン、full val は事後）:

| | |
| --- | --- |
| Tokens | 2,500,001,792 |
| Last train CE | 3.2830 |
| Full val CE / PPL | 3.2548 / 25.9151 |
| Wall time | 29,462.59 s（約 8.18 GPU hours） |
| Training-only tok/s | 85,076 |

```bash
pip install -e ".[dev]"
python scripts/evaluate_lm_suite.py --checkpoint runs/main_2p5b/step-00038147.pt
python scripts/evaluate_lm_suite.py --hf-model openai-community/gpt2
```

## Architecture

GPT-2 因果デコーダ: Pre-Norm、LayerNorm（ε = 1e-5）、学習済み絶対位置、因果的 multi-head self-attention、GELU tanh 近似、Linear と LayerNorm の bias、tied embeddings。詳細: [ホワイトペーパー §4](docs/whitepaper.ja.md#4-モデル)。

| | `gpt2_small` |
| --- | --- |
| Unique parameters | 124,439,808 |
| `vocab_size` | 50,257 |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `head_dim` | 64 |
| `d_ff` | 3,072 |
| `context_length` | 1,024 |
| Training sequence length | **1024** |
| `tie_word_embeddings` | true |
| Training dropout | **0.0**（`GPTConfig` 既定は 0.1） |

`GPTConfig` は `gpt2_medium`、`gpt2_large`、`gpt2_xl` も定義する。それらは設定のみ。このリポジトリが学習するのは `gpt2_small`。

## Tokenizer

GPT-2 byte-level BPE（`tiktoken.get_encoding("gpt2")`）。語彙 50,257。End-of-text id 50,256。独自トークナイザはない。学習取り込みは `encode_ordinary()` のあと文書ごとに EOT を 1 個付与する。Hub 出力は公式 GPT-2 トークナイザファイルを同梱する。詳細: [ホワイトペーパー §5](docs/whitepaper.ja.md#5-トークナイザ)。

## Data

v1.0 は FineWeb-Edu（`sample-10BT`）。v1.1 は FineWeb 2.25B + OpenWebMath 0.25B で継続（生涯ミックス: Edu 50% + FineWeb 45% + OpenWebMath 5%）。Hub ストリームは uint16 `.npy` シャードにパックする。生ダンプとシャードはローカル `data/` にあり git には入らない。

```mermaid
flowchart LR
  raw[Hub_FineWeb]
  shard[tokenize_uint16_shards]
  train[train.py]
  gen[generate.py]
  raw --> shard --> train --> gen
```

ミックス表とライセンス: [ホワイトペーパー §6](docs/whitepaper.ja.md#6-データ)。

## Reproduce

### A. 公開重みを使う（既定）

[Quick start](#quick-start) を見よ。パック済みコーパスは不要。推論に 24 GB GPU は要らない。

### B. FineWeb-Edu 2.5B を再学習する

ディスク数十 GB と Hugging Face Hub ストリームが必要。`scripts/train.py` は凍結 JSON を読まない。本番は同等 CLI フラグを使った。

```bash
python scripts/prepare_fineweb_edu.py \
  --output data/fineweb-edu-2p5b \
  --max-train-tokens 2500000000 \
  --shard-token-target 1000000
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

設定 [`configs/gpt2_small_fineweb_edu_single_gpu.json`](configs/gpt2_small_fineweb_edu_single_gpu.json) は凍結レシピを記録する（provisional、最適主張ではない）。本番のピーク CUDA allocated は **9,523.61 MiB**。

各 `train.py` 起動は `runs/<name>/` に書く。公開された手法と指標は [ホワイトペーパー](docs/whitepaper.ja.md) にある。ステップログ（`metrics.jsonl`）とシャードマニフェスト（`dataset.json`）は gitignore される。

### C. Tiny CPU スモーク（実験のみ）

本番学習セットではない。先にスモーク用シャードが必要。

```bash
python scripts/prepare_fineweb_edu.py --output data/fineweb-edu-smoke
python scripts/train.py --model-preset tiny --max-steps 20 --device cpu --data-dir data/fineweb-edu-smoke
```

## Layout

- `src/basikgpt/model` — GPT-2 バックボーンと因果 LM
- `src/basikgpt/data` — トークナイザ、シャーディング、FineWeb パイプライン
- `src/basikgpt/training` — optimizer、scheduler、trainer、チェックポイント
- `src/basikgpt/generation` — KV キャッシュ生成
- `src/basikgpt/evaluation` — val CE/PPL と English LM suite
- `src/basikgpt/conversion` — Hugging Face GPT-2 入出力
- `scripts` — train、generate、evaluate、prepare、export
- `configs` — 凍結済み単一 GPU JSON
- `docs` — 技術ホワイトペーパー（EN / JA / KO）とレシピメモ
- `benchmarks` — English LM suite プロトコルと点数
- `runs` — 公開 `run.json` / `summary.json`（チェックポイントはローカル）

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Contributing

Issue と pull request を歓迎する。変更を送る前に `pytest tests/ -q` を実行すること。

## License

コードと出力重みは **Apache-2.0**。FineWeb / FineWeb-Edu は **ODC-By 1.0**。OpenWebMath は Hub データセットカードを見よ。再配布前に各カードを確認すること。詳細: [ホワイトペーパー §11](docs/whitepaper.ja.md#11-想定用途限界ライセンス)。
