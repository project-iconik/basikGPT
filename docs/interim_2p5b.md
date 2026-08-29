# 중간 리포트: FineWeb-Edu 2.5B 본런과 영어 스위트

5B 연속학습을 시작하기 **전** 스냅샷이다. 5B 결과·추가 벤치 숫자는 넣지 않는다.

숫자 출처: [`docs/main_2p5b.md`](main_2p5b.md), [`runs/main_2p5b/WHITEPAPER.md`](../runs/main_2p5b/WHITEPAPER.md), [`benchmarks/summary.json`](../benchmarks/summary.json). 스위트 해석의 정적본은 [`docs/english_lm_suite_analysis.md`](english_lm_suite_analysis.md)이다.

## 어떻게 진행됐는지

교육용 GPT-2 Small(124,439,808 unique parameters)을 마일스톤 파이프라인으로 조립한 뒤, 단일 GPU에서 FineWeb-Edu 본런을 돌렸다.

| 항목 | 값 |
|---|---|
| 코퍼스 | `HuggingFaceFW/fineweb-edu` `sample-10BT` revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| 레시피 | BF16, SDPA auto, `compile=false`, B=8, T=1024, G=8, 65,536 tokens/step |
| LR | peak `6e-4` → min `6e-5`, warmup 2,000, cosine |
| Step / 토큰 | 38,147 / **2,500,001,792** |
| GPU | NVIDIA RTX PRO 4500 Blackwell |
| 벽시계 | 29,462.59 s (8.18 GPU-hours) |
| Training-only tok/s | 85,076 |
| Peak allocated | 9.52 GiB |
| tokens / param | ≈ 20.09 |

체크포인트: `runs/main_2p5b/step-00038147.pt` (로컬, git 제외).

Full validation CE / PPL은 **3.255 / 25.92**. 학습 루프 안 val은 131,072 토큰만 본다. HellaSwag `acc_norm`은 체크포인트마다 학습 **후**에 측정했다.

| Step | 토큰 (대략) | Full val CE / PPL | HellaSwag acc_norm |
|---|---|---|---|
| 1,526 | 100M | 4.701 / 110.05 | 25.05% |
| 7,630 | 500M | 3.660 / 38.86 | 27.16% |
| 15,259 | 1B | 3.479 / 32.43 | 27.71% |
| 38,147 | 2.5B | 3.255 / 25.92 | 29.33% |

이어서 같은 split·같은 채점으로 영어 스위트를 돌렸다. 문헌 숫자는 섞지 않았다. 비교 모델 7개: `basikgpt-2p5b`, 공식 `gpt2`, SmolLM2-135M/360M, Pythia-160m/410m, Qwen2.5-0.5B. OpenELM은 transformers 비호환으로 제외. 프로토콜은 [`benchmarks/REPORT.md`](../benchmarks/REPORT.md).

## 벤치에서 부족했던 점

공식 GPT-2 Small(같은 124M 디코더, 같은 tiktoken + completion NLL) 대비.

| 과제 | basikGPT | gpt2 | 델타 | 해석 |
|---|---|---|---|---|
| HellaSwag acc_norm | 29.40% | 30.37% | −0.97pp | 동 아키텍처에서 거의 동일 |
| LAMBADA accuracy | 19.58% | 30.93% | **−11.35pp** | 가장 큰 격차 — 서사 last-word |
| PIQA acc_norm | 61.37% | 62.57% | −1.20pp | 둘 다 우연(50%) 위 |
| WinoGrande acc_raw | 50.51% | 51.62% | −1.11pp | 둘 다 우연과 구분 안 됨 |
| ARC-Easy acc_norm | 43.01% | 38.13% | **+4.88pp** | 유일한 우위 — 교육 코퍼스와 정합 |

- **LAMBADA.** FineWeb-Edu는 교육 웹이지 WebText 같은 소설·포럼 믹스가 아니다. cosine은 이미 min LR에 닿아 있어서, FineWeb-Edu만 더 넣어도 이 격차를 거의 못 좁힌다.
- **HellaSwag / PIQA.** gpt2와 거의 동률이다. 우연 위이지만, 비슷한 크기인 SmolLM2-135M은 HS 42.67%(우리보다 +13.27pp)다. 파라미터 124M이 아니라 **데이터와 레시피** 쪽 점수다. WebText는 수십 B, Pythia는 The Pile ~300B, 현대 소형은 자체 믹스다. 이 표는 컴퓨트 매칭이 아니다.
- **WinoGrande.** n=1,267에서 50%의 표준오차는 약 1.4pp다. 50.51%는 우연과 구분되지 않는다. 이 크기에서 대명사 해소는 거의 안 열린다.
- **스위트에 없는 것.** GSM8K, HumanEval, MMLU. 수학·코드 능력은 아직 측정하지 않았다.

## 잘 된 점

ARC-Easy는 gpt2를 이긴 유일한 primary 지표다. 과학 상식 문항과 교육 코퍼스가 맞는다. Pythia-160M(HS 29.26%, LAMBADA 11.57%, ARC-E 34.22%)보다 LAMBADA·ARC가 높다. 학습이 실패한 것이 아니라 **코퍼스 편향**이다.

이 숫자로 하지 않는 주장: 최적 하이퍼파라미터 탐색이 끝났다, “GPT-2를 이겼다”, 토큰-매칭 공정 비교. 2.5B는 학습 토큰이다.

## 그래서 추가로 하는 것

처음부터 5B를 다시 돌리지 않는다. `step-00038147.pt`에서 가중치·옵티마이저를 잇고, 추가 **2.5B 토큰**만 학습한다.

FineWeb-Edu는 **더 넣지 않는다.** 추가 2.5B 믹스:

| 소스 | 추가 토큰 | 비율 | 역할 |
|---|---|---|---|
| FineWeb (Edu 필터 없음) `sample-10BT` | 2.0B | 80% | 일반 웹. 앞 2.5B의 Edu 치우침을 뉴스·포럼·일반 페이지로 보완. LAMBADA형 산문에 Edu 추가보다 유리. |
| OpenWebMath | 0.25B | 10% | 웹 수학 도메인 주입 |
| SmolLM `python-edu` | 0.25B | 10% | 교육용 Python 코드 주입 |

기대: LAMBADA·HellaSwag의 **작은** 개선, ARC 유지. 비기대: SmolLM2-135M 추월, WinoGrande가 우연 위로 열림, GSM8K/HumanEval 점수(이번 연속학습 실행에는 스위트를 넣지 않음).

영어 스위트는 5B가 끝난 뒤 **별도로** 돌린다.
