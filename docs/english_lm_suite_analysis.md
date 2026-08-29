# 영어 LM 스위트 분석

같은 split·같은 채점으로 다시 측정한 zero-shot 점수. 문헌 숫자는 섞지 않음. 토큰 수·아키텍처는 매칭하지 않음. 우리 모델은 GPT-2 Small 124M 파라미터. 체크포인트 두 개: FineWeb-Edu **2.5B** (`basikgpt-2p5b`)와 이를 이은 **5B** (`basikgpt-5b`, FineWeb 2.25B + OpenWebMath 0.25B). 숫자는 학습 토큰이지 파라미터가 아니다.

출처: [`benchmarks/summary.json`](../benchmarks/summary.json), 프로토콜 [`benchmarks/REPORT.md`](../benchmarks/REPORT.md). 인터랙티브 원본: Cursor 캔버스 `english-lm-suite-analysis.canvas.tsx`.

## 비교가 말해주는 것 / 말해주지 않는 것

공식 gpt2와는 디코더 구조가 같고, SmolLM2·Pythia·Qwen과는 데이터·토크나이저·스케일이 다르다. WebText는 수십 B 토큰, Pythia는 The Pile ~300B, 현대 소형은 자체 믹스다. 이 표는 “공정한 컴퓨트 매칭”이 아니라 동일 프로토콜에서의 위치다. OpenELM, GPT-2 Medium+, MMLU/GSM8K는 스위트에 없다.

## 헤드라인

| 지표 | 값 |
|---|---|
| 5B vs 2.5B LAMBADA | **+3.47pp** (23.05% vs 19.58%) |
| 5B vs 2.5B ARC-Easy | **−4.50pp** (38.51% vs 43.01%) |
| 5B vs 공식 gpt2 LAMBADA | −7.88pp (23.05% vs 30.93%; 2.5B 때는 −11.35pp) |
| 5B HellaSwag acc_norm | 28.75% (2.5B 29.40%, gpt2 30.37%) |

연속학습은 서사 last-word를 조금 올렸고, Edu와 맞던 ARC-Easy 우위를 거의 거둬 갔다. HellaSwag는 소폭 하락. WinoGrande는 여전히 우연과 구분 안 됨.

## 5B vs 2.5B

같은 124M, 같은 채점. 차이는 추가 2.5B 믹스(FineWeb 90% + OpenWebMath 10%).

| 과제 | 2.5B | 5B | 델타 | 해석 |
|---|---|---|---|---|
| HellaSwag acc_norm | 29.40% | 28.75% | −0.65pp | 작은 하락 |
| LAMBADA accuracy | 19.58% | 23.05% | **+3.47pp** | FineWeb 산문이 목적한 방향 |
| PIQA acc_norm | 61.37% | 61.75% | +0.38pp | 노이즈 수준 |
| WinoGrande acc_raw | 50.51% | 50.83% | +0.32pp | 둘 다 우연과 구분 안 됨 |
| ARC-Easy acc_norm | 43.01% | 38.51% | **−4.50pp** | Edu 편향이 옅어짐 |

## 5B vs 공식 GPT-2 Small

둘 다 tiktoken gpt2 + 동일 completion NLL.

| 과제 | 5B | gpt2 | 델타 | 해석 |
|---|---|---|---|---|
| HellaSwag acc_norm | 28.75% | 30.37% | −1.62pp | 2.5B(−0.97pp)보다 격차 조금 큼 |
| LAMBADA accuracy | 23.05% | 30.93% | −7.88pp | 격차는 줄었지만 아직 큼 |
| PIQA acc_norm | 61.75% | 62.57% | −0.82pp | 둘 다 우연(50%) 위 |
| WinoGrande acc_raw | 50.83% | 51.62% | −0.79pp | 둘 다 우연과 구분 안 됨 |
| ARC-Easy acc_norm | 38.51% | 38.13% | +0.38pp | 2.5B의 +4.88pp 우위는 거의 사라짐 |

2.5B vs gpt2 표는 [`benchmarks/REPORT.md`](../benchmarks/REPORT.md)의 `basikgpt-2p5b` 행과 같다.

## 전체 primary 점수

| 모델 | Params | 코퍼스 | HS acc_norm | LAMBADA | PIQA | WG | ARC-E |
|---|---|---|---|---|---|---|---|
| basikGPT 2.5B | 124M | FineWeb-Edu 2.5B tokens | 29.40% | 19.58% | 61.37% | 50.51% | 43.01% |
| basikGPT 5B | 124M | Edu 2.5B + FineWeb 2.25B + OpenWebMath 0.25B | 28.75% | 23.05% | 61.75% | 50.83% | 38.51% |
| gpt2 | 124M | WebText | 30.37% | 30.93% | 62.57% | 51.62% | 38.13% |
| SmolLM2-135M | 135M | SmolLM2 mix | 42.67% | 42.97% | 67.57% | 51.93% | 59.43% |
| SmolLM2-360M | 362M | SmolLM2 mix | 55.23% | 53.25% | 71.71% | 54.14% | 66.75% |
| Pythia-160M | 162M | The Pile | 29.26% | 11.57% | 58.32% | 49.49% | 34.22% |
| Pythia-410M | 405M | The Pile | 39.18% | 47.33% | 67.68% | 51.14% | 45.12% |
| Qwen2.5-0.5B | 494M | Qwen2.5 mix | 51.26% | 51.99% | 70.18% | 55.64% | 57.83% |

WG는 acc_raw, 나머지는 표의 primary metric. n: HS 10,042 · LAMBADA 5,153 · PIQA 1,838 · WG 1,267 · ARC-E 2,376.

우연 수준(보정용, 점수에서 빼지 않음): HellaSwag 25%; PIQA·WinoGrande 50%; ARC-Easy 선택지 수에 따라 1/N(보통 4 → 25%). LAMBADA는 개방 어휘라 단순 우연선이 없다.

## 파라미터 규모 vs HellaSwag acc_norm

왼쪽이 작은 모델. 124M 세 점(2.5B, 5B, gpt2)은 거의 같고, 135M SmolLM2가 그 위를 크게 뛰어넘는다. 파라미터가 비슷해도 학습 토큰과 믹스가 점수를 가른다.

| 모델 | Params (M) | HellaSwag acc_norm |
|---|---|---|
| basikGPT 5B | 124 | 28.75% |
| basikGPT 2.5B | 124 | 29.40% |
| gpt2 | 124 | 30.37% |
| SmolLM2-135M | 135 | 42.67% |
| Pythia-160M | 162 | 29.26% |
| SmolLM2-360M | 362 | 55.23% |
| Pythia-410M | 405 | 39.18% |
| Qwen2.5-0.5B | 494 | 51.26% |

## 과제 성격

**LAMBADA — 서사 last-word.** FineWeb-Edu만으로는 gpt2 30.93% 대비 19.58%(−11.35pp)였다. 일반 FineWeb을 이은 5B는 23.05%(−7.88pp). 방향은 맞았고, WebText 규모를 닫지는 못했다. Pythia-160M 11.57%보다 두 체크포인트 모두 높다.

**WinoGrande — 아직 안 열림.** 전원 49.5–55.6%. n=1,267에서 50%의 표준오차는 약 1.4pp라 2.5B 50.51%와 5B 50.83%는 우연과 구분되지 않는다. 이 크기에서는 대명사 해소가 거의 안 된다.

**ARC-Easy — Edu와 정합이 풀림.** 2.5B는 gpt2를 이긴 유일한 primary 지표였다 (43.01% vs 38.13%). 5B는 38.51%로 gpt2(38.13%)와 동률 근처다. 과학 상식 문항과 교육 코퍼스가 맞았던 이점이, 일반 웹+수학 연속학습 뒤에 거의 사라졌다.

## 이 숫자로 하지 않는 주장

최적 하이퍼파라미터 탐색이 끝났다는 주장, “GPT-2를 이겼다”, 토큰-매칭 공정 비교. 2.5B·5B는 학습 토큰이다. HellaSwag ~29%는 우연(25%) 위이지만 현대 135M(42.67%)과는 데이터·스케일 격차다. 수학 0.25B로 GSM8K가 열렸다는 주장도 하지 않는다(프로토콜에 없음).
