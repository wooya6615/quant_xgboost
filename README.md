# AI 퀀트 트레이딩 - 방향성 예측 파이프라인

XGBoost를 활용해 개별 종목의 단기(N일 후) 방향성을 예측하는 실험 프로젝트입니다.
가격/거래량 기반 기술지표만으로 방향성 예측이 가능한지 검증했고,
이후 완전히 다른 소스인 수급 데이터(외국인/기관 순매수)를 추가했을 때 개선이 있는지까지 검증했습니다.

## 파일 구조

```
.
├── feature_engineering.py           # 단일 종목 feature 생성 (모멘텀/변동성/거래량/상대강도)
├── train_xgboost.py                  # 단순 시간순 80/20 분할 학습 + threshold sweep
├── train_xgboost_wfo.py              # Walk-Forward 학습, fold별 base_rate 비교, 신호 집중도 분석
├── backtest_simulation.py            # 거래비용 반영 백테스트, Buy & Hold / MDD 비교
├── feature_engineering_pooled.py     # 다종목 데이터 풀링 (섹터 구성 변경 가능)
├── train_xgboost_pooled.py           # 날짜 기준 Walk-Forward, ticker를 categorical feature로 학습
├── feature_engineering_investor.py   # 수급 데이터(외국인/기관 순매수) feature 추가
├── train_xgboost_ablation.py         # BASE/INVESTOR_ONLY/COMBINED 3-way ablation + 멀티 시드 검증
├── backtest_comparison.py            # BASE vs COMBINED 신호 기반 실전 백테스트 비교
└── README.md
```

## 실행 순서

### 1) 가상환경 세팅

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install yfinance pandas numpy xgboost scikit-learn pykrx python-dotenv
```

### 2) 단일 종목 파이프라인 (가격 feature만)

```bash
python feature_engineering.py     # {ticker}_features.csv 생성
python train_xgboost_wfo.py       # walk-forward 학습 + fold별 성능
python backtest_simulation.py     # 거래비용 반영 실전 시뮬레이션
```

### 3) 다종목 풀링 파이프라인

```bash
python feature_engineering_pooled.py   # pooled_features.csv 생성
python train_xgboost_pooled.py         # 날짜 기준 walk-forward 학습
```

### 4) 수급 데이터(외국인/기관 순매수) feature 추가 파이프라인

```bash
python feature_engineering_investor.py   # {ticker}_features_with_investor_h{horizon}.csv 생성
python train_xgboost_ablation.py         # BASE/INVESTOR_ONLY/COMBINED 비교 + 멀티 시드 검증
python backtest_comparison.py            # BASE vs COMBINED 실전 백테스트 비교
```

`feature_engineering_investor.py`의 `TICKER`/`TICKER_KRX`/`HORIZON` 상수로 종목·기간 변경 가능합니다.

**사전 준비 (KRX_ID/KRX_PW):**
```bash
# .env 파일 생성 (git에 커밋 금지 -- .gitignore에 이미 포함돼 있어야 함)
KRX_ID=본인_krx_아이디
KRX_PW=본인_krx_비밀번호
```
pykrx가 2025년 12월 KRX 회원제 전환 이후 로그인 방식으로 바뀌었습니다.
KRX가 비공식 스크래핑 도구(IP)를 차단하는 경우가 있어, 막히면 네트워크를 바꿔서(예: 테더링) 재시도하세요.

## 방법론 요약

- **가격 feature**: 모멘텀(수익률, RSI, MACD), 변동성(볼린저밴드, ATR, 역사적변동성), 거래량(OBV, 거래량비율), 상대강도(벤치마크 대비 초과수익) — 13개
- **수급 feature**: 외국인/기관 순매수 3·5일 누적, 거래대금 대비 순매수 비율(정규화), 외국인·기관 동반매수 여부 — 7개
- **Label**: N일 후 수익률이 거래비용 임계값을 넘으면 1, 아니면 0
- **검증**: Walk-Forward + Embargo(라벨 겹침으로 인한 룩어헤드 방지)
- **평가**: 단순 accuracy가 아니라 각 fold의 실제 base_rate(다수클래스 비율) 대비 우위(`vs_base_rate`)로 판단
- **멀티 시드 검증**: XGBoost의 `random_state`를 5개로 바꿔가며 반복, 방향이 일관되는지 확인 (단일 시드 결과의 우연성 배제)
- **실전 검증**: 거래비용 반영 백테스트로 Buy & Hold 대비 실제 손익/MDD 비교

### 수급 데이터 룩어헤드 방지 (중요)

외국인/기관 순매수는 **당일 장 마감 후(저녁)에 발표**됩니다. 반면 백테스트는 "당일 종가 매매"를 가정하므로,
N일차 row에는 N일 당일 수급이 아니라 **N-1일까지 발표분만** 들어가도록 `feature_engineering_investor.py`에서
1일 shift를 적용했습니다. 이 shift를 빠뜨리면 아직 발표되지 않은 정보로 매매한 것처럼 계산되어 결과가 부풀려집니다.

## 핵심 발견 (요약)

### 1) 가격/거래량 기반 기술지표만으로는 엣지 없음

| 실험 | 결과 |
|---|---|
| 단일종목(NVDA) 방향성 분류 | AUC 0.55 내외, 베이스라인 이긴 fold 6/21 (29%) |
| 신호 신뢰도 vs 시장국면 상관관계 | 상관계수 0.79 — 모델의 "확신"이 판별력보다 시장국면에 좌우됨 |
| 거래비용 반영 백테스트 (NVDA) | 전략 205% vs Buy&Hold 1193% (큰 격차로 열위) |
| 거래비용 반영 백테스트 (삼성전자) | 전략 165% vs Buy&Hold 322% (격차는 좁지만 여전히 열위) |
| 다종목 풀링 (다양한 섹터/동일 섹터) | 개선 없음 — `ticker` feature 의존 또는 fold 불안정성 심화 |

### 2) 수급 데이터(외국인/기관 순매수) 추가 실험

| 실험 | 결과 |
|---|---|
| 삼성전자, horizon=10일, COMBINED vs BASE | AUC 소폭 하락 (-0.010), 개선 없음 |
| 삼성전자, horizon=5일, COMBINED vs BASE (단일 시드) | AUC +0.012, 방향 엇갈림 (검증 부족) |
| 삼성전자, horizon=5일, **5개 시드 반복** | AUC 차이 +0.0147 (표준편차 0.0045), **5/5 시드 COMBINED 우위** |
| 현대로템, horizon=5일, **5개 시드 반복** | AUC 차이 **+0.0272** (표준편차 0.0034), **5/5 시드 COMBINED 우위** — 유동성 낮은 종목에서 효과 더 큼 |
| INVESTOR_ONLY (수급 feature만) | 두 종목 모두 base_rate 못 이김 — 수급 데이터 단독으론 무의미, 가격 feature와 결합해야 의미 있음 |
| 실전 백테스트 (현대로템, threshold=0.55) | BASE 271.9% → **COMBINED 497.2%** (거의 2배), MDD도 개선 (-81.1% → -76.1%) |
| 실전 백테스트 vs Buy & Hold | COMBINED도 Buy & Hold(1236.7%) 못 이김 |

**결론**:
1. 가격 기반 기술지표만으로는 방향성 분류에 뚜렷한 엣지가 없음 (기존 결론 유지)
2. 수급 데이터(외국인/기관 순매수)는 가격 feature와 **결합했을 때** 통계적(AUC)·경제적(백테스트 손익) 개선을 일관되게 가져옴 — 종목 2개(삼성전자/현대로템) × 시드 5개, 총 10회의 독립 실행에서 방향이 흔들리지 않았고, 유동성 가설(저유동성 종목일수록 효과 큼)과도 일치
3. 다만 이 개선은 **상대적**인 것이며, "N일 후 방향성 예측 + 순차 진입" 형태의 타이밍 전략 자체는 여전히 강한 상승장에서 Buy & Hold를 이기지 못하는 구조적 한계를 가짐

자세한 실험 로그는 별도 기록 문서 참고.

## 주의사항

- `future_return` 컬럼은 라벨/백테스트 전용이며, 학습 feature로 사용하면 룩어헤드(미래정보 누수)가 됩니다.
- Walk-Forward의 `embargo` 값은 `feature_engineering.py`의 `horizon`과 항상 동일하게 맞춰야 합니다.
- 트리 기반 모델(XGBoost)은 feature 스케일링이 불필요합니다.
- 수급 데이터는 1일 shift가 필수입니다 (위 "수급 데이터 룩어헤드 방지" 참고).
- pykrx 실행 중 `KRX_ID`/`KRX_PW` 관련 에러나 `JSONDecodeError`가 나면, 자격증명 문제가 아니라 KRX의 IP 차단일 가능성이 높습니다. 네트워크를 바꿔서 재시도하세요.
- 단일 시드 결과만으로 "개선됐다/안됐다"를 판단하지 말 것 — 반드시 멀티 시드(최소 5개) 확인 후 결론 내릴 것.
