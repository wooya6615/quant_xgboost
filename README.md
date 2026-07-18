# AI 퀀트 트레이딩 - 방향성 예측 파이프라인

XGBoost를 활용해 개별 종목의 단기(N일 후) 방향성을 예측하는 실험 프로젝트입니다.
가격/거래량 기반 기술지표만으로 방향성 예측이 가능한지, 단일 종목과 다종목 풀링 두 방식으로 검증했습니다.

## 파일 구조

```
.
├── feature_engineering.py         # 단일 종목 feature 생성 (모멘텀/변동성/거래량/상대강도)
├── train_xgboost.py                # 단순 시간순 80/20 분할 학습 + threshold sweep
├── train_xgboost_wfo.py            # Walk-Forward 학습, fold별 base_rate 비교, 신호 집중도 분석
├── backtest_simulation.py          # 거래비용 반영 백테스트, Buy & Hold / MDD 비교
├── feature_engineering_pooled.py   # 다종목 데이터 풀링 (섹터 구성 변경 가능)
├── train_xgboost_pooled.py         # 날짜 기준 Walk-Forward, ticker를 categorical feature로 학습
└── README.md
```

## 실행 순서

### 1) 가상환경 세팅

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install yfinance pandas numpy xgboost scikit-learn
```

### 2) 단일 종목 파이프라인

```bash
python feature_engineering.py     # nvda_features.csv 생성
python train_xgboost_wfo.py       # walk-forward 학습 + fold별 성능
python backtest_simulation.py     # 거래비용 반영 실전 시뮬레이션
```

`feature_engineering.py`의 `build_feature_dataset()` 기본 인자(`ticker`, `benchmark`, `start`, `horizon`)를 바꿔서 다른 종목/기간으로 실행 가능합니다.

### 3) 다종목 풀링 파이프라인

```bash
python feature_engineering_pooled.py   # pooled_features.csv 생성
python train_xgboost_pooled.py         # 날짜 기준 walk-forward 학습
```

`feature_engineering_pooled.py`의 `DEFAULT_TICKERS` 딕셔너리로 종목 구성을 바꿀 수 있습니다.

## 방법론 요약

- **Feature**: 모멘텀(수익률, RSI, MACD), 변동성(볼린저밴드, ATR, 역사적변동성), 거래량(OBV, 거래량비율), 상대강도(벤치마크 대비 초과수익)
- **Label**: N일 후 수익률이 거래비용 임계값을 넘으면 1, 아니면 0
- **검증**: Walk-Forward + Embargo(라벨 겹침으로 인한 룩어헤드 방지)
- **평가**: 단순 accuracy가 아니라 각 fold의 실제 base_rate(다수클래스 비율) 대비 우위(`vs_base_rate`)로 판단
- **실전 검증**: 거래비용 반영 백테스트로 Buy & Hold 대비 실제 손익/MDD 비교

## 핵심 발견 (요약)

여러 각도로 검증한 결과, **가격/거래량 기반 기술지표만으로 단일 종목의 단기 방향성을 예측하는 접근은 뚜렷한 엣지가 없음**을 확인했습니다.

| 실험 | 결과 |
|---|---|
| 단일종목(NVDA) 방향성 분류 | AUC 0.55 내외, 베이스라인 이긴 fold 6/21 (29%) |
| 신호 신뢰도 vs 시장국면 상관관계 | 상관계수 0.79 — 모델의 "확신"이 판별력보다 시장국면에 좌우됨 |
| 거래비용 반영 백테스트 (NVDA) | 전략 205% vs Buy&Hold 1193% (큰 격차로 열위) |
| 거래비용 반영 백테스트 (삼성전자) | 전략 165% vs Buy&Hold 322% (격차는 좁지만 여전히 열위) |
| 데이터 기간 확장 (2020→2016) | Buy&Hold와의 리스크조정 격차 축소, 그러나 Sharpe 0.26~0.32로 여전히 낮음 |
| 다종목 풀링 (다양한 섹터) | 개선 없음 — `ticker` feature가 가장 중요한 변수로 나타남 (진짜 패턴 대신 종목 구분에 의존) |
| 다종목 풀링 (동일 섹터/반도체) | 오히려 악화 — 종목 간 높은 상관관계로 fold 결과가 더 불안정해짐 |

**결론**: 이 접근(단일/풀링 종목 + 공개 기술지표 + 방향성 분류)은 뚜렷한 엣지 없이 실전 손익에서도 Buy & Hold를 이기지 못함을 확인했습니다.
자세한 실험 로그는 별도 기록 문서 참고.

## 주의사항

- `future_return` 컬럼은 라벨/백테스트 전용이며, 학습 feature로 사용하면 룩어헤드(미래정보 누수)가 됩니다.
- Walk-Forward의 `embargo` 값은 `feature_engineering.py`의 `horizon`과 항상 동일하게 맞춰야 합니다.
- 트리 기반 모델(XGBoost)은 feature 스케일링이 불필요합니다.
