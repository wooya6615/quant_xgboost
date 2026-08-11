# quant_xgboost
> **최종 결론 (2026-08, triple-barrier 라인)**: `docs/candidate_model_triple_barrier_pooled.md`
> 참고. Triple-barrier 라벨링 + BASE/VALUATION COMBINED + 현대로템/한전기술/모트렉스
> 3종목 풀링 전략을 5-seed/국면검증/체결타이밍버그수정/청산판정개선까지 전부 통과시켰으나,
> **PBO(Probability of Backtest Overfitting) 검증에서 pt_sl 축 95.6%로 심각한 과적합
> 위험이 확인되어 최종적으로 [배포 부적합] 판정.** 채택했던 pt_sl=(2,1)은 구간별
> in-sample 최고 빈도 순위 4/5위에 불과했고, 이미 폐기했던 pt_sl=(1,1)이 가장 자주
> 1등이었음 -- "5-seed를 통과했다"는 것만으로는 하이퍼파라미터 탐색 과정 자체의
> 과적합을 못 잡아낸다는 걸 확인한 사례.
>
> 방법론적 수정(체결 지연 1일, High/Low 기반 청산 판정, 고정 Buy&Hold 비교 구간,
> CSCV/PBO 계산 인프라)은 그 자체로 유효하며 다음 실험에 재사용 가능. 실거래
> 배포는 pt_sl을 사전 고정한 재검증 전까지 보류.

# AI 퀀트 트레이딩 - 방향성 예측 파이프라인

XGBoost를 활용해 개별 종목의 단기(N일 후) 방향성을 예측하는 실험 프로젝트입니다.
가격/거래량 기반 기술지표만으로 방향성 예측이 가능한지 검증했고,
이후 완전히 다른 소스인 수급 데이터(외국인/기관 순매수), 개별종목 공매도 데이터,
밸류에이션(PER/PBR/배당수익률), 환율(USD/KRW), 외국인 보유율/한도소진율을
추가했을 때 개선이 있는지까지 검증했습니다.

## 파일 구조

```
.
├── src/
│   ├── feature_engineering.py             # 단일 종목 feature 생성 (모멘텀/변동성/거래량/상대강도)
│   ├── train_xgboost.py                   # 단순 시간순 80/20 분할 학습 + threshold sweep
│   ├── train_xgboost_wfo.py               # Walk-Forward 학습, fold별 base_rate 비교, 신호 집중도 분석
│   ├── backtest_simulation.py             # 거래비용 반영 백테스트, Buy & Hold / MDD 비교
│   ├── feature_engineering_pooled.py      # 다종목 데이터 풀링 (섹터 구성 변경 가능)
│   ├── train_xgboost_pooled.py            # 날짜 기준 Walk-Forward, ticker를 categorical feature로 학습
│   ├── feature_engineering_investor.py    # 수급 데이터(외국인/기관 순매수) feature 추가, horizon 스윕 지원
│   ├── train_xgboost_ablation.py          # BASE/INVESTOR_ONLY/COMBINED 3-way ablation + 멀티 시드 검증 + horizon 스윕
│   ├── backtest_comparison.py             # BASE vs COMBINED 신호 기반 실전 백테스트 비교
│   ├── feature_engineering_short_kr.py     # 국내 개별종목 공매도(거래량/비중) feature 추가 -- pykrx 청크 분할로 전체 이력 복원
│   ├── train_xgboost_short_ablation.py     # BASE/SHORT_ONLY/COMBINED 3-way ablation + horizon 스윕(1/3/5/10일)
│   ├── feature_engineering_valuation.py    # 밸류에이션(PER/PBR/배당수익률) feature 추가, horizon 스윕 지원(1/3/5/10/20일)
│   ├── train_xgboost_valuation_ablation.py # BASE/VALUATION_ONLY/COMBINED 3-way ablation + 멀티 시드 + horizon 스윕
│   ├── backtest_valuation_comparison.py    # BASE/VALUATION_ONLY/COMBINED 3자 거래비용 반영 백테스트 + Buy & Hold
│   ├── analyze_signal_concentration.py     # 백테스트 수익이 특정 연도에 몰려있는지(국면 의존성) 연도별 분해
│   ├── backtest_valuation_excl_2025.py     # 2025년 이례적 강세장을 제외하고 재검증
│   ├── feature_engineering_fx.py           # 환율(USD/KRW) feature 추가, 멀티 horizon 저장 지원(1/3/5/10일)
│   ├── train_xgboost_ablation_fx.py        # BASE/FX_ONLY/COMBINED 3-way ablation + 멀티 시드 + horizon 스윕
│   ├── backtest_comparison_fx.py           # BASE vs COMBINED 신호 기반 실전 백테스트 비교 (h=3, 다종목 순회)
│   ├── analyze_signal_concentration_fx.py  # FX 백테스트 수익의 연도별 집중도 분해
│   ├── backtest_fx_excl_regime_year.py     # 국면 지배 연도(종목별로 다름) 거래를 사후 제외하고 재계산
│   ├── feature_engineering_foreign_ownership.py   # 외국인 보유율/한도소진율 feature 추가, 멀티 horizon 저장 지원
│   ├── train_xgboost_ablation_foreign_own.py      # BASE/FOREIGN_OWN_ONLY/COMBINED 3-way ablation + 멀티 시드 + horizon 스윕
│   ├── backtest_comparison_foreign_own.py         # BASE vs COMBINED 실전 백테스트 비교 (h=5, 다종목 순회)
│   ├── analyze_signal_concentration_foreign_own.py # BASE/COMBINED 각각의 연도별 손익 분해 (구조적 저하 여부 확인용)
│   ├── feature_engineering_dart.py                # DART 공시 빈도 feature 추가
│   ├── train_xgboost_ablation_dart.py             # BASE/DART_ONLY/COMBINED 3-way ablation
│   ├── feature_engineering_dart_major_holder.py   # DART 대량보유상황보고(5% Rule) feature 추가
│   ├── train_xgboost_ablation_major_holder.py     # BASE/MAJOR_HOLDER_ONLY/COMBINED 3-way ablation
│   ├── analyze_signal_concentration_daehan_valuation.py # 대한제강 밸류에이션 백테스트 연도별 분해
│   └── train_final_model.py                # 최종 프로덕션 모델 (현대로템, BASE+밸류에이션 COMBINED, h=20) 학습/저장/예측
├── data/                                   # 생성되는 {ticker}_features*.csv (gitignore 처리, 스크립트가 자동 생성)
├── models/                                 # train_final_model.py가 저장하는 .joblib/_metadata.json (gitignore 처리)
├── docs/
│   └── PROJECT_SUMMARY.md                  # 프로젝트 종합 보고서 (실험별 결과, 핵심 교훈)
└── README.md
```

모든 스크립트는 **레포 루트에서** `python src/스크립트명.py` 형태로 실행합니다 (아래 "실행 순서"의 모든 명령어도 동일). 각 스크립트는 CSV를 읽고 쓸 때 `data/` 폴더를 자동으로 바라보도록 `Path(__file__)` 기준 상대경로로 되어 있어, 실행 위치만 레포 루트로 맞추면 별도 설정 없이 그대로 동작합니다.

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
python src/feature_engineering.py     # {ticker}_features.csv 생성
python src/train_xgboost_wfo.py       # walk-forward 학습 + fold별 성능
python src/backtest_simulation.py     # 거래비용 반영 실전 시뮬레이션
```

### 3) 다종목 풀링 파이프라인

```bash
python src/feature_engineering_pooled.py   # pooled_features.csv 생성
python src/train_xgboost_pooled.py         # 날짜 기준 walk-forward 학습
```

### 4) 수급 데이터(외국인/기관 순매수) feature 추가 파이프라인

```bash
python src/feature_engineering_investor.py   # {ticker}_features_with_investor_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python src/train_xgboost_ablation.py         # BASE/INVESTOR_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python src/backtest_comparison.py            # BASE vs COMBINED 실전 백테스트 비교
```

`feature_engineering_investor.py`의 `TICKER`/`TICKER_KRX`/`DEFAULT_HORIZON_COST_MAP` 상수로 종목·기간·horizon별 라벨 임계값 변경 가능합니다.

### 5) 개별종목 공매도(국내) feature 추가 파이프라인

```bash
python src/feature_engineering_short_kr.py   # {ticker_krx}_features_with_short_kr_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python src/train_xgboost_short_ablation.py   # BASE/SHORT_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
```

`feature_engineering_short_kr.py`의 `TICKER`/`TICKER_KRX` 상수로 종목 변경 가능합니다.

### 6) 밸류에이션(PER/PBR/배당수익률) feature 추가 파이프라인

```bash
python src/feature_engineering_valuation.py       # {ticker_krx}_features_with_valuation_h{horizon}.csv 생성 (horizon 1/3/5/10/20 자동 생성)
python src/train_xgboost_valuation_ablation.py    # BASE/VALUATION_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python src/backtest_valuation_comparison.py       # BASE/VALUATION_ONLY/COMBINED 3자 백테스트 + Buy & Hold (horizon=10, 20)
python src/analyze_signal_concentration.py        # 백테스트 수익이 특정 연도에 몰려있는지 확인
python src/backtest_valuation_excl_2025.py        # 2025년 이례적 강세장을 제외하고 재검증
```

PER/PBR은 그날 종가 기준으로 계산되는 값(EPS/BPS는 이미 공시된 분기 실적)이라 수급/공매도와 달리
1일 shift가 필요 없습니다 -- 가격 feature와 동일하게 "당일 종가 시점에 이미 확정된 정보"로 취급합니다.
절대 PER/PBR은 종목/업종마다 기준이 달라서, 그 종목 자체의 최근 1년(252거래일) 분포 대비
z-score로 정규화한 `per_zscore_252d`/`pbr_zscore_252d`도 함께 사용합니다.

**pykrx 공매도 데이터 이력 복원 (중요):**
pykrx의 `get_shorting_volume_by_date()` wrapper 함수는 약 2년(730일)이 넘는 기간을 한 번에 요청하면
내부에서 `KeyError('거래량')`로 죽습니다. 원인은 wrapper가 내부적으로 호출하는 raw 함수
(`pykrx.website.krx.get_shorting_trading_value_and_volume_by_date`)가 긴 기간 요청 시 깨진 응답을
주기 때문입니다. `feature_engineering_short_kr.py`는 이 raw 함수를 600일 단위 청크로 나눠서 호출 →
이어붙이는 방식으로 이 한계를 우회하며, 2015년부터 전체 이력(약 2800행)을 복원합니다.
공매도 전면 금지 기간(2023.11~2025.03)에 걸리는 청크도 정상적으로 값이 채워져서 넘어옵니다.

**사전 준비 (KRX_ID/KRX_PW):**
```bash
# .env 파일 생성 (git에 커밋 금지 -- .gitignore에 이미 포함돼 있어야 함)
KRX_ID=본인_krx_아이디
KRX_PW=본인_krx_비밀번호
```
pykrx가 2025년 12월 KRX 회원제 전환 이후 로그인 방식으로 바뀌었습니다.
`load_dotenv()`는 반드시 pykrx(또는 `pykrx.website`) import보다 먼저 실행해야 하며, `python -c "..."`로
직접 pykrx 함수를 호출할 때도 이 순서를 빠뜨리면 로그인 실패로 인해 API가 빈 응답을 주고, 그로 인해
전혀 다른 원인(예: 기간 제한)으로 착각하기 쉬우니 주의하세요.
KRX가 비공식 스크래핑 도구(IP)를 차단하는 경우가 있어, 막히면 네트워크를 바꿔서(예: 테더링) 재시도하세요.

### 7) 환율(USD/KRW) feature 추가 파이프라인

```bash
python src/feature_engineering_fx.py           # {ticker_krx}_features_with_fx_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python src/train_xgboost_ablation_fx.py        # BASE/FX_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python src/backtest_comparison_fx.py           # BASE vs COMBINED 실전 백테스트 비교 (h=3, 현대로템/대한제강 순회)
python src/analyze_signal_concentration_fx.py  # 백테스트 수익이 특정 연도에 몰려있는지 확인
python src/backtest_fx_excl_regime_year.py     # 국면 지배 연도의 거래를 사후 제외하고 재계산
```

환율은 yfinance `KRW=X`로 조회하며, KRX(평일 장중만 거래)와 FX 시장(사실상 24시간 거래)의
캘린더가 어긋나므로 수급/공매도와 마찬가지로 1일 shift를 적용합니다. `feature_engineering_fx.py`의
`build_multi_horizon_datasets_fx()`는 환율 원본 데이터를 한 번만 다운로드해 캐싱한 뒤 horizon별로
재사용합니다 (환율 자체는 종목/horizon과 무관하게 동일하기 때문).

### 8) 외국인 보유율/한도소진율 feature 추가 파이프라인

```bash
python src/feature_engineering_foreign_ownership.py    # {ticker_krx}_features_with_foreign_own_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python src/train_xgboost_ablation_foreign_own.py       # BASE/FOREIGN_OWN_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python src/backtest_comparison_foreign_own.py          # BASE vs COMBINED 실전 백테스트 비교 (h=5, 현대로템/삼성전자 순회)
python src/analyze_signal_concentration_foreign_own.py # BASE/COMBINED 각각의 연도별 손익 분해
```

수급(외국인 순매수)이 "당일 사고판 양(flow)"인 반면, 이 실험은 "지금 얼마나 들고 있나(누적 레벨,
stock)"를 feature로 씁니다. pykrx `get_exhaustion_rates_of_foreign_investment()`로 지분율/한도소진율을
조회하며, 공식 문서상 "전일자 확정치"라 수급과 동일하게 1일 shift를 적용합니다. 단, pykrx 자체 문서에
"D-2 데이터이며 D-1엔 0으로 나온다"는 상반된 설명도 있어, `load_foreign_ownership_data()`는 원본 데이터의
0값 비율을 출력해 실제로 지연 이슈가 있는지 데이터 품질을 확인하도록 설계했습니다 (0값 비율이 2%를
넘으면 `shift_days`를 2로 조정 권장).

**pykrx 컬럼명 버전 차이 주의:** `get_exhaustion_rates_of_foreign_investment()`의 한도소진율 컬럼명이
`'한도소진율'`/`'한도소진률'`로 pykrx 버전에 따라 다르게 나올 수 있어, `feature_engineering_foreign_ownership.py`는
컬럼명에 `'한도소진'`이 포함되는지로 동적 매칭합니다. 정확한 컬럼명이 궁금하면 콘솔에 찍히는
`[디버그] raw 컬럼명` 출력을 확인하세요.

**pandas ffill dtype 에러 주의:** `pykrx` 반환 컬럼이 nullable extension dtype(`Int64`/`Float64`)으로
남아있으면 `.ffill()` 호출 시 `TypeError: No matching signature found`가 날 수 있습니다.
`pd.to_numeric(..., errors="coerce").astype("float64")`로 표준 numpy dtype으로 명시적 캐스팅한 뒤
`ffill()`을 호출해야 합니다 (`feature_engineering_foreign_ownership.py`의 `add_foreign_ownership_features()` 참고).

### 9) 최종 프로덕션 모델 (현대로템, BASE+밸류에이션 COMBINED, horizon=20)

```bash
python src/train_final_model.py              # 전체 히스토리로 최종 학습 + 모델/메타데이터 저장
python src/train_final_model.py --predict     # 저장된 모델로 최신 시점 기준 예측
```

지금까지의 ablation/backtest 스크립트는 전부 Walk-Forward로 "검증"만 했음 (매 fold마다
재학습 → 평가). 이 스크립트는 실제 배포를 가정하고, 사용 가능한 전체 히스토리로 모델을
**한 번만 학습**해서 `models/final_model_064350_h20.joblib` + `models/final_model_064350_h20_metadata.json`으로
저장함. 선정 근거는 docs/PROJECT_SUMMARY.md 7절 참고.

**⚠️ 예측 시 반드시 알아야 할 버그/우회 사항**: `feature_engineering.py`의
`build_feature_dataset()`은 마지막에 `df[feature_cols].dropna()`를 호출하는데, `feature_cols`에
`label`/`future_return`이 포함돼 있음. 라벨은 horizon(20)일 후 수익률로 계산되므로 데이터
맨 끝 20영업일치는 항상 NaN이고, 이 dropna 때문에 **최근 20영업일이 통째로 사라짐** (학습
목적으로는 정상 동작이지만, "오늘 시점 예측"에는 방해가 됨). `train_final_model.py`의
`predict_latest()`는 `add_label()`을 거치지 않고 개별 feature 함수(`add_momentum_features` 등)만
직접 호출해서 이 문제를 우회함.

**실제 검증 사례**: 2026년 8월 초 예측 시점에 모델이 강하게 베어리시한 확률(0.0956)을 냄.
원인을 추적한 결과 7/24 발표된 2분기 어닝쇼크(영업이익 컨센서스 대비 -14%)로 7/27 하루
-16.14% 폭락 + 같은 주 코스피 전체 폭락(7/28 -10.84%, 7/29 -5.98%)이 겹친 실제 시장 이벤트였음.
모델이 이 불안정한 국면(높은 변동성, 어닝쇼크 후 불완전한 반등)을 제대로 포착한 것으로 해석됨 --
데이터 글리치가 아니라는 걸 yfinance raw 데이터(`auto_adjust=False`)와 외부 소스(증권사 리포트,
뉴스 기사) 교차검증으로 확인.

---

## 방법론 요약

- **가격 feature**: 모멘텀(수익률, RSI, MACD), 변동성(볼린저밴드, ATR, 역사적변동성), 거래량(OBV, 거래량비율), 상대강도(벤치마크 대비 초과수익) — 13개
- **수급 feature**: 외국인/기관 순매수 3·5일 누적, 거래대금 대비 순매수 비율(정규화), 외국인·기관 동반매수 여부 — 7개
- **공매도 feature (국내, 개별종목)**: 공매도 거래량 3·5일 누적, 공매도 비중 5일 평균, 숏커버링 신호(최근 3일 공매도량 급감 여부) — 4개
- **밸류에이션 feature**: PER/PBR/배당수익률 + 종목별 1년 z-score 정규화 — 6개
- **환율 feature**: 원달러 환율 5·10·20일 모멘텀, 20일 변동성, 20일 이동평균 이격도 — 5개
- **외국인 보유율 feature**: 지분율 5·20일 변화폭, 변화의 가속도(5일), 한도소진율 레벨 — 4개
- **Label**: N일 후 수익률이 거래비용 임계값을 넘으면 1, 아니면 0 (horizon이 짧을수록 임계값도 낮춰서 라벨 불균형 방지)
- **검증**: Walk-Forward + Embargo(라벨 겹침으로 인한 룩어헤드 방지)
- **평가**: 단순 accuracy가 아니라 각 fold의 실제 base_rate(다수클래스 비율) 대비 우위(`vs_base_rate`)로 판단
- **멀티 시드 검증**: XGBoost의 `random_state`를 5개로 바꿔가며 반복, 방향이 일관되는지 확인 (단일 시드 결과의 우연성 배제)
- **Horizon 스윕**: 라벨 horizon을 1/3/5/10(/20)일로 바꿔가며 반복 — 특정 신호가 며칠 후에 가장 잘 반영되는지 확인
- **실전 검증**: 거래비용 반영 백테스트로 Buy & Hold 대비 실제 손익/MDD 비교
- **국면 집중도 검증**: 백테스트 수익이 특정 연도 하나에 몰려있지 않은지 연도별로 분해해서 확인
- **구조적 저하 검증**: (외국인 보유율 실험에서 도입) BASE/COMBINED 각각을 연도별로 분해해서 손실/이익 연도 개수 자체를 비교 -- 국면 우연과 여러 해에 걸친 구조적 저하를 구분하기 위함

### 수급/공매도/환율/외국인보유율 데이터 룩어헤드 방지 (중요)

외국인/기관 순매수, 공매도 거래량, 외국인 보유율/한도소진율은 **당일 장 마감 후(저녁)에 발표/집계**됩니다.
환율은 발표 지연은 없지만 KRX(평일 장중)와 FX 시장(사실상 24시간)의 거래 캘린더가 어긋나 있어 같은 성격의
문제가 생깁니다. 반면 백테스트는 "당일 종가 매매"를 가정하므로, N일차 row에는 N일 당일 값이 아니라
**N-1일까지 확정된 값만** 들어가도록 각 feature 생성 모듈에서 1일 shift를 적용했습니다.
이 shift를 빠뜨리면 아직 확정되지 않은 정보로 매매한 것처럼 계산되어 결과가 부풀려집니다.
(반면 PER/PBR은 그날 종가 기준 계산값이라 이 shift가 필요 없습니다 -- 위 6번 섹션 참고.)

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
| 삼성전자, horizon=5일, **5개 시드 반복** | AUC 차이 +0.0147 (표준편차 0.0045), **5/5 시드 COMBINED 우위** |
| 현대로템, horizon=5일, **5개 시드 반복** | AUC 차이 **+0.0272** (표준편차 0.0034), **5/5 시드 COMBINED 우위** — 유동성 낮은 종목에서 효과 더 큼 |
| INVESTOR_ONLY (수급 feature만) | 두 종목 모두 base_rate 못 이김 — 수급 데이터 단독으론 무의미, 가격 feature와 결합해야 의미 있음 |
| 실전 백테스트 (현대로템, threshold=0.55) | BASE 271.9% → **COMBINED 497.2%** (거의 2배), MDD도 개선 |
| 실전 백테스트 vs Buy & Hold | COMBINED도 Buy & Hold(1236.7%) 못 이김 |

**결론**: 수급 데이터는 가격 feature와 결합했을 때 통계적·경제적 개선을 일관되게 가져오고 유동성 가설과도
일치하지만, 그 개선은 상대적인 것일 뿐 "N일 후 방향성 예측 + 순차 진입" 전략 자체는 강한 상승장에서
Buy & Hold를 이기지 못하는 구조적 한계를 가짐.

### 3) 개별종목 공매도(국내) 데이터 추가 실험

| 실험 | 결과 |
|---|---|
| 현대로템, horizon=5일, **5개 시드 반복** | AUC 차이 **-0.0147**, **5/5 시드 COMBINED가 오히려 BASE보다 나쁨** |
| 삼성전자, horizon=5일, **5개 시드 반복** | AUC 차이 -0.0050, 1/5 시드만 개선 |
| Horizon 스윕 (1/3/5/10일) | 모든 horizon에서 표준편차가 평균과 비슷하거나 더 큼 — 노이즈 수준 |

**결론**: 2종목 × 4개 horizon × 5개 시드(40회 독립 실행)에서 재현 가능한 엣지를 확인하지 못함.
가격 feature가 이미 담고 있는 정보 이상의 독립적인 신호를 주지 못하는 것으로 보임.

### 4) 밸류에이션(PER/PBR/배당수익률) 데이터 추가 실험

| 실험 | 결과 |
|---|---|
| Horizon 스윕 (1/3/5/10/20일) | horizon이 길어질수록 COMBINED-BASE AUC 차이가 단조 증가, 대부분 종목 5/5 시드 일관 |
| **2025년 제외 재검증** | 현대로템: COMBINED 64.1% > Buy&Hold 29.5% (재현). 삼성전자: 2025년 빼자 엣지 소멸 |
| **한전기술 (저유동성)** | 2025년 제외해도 Buy&Hold 재차 상회. 단, 최대 기여 연도는 2021년(원전 테마) |
| **모트렉스 (초저유동성)** | ablation은 전체 최고(AUC 0.693)인데 백테스트는 Buy&Hold 못 이김 -- AUC와 손익이 괴리 |
| **대한제강 (밸류에이션 실험 당시)** | ablation부터 horizon=5 방향이 뒤집히는 첫 예외, 백테스트 최악(-68.0%) |

**5종목 종합**: 저유동성 4종목 중 백테스트가 Buy&Hold를 이긴 건 2개(현대로템, 한전기술)뿐. 유동성은
AUC와는 상관관계를 보이지만 실전 손익 우위를 보장하는 충분조건은 아니며, 종목별 특수성이 더 크게 작용.

### 5) 환율(USD/KRW) 데이터 추가 실험

발표 지연은 없지만 KRX/FX 캘린더 불일치로 인한 룩어헤드 방지 위해 1일 shift 적용. 수출주(현대로템)와
수입원자재 의존주(대한제강)가 환율 방향에 반대로 반응할 것이라는 가설로 시작.

| 실험 | 결과 |
|---|---|
| Horizon 스윕 (1/3/5/10일, 3종목) | **h=3에서만** 유의미한 재현: 현대로템 +0.0106(5/5, std/mean 18%), 대한제강 +0.0153(5/5, std/mean 25%). h=1/5/10은 종목마다 부호가 갈리거나 노이즈 수준 |
| 삼성전자 h=3 교차검증 | +0.0039(5/5 시드지만 std/mean 62%로 판정 기준 초과) -- 유동성 가설과 일치, 고유동성 종목에서 신호 실질적으로 소멸 |
| 가설과의 불일치 | 두 종목(수출/수입원자재) 모두 h=3에서 **같은 부호(+)**로 나옴 -- 종목 펀더멘털 방향성이 아니라 시장 전반의 타이밍 신호에 가까움을 시사 |
| 백테스트 (h=3, threshold=0.55) | 현대로템: BASE 13.6% → COMBINED **157.2%** (Buy&Hold 1236.7%는 못 이김). 대한제강: BASE -24.4% → COMBINED **-60.9%** (AUC는 더 좋았는데 손익은 더 나쁨 -- 종목 간 정반대) |
| 국면 집중도 분석 | 현대로템: 2025년 한 해가 log-return 기여도 109.3%(그 해 주가 +261.4%, 방산주 랠리). 대한제강: 2018년 한 해가 기여도 78.2%(그 해 주가 -39.9%, 하락장) |
| **국면 지배 연도 제외 재검증** | 현대로템: 2025년 제외 시 BASE/COMBINED **둘 다 마이너스로 전환**(13.6%→-53.0%, 157.2%→-8.4%) -- COMBINED가 BASE보다 낫다는 방향은 유지되나 "수익 나는 전략"이라는 결론은 착시였음. 대한제강: 2018년 제외해도 COMBINED가 BASE보다 못함(-18.6% vs +52.5%) -- 국면 우연이 아니라 일관된 역효과 |

**결론**: Ablation 수준(AUC)의 재현성(h=3, 3종목 5/5 시드)은 진짜였고 유동성 가설과도 일치했으나, 이
재현성이 실전 손익 edge로 이어진다는 증거는 국면 집중도 체크 이후 대부분 무너짐. 4개 alternative data
실험(수급/공매도/밸류에이션/FX) 중 AUC 재현성과 backtest robustness가 가장 크게 괴리된 케이스.

### 6) 외국인 보유율/한도소진율 데이터 추가 실험

수급(순매수, flow) 데이터와 달리 "지금 얼마나 들고 있나"의 누적 레벨(stock) 정보. 수급 실험과 같은
근본 원인(외국인 자금이 관여하는 저유동성 종목)으로 좋은 결과가 나오는 건지, 진짜 독립적인 신호인지
가리는 게 목적.

| 실험 | 결과 |
|---|---|
| Horizon 스윕 (1/3/5/10일, 현대로템) | **h=5에서만** 유의미한 재현: +0.0225(5/5, std/mean 14% -- 지금까지 실험 중 두 번째로 강하고 깨끗함). h=1/3/10은 노이즈~경계선 수준 |
| 삼성전자 h=5 교차검증 | +0.0101(5/5 시드, std/mean 42% -- 판정 기준은 통과하지만 로템의 절반 이하). FOREIGN_OWN_ONLY 단독 AUC는 0.4999로 사실상 랜덤 -- 유동성 가설과 일치 |
| 백테스트 (h=5, threshold=0.55) | 현대로템: BASE 258.2% → COMBINED **113.8%** (오히려 절반 이하로 감소). 삼성전자: BASE 819.8% → COMBINED **431.5%** (마찬가지로 절반 수준 감소). **두 종목 모두 AUC는 개선됐는데 실전 손익은 일관되게 악화** -- FX처럼 종목별로 갈리지 않고 방향이 같았음 |
| **BASE/COMBINED 각각 연도별 분해 (구조적 저하 검증)** | 현대로템: 손실 연도가 BASE 4개→COMBINED **7개**로 증가(이익/손실 비율 7승4패→4승7패로 역전). 삼성전자: 손실 연도가 BASE 3개→COMBINED **4개**로 증가. 원래 수익 나던 해(로템 2020/2021/2026, 삼성 2021)가 COMBINED에서 손실로 전환되는 패턴이 반복 -- 특정 연도 하나의 우연이 아니라 **여러 해에 걸친 구조적 저하** |

**결론**:
1. Ablation 수준(AUC)에서는 h=5, 두 종목 모두 5/5 시드로 재현됐고 유동성 가설과도 일치 --
   통계적으로는 이번 실험 중 가장 깨끗한 신호 중 하나
2. 그러나 백테스트에서는 국면 집중(FX/밸류에이션 패턴)이 아니라 **다른 종류의 실패 패턴**이 나타남:
   두 종목 모두 COMBINED가 BASE보다 일관되게 나빴고, 원인은 연도별 분해에서 "원래 이익이던 해를
   손실로 뒤집는" 구조적 저하로 확인됨
3. 즉 이 feature는 AUC(전체 방향성 판별력)는 개선시키지만, threshold=0.55를 넘는 고확신 거래
   구간에서는 오히려 신호를 왜곡시키는 것으로 보임 -- "AUC 개선 ≠ 실전 손익 개선"이라는 프로젝트
   공통 결론에 새로운 실패 유형(국면 우연이 아닌 구조적 저하)을 추가한 사례

자세한 실험 로그는 별도 기록 문서 참고.

## 주의사항

- `future_return` 컬럼은 라벨/백테스트 전용이며, 학습 feature로 사용하면 룩어헤드(미래정보 누수)가 됩니다.
- Walk-Forward의 `embargo` 값은 각 실험의 `horizon`과 항상 동일하게 맞춰야 합니다.
- 트리 기반 모델(XGBoost)은 feature 스케일링이 불필요합니다.
- 수급/공매도/환율/외국인보유율 데이터는 모두 1일 shift가 필수입니다 (위 "룩어헤드 방지" 섹션 참고).
- pykrx 실행 중 `KRX_ID`/`KRX_PW` 관련 에러나 `JSONDecodeError`, 혹은 `KeyError('거래량')`가 나면,
  자격증명 문제(로그인 순서 누락)나 KRX의 IP 차단, 또는 한 번에 너무 긴 기간(730일 이상)을 요청한 것일
  가능성이 높습니다.
- pykrx 컬럼명은 버전에 따라 표기가 다를 수 있습니다 (예: `한도소진율`/`한도소진률`). 정확한 이름을
  모를 땐 하드코딩하지 말고 `raw.columns`를 출력해서 확인 후 동적으로 매칭할 것.
- pykrx 반환값이 nullable extension dtype(`Int64`/`Float64`)으로 남아있으면 `.ffill()`에서
  `TypeError: No matching signature found`가 날 수 있습니다. `pd.to_numeric(..., errors="coerce").astype("float64")`로
  명시적 캐스팅 후 ffill할 것.
- 단일 시드 결과만으로 "개선됐다/안됐다"를 판단하지 말 것 — 반드시 멀티 시드(최소 5개) 확인 후 결론 내릴 것.
- horizon을 바꿀 땐 라벨 임계값(`cost_threshold`)도 같이 조정할 것.
- 백테스트에서 좋은 누적수익률이 나와도 곧바로 신뢰하지 말 것 — `analyze_signal_concentration*.py`로
  연도별 기여도를 분해해서, 특정 연도(이례적 강세장/약세장) 하나에 손익이 몰려있는 건 아닌지 반드시
  확인할 것. 밸류에이션(삼성전자 464%가 2025년 착시)과 FX(현대로템 157.2%가 2025년 착시, 대한제강
  -60.9%가 2018년 하락장 집중) 실험 둘 다에서 이 패턴이 반복 확인됨 -- alternative data 개별 feature의
  결함이 아니라, 소수 거래 표본(400건 미만)에 의존하는 이 백테스트 방법론 자체의 구조적 취약점일
  가능성을 염두에 둘 것.
- **AUC 개선이 실전 손익으로 안 이어지는 데는 최소 두 가지 서로 다른 실패 유형이 있음을 구분할 것**:
  (1) 국면 집중형 -- 특정 연도 하나의 우연에 기댄 경우 (FX/밸류에이션에서 확인), 국면 지배 연도를
  빼면 결론이 뒤집히거나 착시였음이 드러남. (2) 구조적 저하형 -- 여러 해에 걸쳐 일관되게 원래
  이익이던 구간을 손실로 뒤집는 경우 (외국인 보유율 실험에서 확인), 특정 연도를 제외해도 해결되지
  않음. 두 유형은 원인과 대응이 다르므로, `analyze_signal_concentration*.py`로 최대 기여 연도 비중뿐
  아니라 BASE 대비 COMBINED의 손실/이익 연도 개수 변화도 함께 확인할 것.
- 국면 지배 연도가 데이터 중간에 있으면(예: 대한제강 2018년) `end=` 파라미터로 잘라내는 방식이
  안 통합니다. 이 경우 `backtest_fx_excl_regime_year.py`처럼 이미 생성된 거래 목록에서 해당 연도
  거래만 사후 제거하는 방식을 씁니다. 단, 이는 "그 연도를 빼고 재학습했다면 어땠을지"를 근사할
  뿐 완벽한 인과 분리는 아니라는 점에 유의할 것.
- "저유동성 종목"을 고를 땐 시가총액이 작다고 거래대금도 낮다는 보장이 없습니다. `check_liquidity.py`로
  직접 조회해서 숫자로 확인할 것. "저유동성"과 "저변동성"도 다른 개념이니 혼동하지 말 것.
- 코스닥 종목은 yfinance 티커 접미사가 `.KS`(코스피)가 아니라 `.KQ`입니다 -- 다르게 넣으면 조용히
  1행짜리 빈 데이터가 나올 뿐 에러가 안 나서 원인 파악이 늦어질 수 있습니다.
- "AUC가 개선됐다"와 "그 종목에서 실전 백테스트가 Buy&Hold를 이겼다"는 별개 결론입니다 -- 둘을 같은
  결론으로 섞어서 보고하지 말 것. 밸류에이션 4종목, FX 2종목, 외국인 보유율 2종목 모두에서 이 괴리가
  확인됐습니다.
- 여러 세션에 걸쳐 재사용하는 종목코드는 주기적으로 재검증할 것 -- 대한제강을 001430(세아베스틸지주)으로
  잘못 쓴 채 진행된 실험이 있었고, 084010(진짜 대한제강)으로 재검증하니 결론이 완전히 뒤집혔습니다.
- `build_feature_dataset()`은 학습용으로 설계돼 있어 라벨(`label`/`future_return`) NaN인 행까지 전부
  dropna로 제거합니다. "지금 시점 예측"처럼 라벨이 필요 없는 추론 용도로 쓸 땐 `add_label()`을
  거치지 않고 개별 feature 함수만 호출해서 이 dropna를 우회해야 최신 데이터를 온전히 쓸 수 있습니다
  (`train_final_model.py`의 `predict_latest()` 참고).
- 모델 예측이 극단적으로 나올 때(예: base_rate 대비 크게 벗어난 확률) 바로 "모델이 이상하다"고
  단정하지 말 것 -- 실제 시장에 어닝쇼크나 지수 전체 폭락 같은 이벤트가 있었을 수 있습니다. yfinance
  `auto_adjust=False`로 raw Close/Adj Close를 비교하고, 뉴스/증권사 리포트로 교차검증할 것.
