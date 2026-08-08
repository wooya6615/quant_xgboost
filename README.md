# AI 퀀트 트레이딩 - 방향성 예측 파이프라인

XGBoost를 활용해 개별 종목의 단기(N일 후) 방향성을 예측하는 실험 프로젝트입니다.
가격/거래량 기반 기술지표만으로 방향성 예측이 가능한지 검증했고,
이후 완전히 다른 소스인 수급 데이터(외국인/기관 순매수), 개별종목 공매도 데이터,
밸류에이션(PER/PBR/배당수익률), 환율(USD/KRW)을 추가했을 때 개선이 있는지까지 검증했습니다.

## 파일 구조

```
.
├── feature_engineering.py             # 단일 종목 feature 생성 (모멘텀/변동성/거래량/상대강도)
├── train_xgboost.py                   # 단순 시간순 80/20 분할 학습 + threshold sweep
├── train_xgboost_wfo.py               # Walk-Forward 학습, fold별 base_rate 비교, 신호 집중도 분석
├── backtest_simulation.py             # 거래비용 반영 백테스트, Buy & Hold / MDD 비교
├── feature_engineering_pooled.py      # 다종목 데이터 풀링 (섹터 구성 변경 가능)
├── train_xgboost_pooled.py            # 날짜 기준 Walk-Forward, ticker를 categorical feature로 학습
├── feature_engineering_investor.py    # 수급 데이터(외국인/기관 순매수) feature 추가, horizon 스윕 지원
├── train_xgboost_ablation.py          # BASE/INVESTOR_ONLY/COMBINED 3-way ablation + 멀티 시드 검증 + horizon 스윕
├── backtest_comparison.py             # BASE vs COMBINED 신호 기반 실전 백테스트 비교
├── feature_engineering_short_kr.py     # 국내 개별종목 공매도(거래량/비중) feature 추가 -- pykrx 청크 분할로 전체 이력 복원
├── train_xgboost_short_ablation.py     # BASE/SHORT_ONLY/COMBINED 3-way ablation + horizon 스윕(1/3/5/10일)
├── feature_engineering_valuation.py    # 밸류에이션(PER/PBR/배당수익률) feature 추가, horizon 스윕 지원(1/3/5/10/20일)
├── train_xgboost_valuation_ablation.py # BASE/VALUATION_ONLY/COMBINED 3-way ablation + 멀티 시드 + horizon 스윕
├── backtest_valuation_comparison.py    # BASE/VALUATION_ONLY/COMBINED 3자 거래비용 반영 백테스트 + Buy & Hold
├── analyze_signal_concentration.py     # 백테스트 수익이 특정 연도에 몰려있는지(국면 의존성) 연도별 분해
├── backtest_valuation_excl_2025.py     # 2025년 이례적 강세장을 제외하고 재검증
├── check_liquidity.py                  # 후보 종목들의 실제 시가총액/거래대금을 직접 조회 (저유동성 판단용)
├── feature_engineering_fx.py           # 환율(USD/KRW) feature 추가, 멀티 horizon 저장 지원(1/3/5/10일)
├── train_xgboost_ablation_fx.py        # BASE/FX_ONLY/COMBINED 3-way ablation + 멀티 시드 + horizon 스윕
├── backtest_comparison_fx.py           # BASE vs COMBINED 신호 기반 실전 백테스트 비교 (h=3, 다종목 순회)
├── analyze_signal_concentration_fx.py  # FX 백테스트 수익의 연도별 집중도 분해
├── backtest_fx_excl_regime_year.py     # 국면 지배 연도(종목별로 다름) 거래를 사후 제외하고 재계산
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
python feature_engineering_investor.py   # {ticker}_features_with_investor_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python train_xgboost_ablation.py         # BASE/INVESTOR_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python backtest_comparison.py            # BASE vs COMBINED 실전 백테스트 비교
```

`feature_engineering_investor.py`의 `TICKER`/`TICKER_KRX`/`DEFAULT_HORIZON_COST_MAP` 상수로 종목·기간·horizon별 라벨 임계값 변경 가능합니다.

### 5) 개별종목 공매도(국내) feature 추가 파이프라인

```bash
python feature_engineering_short_kr.py   # {ticker_krx}_features_with_short_kr_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python train_xgboost_short_ablation.py   # BASE/SHORT_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
```

`feature_engineering_short_kr.py`의 `TICKER`/`TICKER_KRX` 상수로 종목 변경 가능합니다.

### 6) 밸류에이션(PER/PBR/배당수익률) feature 추가 파이프라인

```bash
python feature_engineering_valuation.py       # {ticker_krx}_features_with_valuation_h{horizon}.csv 생성 (horizon 1/3/5/10/20 자동 생성)
python train_xgboost_valuation_ablation.py    # BASE/VALUATION_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python backtest_valuation_comparison.py       # BASE/VALUATION_ONLY/COMBINED 3자 백테스트 + Buy & Hold (horizon=10, 20)
python analyze_signal_concentration.py        # 백테스트 수익이 특정 연도에 몰려있는지 확인
python backtest_valuation_excl_2025.py        # 2025년 이례적 강세장을 제외하고 재검증
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
python feature_engineering_fx.py           # {ticker_krx}_features_with_fx_h{horizon}.csv 생성 (horizon 1/3/5/10 자동 생성)
python train_xgboost_ablation_fx.py        # BASE/FX_ONLY/COMBINED 비교 + 멀티 시드 검증 + horizon 스윕
python backtest_comparison_fx.py           # BASE vs COMBINED 실전 백테스트 비교 (h=3, 현대로템/대한제강 순회)
python analyze_signal_concentration_fx.py  # 백테스트 수익이 특정 연도에 몰려있는지 확인
python backtest_fx_excl_regime_year.py     # 국면 지배 연도의 거래를 사후 제외하고 재계산
```

환율은 yfinance `KRW=X`로 조회하며, KRX(평일 장중만 거래)와 FX 시장(사실상 24시간 거래)의
캘린더가 어긋나므로 수급/공매도와 마찬가지로 1일 shift를 적용합니다. `feature_engineering_fx.py`의
`build_multi_horizon_datasets_fx()`는 환율 원본 데이터를 한 번만 다운로드해 캐싱한 뒤 horizon별로
재사용합니다 (환율 자체는 종목/horizon과 무관하게 동일하기 때문).

## 방법론 요약

- **가격 feature**: 모멘텀(수익률, RSI, MACD), 변동성(볼린저밴드, ATR, 역사적변동성), 거래량(OBV, 거래량비율), 상대강도(벤치마크 대비 초과수익) — 13개
- **수급 feature**: 외국인/기관 순매수 3·5일 누적, 거래대금 대비 순매수 비율(정규화), 외국인·기관 동반매수 여부 — 7개
- **공매도 feature (국내, 개별종목)**: 공매도 거래량 3·5일 누적, 공매도 비중 5일 평균, 숏커버링 신호(최근 3일 공매도량 급감 여부) — 4개
- **밸류에이션 feature**: PER/PBR/배당수익률 + 종목별 1년 z-score 정규화 — 6개
- **환율 feature**: 원달러 환율 5·10·20일 모멘텀, 20일 변동성, 20일 이동평균 이격도 — 5개
- **Label**: N일 후 수익률이 거래비용 임계값을 넘으면 1, 아니면 0 (horizon이 짧을수록 임계값도 낮춰서 라벨 불균형 방지)
- **검증**: Walk-Forward + Embargo(라벨 겹침으로 인한 룩어헤드 방지)
- **평가**: 단순 accuracy가 아니라 각 fold의 실제 base_rate(다수클래스 비율) 대비 우위(`vs_base_rate`)로 판단
- **멀티 시드 검증**: XGBoost의 `random_state`를 5개로 바꿔가며 반복, 방향이 일관되는지 확인 (단일 시드 결과의 우연성 배제)
- **Horizon 스윕**: 라벨 horizon을 1/3/5/10(/20)일로 바꿔가며 반복 — 특정 신호가 며칠 후에 가장 잘 반영되는지 확인
- **실전 검증**: 거래비용 반영 백테스트로 Buy & Hold 대비 실제 손익/MDD 비교
- **국면 집중도 검증**: 백테스트 수익이 특정 연도 하나에 몰려있지 않은지 연도별로 분해해서 확인

### 수급/공매도/환율 데이터 룩어헤드 방지 (중요)

외국인/기관 순매수, 공매도 거래량은 **당일 장 마감 후(저녁)에 발표/집계**됩니다. 환율은 발표 지연은
없지만 KRX(평일 장중)와 FX 시장(사실상 24시간)의 거래 캘린더가 어긋나 있어 같은 성격의 문제가 생깁니다.
반면 백테스트는 "당일 종가 매매"를 가정하므로, N일차 row에는 N일 당일 값이 아니라 **N-1일까지
확정된 값만** 들어가도록 각 feature 생성 모듈에서 1일 shift를 적용했습니다.
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

### 5) 환율(USD/KRW) 데이터 추가 실험 -- 지금까지 중 가장 약한 결과

발표 지연은 없지만 KRX/FX 캘린더 불일치로 인한 룩어헤드 방지 위해 1일 shift 적용. 수출주(현대로템)와
수입원자재 의존주(대한제강)가 환율 방향에 반대로 반응할 것이라는 가설로 시작.

| 실험 | 결과 |
|---|---|
| Horizon 스윕 (1/3/5/10일, 3종목) | **h=3에서만** 유의미한 재현: 현대로템 +0.0106(5/5, std/mean 18%), 대한제강 +0.0153(5/5, std/mean 25%). h=1/5/10은 종목마다 부호가 갈리거나 노이즈 수준 |
| 삼성전자 h=3 교차검증 | +0.0039(5/5 시드지만 std/mean 62%로 판정 기준 초과) -- 유동성 가설과 일치, 고유동성 종목에서 신호 실질적으로 소멸 |
| 가설과의 불일치 | 두 종목(수출/수입원자재) 모두 h=3에서 **같은 부호(+)**로 나옴 -- 종목 펀더멘털 방향성이 아니라 시장 전반의 타이밍 신호에 가까움을 시사 |
| 백테스트 (h=3, threshold=0.55) | 현대로템: BASE 13.6% → COMBINED **157.2%** (Buy&Hold 1236.7%는 못 이김). 대한제강: BASE -24.4% → COMBINED **-60.9%** (AUC는 더 좋았는데 손익은 더 나쁨 -- 종목 간 정반대) |
| 국면 집중도 분석 | 현대로템: 2025년 한 해가 log-return 기여도 109.3%(그 해 주가 +261.4%). 대한제강: 2018년 한 해가 기여도 78.2%(그 해 주가 -39.9%) |
| **국면 지배 연도 제외 재검증** | 현대로템: 2025년 제외 시 BASE/COMBINED **둘 다 마이너스로 전환**(13.6%→-53.0%, 157.2%→-8.4%) -- COMBINED가 BASE보다 낫다는 방향은 유지되나 "수익 나는 전략"이라는 결론은 착시였음. 대한제강: 2018년 제외해도 COMBINED가 BASE보다 못함(-18.6% vs +52.5%) -- 국면 우연이 아니라 일관된 역효과 |

**결론**:
1. Ablation 수준(AUC)의 재현성(h=3, 3종목 5/5 시드)은 진짜였고, 유동성 가설과도 일치함
2. 그러나 이 재현성이 실전 손익 edge로 이어진다는 증거는 국면 집중도 체크 이후 대부분 무너짐 --
   특히 대한제강은 국면 지배 연도를 빼도 COMBINED가 BASE보다 일관되게 못한 유일한 사례
3. 4개 alternative data 실험(수급/공매도/밸류에이션/FX) 중 AUC 재현성과 backtest robustness가
   가장 크게 괴리된 케이스로, "AUC 재현성"과 "backtest robustness"는 서로 다른 것을 검증하며
   하나가 통과했다고 다른 하나가 자동으로 보장되지 않는다는 프로젝트 공통 패턴을 가장 극명하게 보여줌

자세한 실험 로그는 별도 기록 문서 참고.

## 주의사항

- `future_return` 컬럼은 라벨/백테스트 전용이며, 학습 feature로 사용하면 룩어헤드(미래정보 누수)가 됩니다.
- Walk-Forward의 `embargo` 값은 각 실험의 `horizon`과 항상 동일하게 맞춰야 합니다.
- 트리 기반 모델(XGBoost)은 feature 스케일링이 불필요합니다.
- 수급/공매도/환율 데이터는 모두 1일 shift가 필수입니다 (위 "수급/공매도/환율 데이터 룩어헤드 방지" 참고).
- pykrx 실행 중 `KRX_ID`/`KRX_PW` 관련 에러나 `JSONDecodeError`, 혹은 `KeyError('거래량')`가 나면,
  자격증명 문제(로그인 순서 누락)나 KRX의 IP 차단, 또는 한 번에 너무 긴 기간(730일 이상)을 요청한 것일
  가능성이 높습니다.
- 단일 시드 결과만으로 "개선됐다/안됐다"를 판단하지 말 것 — 반드시 멀티 시드(최소 5개) 확인 후 결론 내릴 것.
- horizon을 바꿀 땐 라벨 임계값(`cost_threshold`)도 같이 조정할 것.
- 백테스트에서 좋은 누적수익률이 나와도 곧바로 신뢰하지 말 것 — `analyze_signal_concentration*.py`로
  연도별 기여도를 분해해서, 특정 연도(이례적 강세장/약세장) 하나에 손익이 몰려있는 건 아닌지 반드시
  확인할 것. 밸류에이션(삼성전자 464%가 2025년 착시)과 FX(현대로템 157.2%가 2025년 착시, 대한제강
  -60.9%가 2018년 하락장 집중) 실험 둘 다에서 이 패턴이 반복 확인됨 -- alternative data 개별 feature의
  결함이 아니라, 소수 거래 표본(400건 미만)에 의존하는 이 백테스트 방법론 자체의 구조적 취약점일
  가능성을 염두에 둘 것.
- 국면 지배 연도가 데이터 중간에 있으면(예: 대한제강 2018년) `end=` 파라미터로 잘라내는 방식이
  안 통합니다. 이 경우 `backtest_fx_excl_regime_year.py`처럼 이미 생성된 거래 목록에서 해당 연도
  거래만 사후 제거하는 방식을 씁니다. 단, 이는 "그 연도를 빼고 재학습했다면 어땠을지"를 근사할
  뿐 완벽한 인과 분리는 아니라는 점에 유의할 것 (walk-forward 특성상 학습 구간과 그 연도가
  겹치지 않는 fold도 있어 어느 정도 근사는 되지만, `end=` 절단 방식만큼 엄밀하진 않음).
- "저유동성 종목"을 고를 땐 시가총액이 작다고 거래대금도 낮다는 보장이 없습니다. `check_liquidity.py`로
  직접 조회해서 숫자로 확인할 것. "저유동성"과 "저변동성"도 다른 개념이니 혼동하지 말 것.
- 코스닥 종목은 yfinance 티커 접미사가 `.KS`(코스피)가 아니라 `.KQ`입니다 -- 다르게 넣으면 조용히
  1행짜리 빈 데이터가 나올 뿐 에러가 안 나서 원인 파악이 늦어질 수 있습니다.
- "AUC가 개선됐다"와 "그 종목에서 실전 백테스트가 Buy&Hold를 이겼다"는 별개 결론입니다 -- 둘을 같은
  결론으로 섞어서 보고하지 말 것. 밸류에이션 4종목, FX 2종목 모두에서 이 괴리가 확인됐습니다.
