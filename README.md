# AI 퀀트 트레이딩 - 방향성 예측 파이프라인

XGBoost를 활용해 개별 종목의 단기(N일 후) 방향성을 예측하는 실험 프로젝트입니다.
가격/거래량 기반 기술지표 → 수급 데이터(외국인/기관 순매수) → 공매도 데이터 순으로
"신선한 재료"를 추가하며 검증했고, 매번 같은 방법론(WFO+embargo+멀티시드+실전백테스트)으로
엄격하게 확인했습니다.

## 파일 구조

```
.
├── feature_engineering.py              # 단일 종목 feature 생성 (모멘텀/변동성/거래량/상대강도)
├── train_xgboost.py                     # 단순 시간순 80/20 분할 학습 + threshold sweep
├── train_xgboost_wfo.py                 # Walk-Forward 학습, fold별 base_rate 비교
├── backtest_simulation.py               # 거래비용 반영 백테스트, Buy & Hold / MDD 비교
├── feature_engineering_pooled.py        # 다종목 데이터 풀링
├── train_xgboost_pooled.py              # 날짜 기준 Walk-Forward, ticker categorical feature
├── feature_engineering_investor.py      # 수급 데이터(외국인/기관 순매수) feature 추가
├── train_xgboost_ablation.py            # 수급 feature ablation + 멀티 시드 검증
├── backtest_comparison.py               # 수급 feature BASE vs COMBINED 실전 백테스트
├── feature_engineering_short.py         # 공매도 데이터(Kiwoom API) feature 추가
├── train_xgboost_short_ablation.py      # 공매도 feature ablation + 멀티 시드 검증
├── backtest_comparison_short.py         # 공매도 feature BASE vs COMBINED 실전 백테스트
└── README.md
```

## 실행 순서

### 1) 가상환경 세팅

```bash
python -m venv venv
venv\Scripts\activate        # Windows

pip install yfinance pandas numpy xgboost scikit-learn pykrx python-dotenv requests
```

### 2) 단일 종목 파이프라인 (가격 feature만)

```bash
python feature_engineering.py
python train_xgboost_wfo.py
python backtest_simulation.py
```

### 3) 수급 데이터(외국인/기관 순매수) 파이프라인

```bash
python feature_engineering_investor.py   # pykrx 기반, .env에 KRX_ID/KRX_PW 필요
python train_xgboost_ablation.py
python backtest_comparison.py
```

### 4) 공매도 데이터 파이프라인 (Kiwoom 공식 REST API 기반)

```bash
python feature_engineering_short.py      # kiwoomcli 인증 완료 + kiwoom 패키지 설치 필요
python train_xgboost_short_ablation.py
python backtest_comparison_short.py
```

`feature_engineering_investor.py`/`feature_engineering_short.py`의 `TICKER`/`TICKER_KRX`로 종목 변경 가능.

**사전 준비:**
- 수급 데이터: `.env`에 `KRX_ID`/`KRX_PW` (pykrx, KRX Data Marketplace 회원가입 필요)
- 공매도 데이터: `kiwoomcli setup`으로 키움 API 인증 완료 + `pip install -e "경로\Kiwoom-REST-API"` (공식 레포: github.com/Kiwoom-Securities/Kiwoom-REST-API)
- 두 데이터 모두 국내 IP가 차단되는 경우가 있어, 안 되면 네트워크(테더링 등)를 바꿔서 재시도

## 방법론 요약

- **가격 feature (13개)**: 모멘텀, 변동성, 거래량, 상대강도
- **수급 feature (7개)**: 외국인/기관 순매수 3·5일 누적, 거래대금 대비 비율, 동반매수 여부
- **공매도 feature (4개)**: 공매도량 3·5일 누적, 매매비중(API 제공값) 5일 평균, 숏커버링 신호
- **검증**: Walk-Forward + Embargo, `step == test_size`로 fold 간 겹침 방지
- **평가**: `vs_base_rate`(다수클래스 대비 우위) + AUC를 항상 같이 봄
- **멀티 시드 검증**: `random_state` 5개(42, 1, 7, 123, 2024)로 반복, 방향 일관성 확인
- **실전 검증**: 거래비용(왕복 0.2%) 반영 백테스트로 Buy & Hold 대비 비교

### 룩어헤드 방지 (수급·공매도 공통)

두 데이터 모두 당일 장 마감 후 발표되므로, N일차 row에는 N-1일까지 발표분만 들어가도록 1일 shift 적용.

## 핵심 발견 (요약)

### 1) 가격 feature만으로는 엣지 없음 (기존 결론)
NVDA/삼성전자 단일종목 실험, 다종목 풀링 모두 뚜렷한 엣지 없이 Buy & Hold 대비 열위.

### 2) 수급 데이터(외국인/기관 순매수) — 상대적 개선 재현됨, 절대 수익성은 실패

| 실험 | 결과 |
|---|---|
| 삼성전자, 5개 시드 | AUC 차이 +0.0147, 5/5 시드 COMBINED 우위 |
| 현대로템, 5개 시드 | AUC 차이 +0.0272, 5/5 시드 COMBINED 우위 — 저유동성 종목에서 효과 더 큼 |
| INVESTOR_ONLY (수급 단독) | 두 종목 모두 base_rate 못 이김 |
| 실전 백테스트 (현대로템) | BASE 271.9% -> COMBINED 497.2%, 둘 다 Buy & Hold(1236.7%) 못 이김 |

### 3) 공매도 데이터 — 데이터 접근성 자체가 여러 겹의 장벽

| 시도 | 결과 |
|---|---|
| 한국 (pykrx, 장기간) | 2023.11~2025.03 전면 공매도 금지 이력으로 다년치 조회 시 API 응답 깨짐 |
| 한국 (Kiwoom API) | 공매도 재개 이후(2025.01~) 약 1.5년치만 확보 가능 (구조적 한계) |
| 미국 (FINRA 텍스트 파일 스크래핑) | Cloudflare 봇 차단으로 실패 |
| 미국 (FINRA 공식 API) | 인증/호출 성공했으나 최근 1년 롤링 윈도우만 제공 -- 미채택 |

**한국 Kiwoom API 데이터(1.5년, fold 12개)로 검증:**

| 실험 | 결과 |
|---|---|
| 현대로템(저유동성), 5개 시드 | AUC 차이 +0.0560, 5/5 시드 모두 COMBINED 우위 -- 가장 일관된 신호 |
| 삼성전자(초고유동성), 5개 시드 | AUC 5/5 개선이지만 vs_base_rate는 3/5만, SHORT_ONLY AUC 0.46 -- 노이즈에 가까움 |
| 실전 백테스트 (현대로템) | BASE -20.6% -> COMBINED -10.8% (둘 다 손실이지만 절반으로 축소), Buy & Hold +2.7% 못 이김 |

### 유동성 가설, 두 번째로 검증됨

"유동성 낮은 종목일수록 수급/공매도 정보가 가격에 늦게 반영돼 엣지가 남는다"는 가설이
수급 데이터(현대로템 > 삼성전자)에 이어 공매도 데이터에서도 같은 방향으로 재현됨.

### 종합 결론

1. 가격 기반 기술지표만으로는 엣지 없음 (유지)
2. 수급/공매도 feature는 저유동성 종목에 한해 통계적·경제적으로 일관된 개선을 가져옴
3. 이 개선은 항상 상대적이며, 타이밍 전략 자체는 모든 조합에서 Buy & Hold를 이기지 못함
4. 데이터 접근성 자체가 실험 설계의 큰 제약이 될 수 있음을 여러 사례로 확인

## 주의사항

- `future_return`은 라벨/백테스트 전용, 학습 feature로 쓰면 룩어헤드.
- `embargo`는 `horizon`과 항상 동일하게.
- walk-forward split 시 `step >= test_size`를 반드시 지킬 것 (fold 겹침 방지 -- 실제로 한 번 실수로 AUC 차이가 +0.12까지 부풀려졌다가 +0.056으로 정정된 사례 있음).
- 수급/공매도 데이터는 1일 shift 필수.
- 한국 공매도 데이터는 2023.11~2025.03 전면 금지 이력 때문에 장기간 조회 시 API가 깨질 수 있음.
- pykrx/Kiwoom 모두 국내 IP 차단 사례가 있었음 -- 안 되면 네트워크를 바꿔서 재시도.
- 단일 시드 결과만으로 결론 내리지 말 것 -- 반드시 멀티 시드(최소 5개) 확인 후 판단.
- AUC와 vs_base_rate가 서로 다른 방향을 가리키면 노이즈일 가능성이 높다는 신호로 취급할 것.
