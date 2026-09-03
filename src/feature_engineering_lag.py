"""
과거 시점 피처(lag feature) 추가.

XGBoost는 row(날짜)를 독립 샘플로 취급해서 "어제/그제 값"을 직접 보지 못함.
이 실험은 그 한계를 가장 싸게 우회하는 방법 -- 과거 며칠 전의 피처 값을
오늘 row에 나란히 붙여줘서, 모델이 최소한 "최근 궤적"은 참조할 수 있게 함.
진짜 시퀀스 모델(LSTM/TCN/Transformer) 넘어가기 전 사전 점검용.

⚠️ 사전 등록 (실험 전 고정, 결과 보고 나서 바꾸지 않음):
- LAG_COLS: BASE 13개 중 "상태가 계속 바뀌는" 5개만 선택
  (return_5d, rsi_14, macd_hist, bb_position, excess_return_5d)
  -- 정적 성격이 강한 나머지(hist_vol_20d, atr_14, bb_width, volume_ratio_20d,
     obv_change_20d, return_10d, return_20d, excess_return_20d)는 제외해서
     피처 수 폭증(13 -> 13+39) 방지. COMBINED = 13+15 = 28개로 유지
     (기존 캔들 패턴 실험 13+11=24, 섹터 실험 13+6=19와 비슷한 스케일)
- LAGS: [5, 10, 20] 거래일 -- 기존 BASE 피처의 5/10/20일 윈도우와 스케일 맞춤
- 외부 데이터 조회 불필요 (build_feature_dataset() 결과를 그대로 shift만 하면 됨)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LAG_COLS = ["return_5d", "rsi_14", "macd_hist", "bb_position", "excess_return_5d"]
LAGS = [5, 10, 20]


# ------------------------------------------------------------------
# 1. lag 피처 추가
# ------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, lag_cols: list = None, lags: list = None) -> pd.DataFrame:
    lag_cols = lag_cols if lag_cols is not None else LAG_COLS
    lags = lags if lags is not None else LAGS

    result = df.copy()
    for col in lag_cols:
        for lag in lags:
            result[f"{col}_lag{lag}"] = result[col].shift(lag)
    return result


def lag_feature_names(lag_cols: list = None, lags: list = None) -> list:
    lag_cols = lag_cols if lag_cols is not None else LAG_COLS
    lags = lags if lags is not None else LAGS
    return [f"{col}_lag{lag}" for col in lag_cols for lag in lags]


# ------------------------------------------------------------------
# 2. 여러 horizon 데이터셋 생성 + 저장
# ------------------------------------------------------------------
def build_multi_horizon_datasets_lag(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizons: list = None,
    save: bool = True,
) -> dict:
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    lag_cols = lag_feature_names()

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_lag_features(base)

        # 가장 긴 lag(20일)만큼 데이터 시작 부분은 shift로 NaN이 됨 -> dropna로 제거.
        # (이건 "결측"이 아니라 "그 시점에 해당 값 자체가 아직 없음"이라 XGBoost의
        #  native missing-value handling에 맡기지 않고 명확히 워밍업 구간으로 잘라냄
        #  -- PER=0 버그처럼 "가짜 값"을 남겨두는 것과는 다른 상황.)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=lag_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = DATA_DIR / f"{ticker_krx}_features_with_lag_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"
    HORIZONS = [1, 3, 5, 10, 20]

    datasets = build_multi_horizon_datasets_lag(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)

    print("\n=== 전체 horizon별 요약 ===")
    for horizon, dataset in datasets.items():
        print(f"horizon={horizon}: {dataset.shape[0]}행, label=1 비율 {dataset['label'].mean():.3f}")