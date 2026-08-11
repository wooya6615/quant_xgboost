"""
캔들스틱 패턴 feature 생성 (도지/장악형/망치형/유성형/장대형 + 방향 streak).

배경: 지금까지의 BASE 13개 feature는 전부 OHLCV를 스칼라 요약통계로 가공한 값
(return/RSI/MACD/ATR 등)이라, 캔들 하나하나의 "모양"(몸통 대비 꼬리 길이, 전일 대비
장악 여부 등)은 반영되지 않았음. TA-Lib은 C 라이브러리 의존성 때문에 설치가 까다로워서
직접 pandas로 구현함 -- feature_engineering.py의 RSI/MACD 수동 구현과 동일한 스타일.

⚠️ 룩어헤드 없음: 전부 당일(D일) 종가 확정 후 계산 가능한 값들이라 shift 불필요
(당일 종가로 신호를 만들어 다음날 진입한다고 가정하는 기존 파이프라인과 동일 전제).

⚠️ build_feature_dataset()을 그대로 재사용하지 않는 이유:
그 함수는 마지막에 Open/High/Low를 버리고 [Close, Volume, BASE feature들, label]만
남기기 때문에, 캔들 패턴 계산에 필요한 Open/High/Low가 이미 사라진 뒤임. 그래서
load_data()부터 개별 feature 함수를 직접 호출하는 방식으로 재구성함 (train_final_model.py의
predict_latest()가 라벨 종속 dropna를 우회할 때 썼던 것과 같은 패턴).

사용법 (레포 루트에서):
    python src/feature_engineering_candlestick.py

전제:
    feature_engineering.py, feature_engineering_valuation.py와 같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features, add_label,
)
from feature_engineering_valuation import load_valuation, add_valuation_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_CANDLE = [
    "candle_body_pct", "candle_upper_shadow_pct", "candle_lower_shadow_pct",
    "candle_range_pct", "is_doji", "is_hammer", "is_shooting_star",
    "is_bullish_engulfing", "is_bearish_engulfing", "is_marubozu",
    "candle_direction_streak",
]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION
FEATURE_COLS_COMBINED_CANDLE = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION + FEATURE_COLS_CANDLE


# ------------------------------------------------------------------
# 1. 캔들스틱 패턴 feature
# ------------------------------------------------------------------
def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    body = c - o
    candle_range = (h - l).replace(0, np.nan)  # 상하한가 등 0으로 나누기 방지

    df["candle_body_pct"] = body / candle_range
    df["candle_upper_shadow_pct"] = (h - np.maximum(o, c)) / candle_range
    df["candle_lower_shadow_pct"] = (np.minimum(o, c) - l) / candle_range
    df["candle_range_pct"] = candle_range / c  # 당일 변동폭의 상대적 크기 (단기 변동성 성격)

    # 도지: 몸통이 전체 범위의 10% 미만 -- 시가/종가가 거의 같음 (매수/매도 균형 신호)
    df["is_doji"] = (df["candle_body_pct"].abs() < 0.1).astype(int)

    # 망치형: 아래꼬리가 몸통의 2배 이상 + 위꼬리는 짧음 (하락 후 나오면 반전 신호로 흔히 해석)
    df["is_hammer"] = (
        (df["candle_lower_shadow_pct"] > 2 * df["candle_body_pct"].abs())
        & (df["candle_upper_shadow_pct"] < 0.1)
    ).astype(int)

    # 유성형: 망치형의 반대 -- 위꼬리가 길고 아래꼬리는 짧음
    df["is_shooting_star"] = (
        (df["candle_upper_shadow_pct"] > 2 * df["candle_body_pct"].abs())
        & (df["candle_lower_shadow_pct"] < 0.1)
    ).astype(int)

    # 장악형: 전일 몸통을 당일 몸통이 완전히 감쌈 + 방향이 반대
    prev_o, prev_c = o.shift(1), c.shift(1)
    df["is_bullish_engulfing"] = (
        (c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)
    ).astype(int)
    df["is_bearish_engulfing"] = (
        (c < o) & (prev_c > prev_o) & (o > prev_c) & (c < prev_o)
    ).astype(int)

    # 장대형: 꼬리가 거의 없이 몸통이 전체 범위를 거의 다 차지
    df["is_marubozu"] = (df["candle_body_pct"].abs() > 0.9).astype(int)

    # 연속 상승/하락일 수 (몸통 방향 기준, 부호 있는 streak. 예: +3 = 3일 연속 양봉)
    direction = np.sign(body).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    df["candle_direction_streak"] = streak * direction

    return df


# ------------------------------------------------------------------
# 2. BASE + VALUATION + CANDLE 통합 데이터셋 생성
# ------------------------------------------------------------------
def build_feature_dataset_with_candlestick(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 20,
    cost_threshold: float = 0.012,
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)

    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)
    df = add_candlestick_features(df)
    df = add_label(df, horizon=horizon, cost_threshold=cost_threshold)

    valuation_df = load_valuation(ticker_krx, start, end)
    df = add_valuation_features(df, valuation_df)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_BASE,
        *FEATURE_COLS_VALUATION,
        *FEATURE_COLS_CANDLE,
        "future_return",
        "label",
    ]

    result = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return result


if __name__ == "__main__":
    HORIZON = 20
    COST_THRESHOLD = 0.012  # DEFAULT_HORIZON_COST_MAP[20] (feature_engineering_valuation.py)와 동일

    dataset = build_feature_dataset_with_candlestick(horizon=HORIZON, cost_threshold=COST_THRESHOLD)
    print(f"생성된 데이터셋 shape: {dataset.shape}")
    print(f"기간: {dataset.index.min().date()} ~ {dataset.index.max().date()}")

    print(f"\n캔들 패턴 발생 빈도 (전체 대비 비율):")
    for col in ["is_doji", "is_hammer", "is_shooting_star",
                "is_bullish_engulfing", "is_bearish_engulfing", "is_marubozu"]:
        print(f"  {col}: {dataset[col].mean():.1%}")

    out_path = DATA_DIR / "064350_features_with_candlestick_h20.csv"
    dataset.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")