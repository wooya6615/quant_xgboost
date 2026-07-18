"""
방향성 분류 ML 모델을 위한 feature engineering
모멘텀 / 변동성 / 거래량 / 상대강도 4개 그룹 계산

사용법:
    python feature_engineering.py

주의:
    이 스크립트는 yfinance로 실제 인터넷에서 데이터를 받아옵니다.
    Claude 샌드박스 환경에서는 외부 금융 API 접속이 막혀 있어 실행이 안 되니,
    본인 로컬 환경(VSCode, 주피터 등)에서 실행하세요.
"""

import pandas as pd
import numpy as np
import yfinance as yf


# ------------------------------------------------------------------
# 1. 데이터 수집
# ------------------------------------------------------------------
def load_data(ticker: str, benchmark: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ticker: 종목 코드 (예: 삼성전자 = "005930.KS")
    benchmark: 상대강도 비교용 지수 (예: 코스피 = "^KS11")
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    bench = yf.download(benchmark, start=start, end=end, auto_adjust=True)

    # 컬럼이 멀티인덱스로 오는 경우 정리 (yfinance 최신 버전 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)

    df = df.dropna()
    bench = bench.dropna()
    return df, bench


# ------------------------------------------------------------------
# 2. 모멘텀 feature
# ------------------------------------------------------------------
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # 단순 수익률 (5/10/20일)
    for n in [5, 10, 20]:
        df[f"return_{n}d"] = close.pct_change(n)

    # RSI (14일)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, signal 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    return df


# ------------------------------------------------------------------
# 3. 변동성 feature
# ------------------------------------------------------------------
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # 역사적 변동성 (20일 수익률 표준편차, 연율화)
    daily_ret = close.pct_change()
    df["hist_vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)

    # 볼린저밴드 폭 (20일, 2 std)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_width"] = (upper - lower) / ma20
    df["bb_position"] = (close - lower) / (upper - lower)  # 밴드 내 위치 (0~1)

    # ATR (Average True Range, 14일)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    return df


# ------------------------------------------------------------------
# 4. 거래량 feature
# ------------------------------------------------------------------
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    volume = df["Volume"]

    # 거래량 변화율 (20일 평균 대비)
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio_20d"] = volume / vol_ma20

    # OBV (On-Balance Volume)
    direction = np.sign(close.diff()).fillna(0)
    df["obv"] = (direction * volume).cumsum()
    # OBV는 절대값보다 추세가 중요하므로 20일 변화율로 정규화
    df["obv_change_20d"] = df["obv"].pct_change(20)

    return df


# ------------------------------------------------------------------
# 5. 상대강도 feature (벤치마크 대비)
# ------------------------------------------------------------------
def add_relative_strength_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    stock_ret = df["Close"].pct_change()
    bench_ret = bench["Close"].pct_change()

    # 날짜 정렬 맞추기
    aligned = pd.DataFrame({"stock": stock_ret, "bench": bench_ret}).dropna()

    # 초과수익률 (5/20일 누적)
    for n in [5, 20]:
        stock_cum = (1 + aligned["stock"]).rolling(n).apply(lambda x: x.prod() - 1, raw=True)
        bench_cum = (1 + aligned["bench"]).rolling(n).apply(lambda x: x.prod() - 1, raw=True)
        df[f"excess_return_{n}d"] = (stock_cum - bench_cum).reindex(df.index)

    return df


# ------------------------------------------------------------------
# 6. Label 생성 (N일 후 방향성, 거래비용 반영)
# ------------------------------------------------------------------
def add_label(df: pd.DataFrame, horizon: int = 5, cost_threshold: float = 0.005) -> pd.DataFrame:
    """
    horizon일 후 수익률이 거래비용(cost_threshold)을 넘으면 1, 아니면 0
    → 미세한 등락은 애초에 '맞다'고 학습하지 않도록 필터링

    future_return: N일 후 실제 수익률(연속값) — 학습 feature로는 절대 쓰면 안 됨(라벨 자체이므로 룩어헤드).
                   대신 백테스트에서 실제 손익 계산할 때만 사용.
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["future_return"] = future_return
    df["label"] = (future_return > cost_threshold).astype(int)
    # 마지막 horizon일치는 미래 데이터가 없어 라벨 생성 불가 → 제거 대상으로 표시
    df.loc[df.index[-horizon:], "label"] = np.nan
    df.loc[df.index[-horizon:], "future_return"] = np.nan
    return df


# ------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------
def build_feature_dataset(
    ticker: str = "005930.KS",         # 엔비디아
    benchmark: str = "^KS11",     # 나스닥 종합지수
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 10,
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)

    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)
    df = add_label(df, horizon=horizon)

    feature_cols = [
        "Close",  # 백테스트 벤치마크(buy&hold) 계산용
        "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
        "hist_vol_20d", "bb_width", "bb_position", "atr_14",
        "volume_ratio_20d", "obv_change_20d",
        "excess_return_5d", "excess_return_20d",
        "future_return",  # 백테스트 전용 -- 절대 학습 feature로 쓰지 말 것 (룩어헤드)
        "label",
    ]

    # RSI 등에서 분모가 0이 되는 구간(inf) 방어 처리 -> NaN으로 바꿔서 제거
    result = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return result


if __name__ == "__main__":
    dataset = build_feature_dataset()
    print(f"생성된 feature 데이터셋 shape: {dataset.shape}")
    print(f"\n라벨 분포:\n{dataset['label'].value_counts(normalize=True)}")
    print(f"\n샘플:\n{dataset.tail()}")

    dataset.to_csv("005930_features.csv")
    print("\n저장 완료: 005930_features.csv")