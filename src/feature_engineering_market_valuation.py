"""
코스피 지수 전체 PER/PBR feature (개별종목 밸류에이션과 다른 축 -- "시장 전체가
비싼 국면인가"를 반영). 개별 PER/PBR/DIV가 이미 triple-barrier COMBINED에서
5/5 시드 통과했던 전례가 있어, 같은 전처리 패턴(원값 + 252일 z-score)을
그대로 재사용함.

데이터 출처: pykrx.stock.get_index_fundamental_by_date(fromdate, todate, ticker)
ticker="1001" = 코스피 종합주가지수.

⚠️ 개별 PER과 마찬가지로 발표 지연은 없지만(지수 PER은 당일 구성종목 재무제표
기준으로 매일 갱신), look-ahead 방지를 위해 FX/수급 실험과 동일하게 1일 shift.

사용법 (레포 루트에서):
    python src/feature_engineering_market_valuation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # KRX_ID/KRX_PW를 pykrx import 전에 환경변수로 로드

from pykrx import stock

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KOSPI_INDEX_TICKER = "1001"  # 코스피 종합주가지수


def load_market_valuation(
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    market: str = "KOSPI",  # "KOSPI" or "KOSDAQ"
) -> pd.DataFrame:
    fromdate = start.replace("-", "")
    todate = end.replace("-", "")

    index_ticker = "1001" if market == "KOSPI" else "2001"
    df = stock.get_index_fundamental_by_date(fromdate, todate, index_ticker)
    df = df.rename(columns={"PER": "market_per", "PBR": "market_pbr"})
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"{market} 지수 밸류에이션 데이터: {df.shape[0]}행, "
          f"{df.index.min().date()} ~ {df.index.max().date()}")

    return df[["market_per", "market_pbr"]]


def add_market_valuation_features(df: pd.DataFrame, market_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    m = market_df.shift(shift_days)

    result = df.join(m, how="left")
    result["market_per"] = result["market_per"].ffill()
    result["market_pbr"] = result["market_pbr"].ffill()

    result["market_per_zscore_252d"] = (
        (result["market_per"] - result["market_per"].rolling(252).mean())
        / result["market_per"].rolling(252).std()
    )
    result["market_pbr_zscore_252d"] = (
        (result["market_pbr"] - result["market_pbr"].rolling(252).mean())
        / result["market_pbr"].rolling(252).std()
    )
    return result


def build_multi_horizon_datasets_market_valuation(
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

    market_df = load_market_valuation(start, end)  # 한 번만 조회해서 재사용

    mkt_cols = ["market_per", "market_pbr", "market_per_zscore_252d", "market_pbr_zscore_252d"]

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_market_valuation_features(base, market_df)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=mkt_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = DATA_DIR / f"{ticker_krx}_features_with_market_valuation_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"
    HORIZONS = [1, 3, 5, 10, 20]

    build_multi_horizon_datasets_market_valuation(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)