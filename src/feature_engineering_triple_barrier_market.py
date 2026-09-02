"""
Triple-barrier 라벨 + BASE + VALUATION(개별) + MARKET_VALUATION(코스피 지수) COMBINED.

[배경] BASE+VALUATION+MARKET_VAL ablation(h=20, 이진분류 기준)에서 COMBINED가 BASE
대비 4/5 시드 승, AUC diff +0.0056 (약하지만 부호 안 뒤집힘, rate_spread와 다름)
-- triple-barrier로 실제 백테스트까지 가보기로 함.

⚠️ pt_sl=(2,1), num_days=20은 사전등록된 값 그대로 유지 (재선택 안 함).
build_triple_barrier_dataset_combined()의 BASE+VALUATION 로직을 100% 재사용하고
그 위에 코스피 지수 PER/PBR feature 4개만 추가로 merge.

사용법 (레포 루트에서):
    python src/feature_engineering_triple_barrier_market.py

전제:
    feature_engineering_triple_barrier.py, feature_engineering_market_valuation.py가
    같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering_triple_barrier import (
    build_triple_barrier_dataset_combined, FEATURE_COLS_BASE, FEATURE_COLS_VALUATION,
)
from feature_engineering_market_valuation import load_market_valuation, add_market_valuation_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_MARKET_VAL = [
    "market_per", "market_pbr",
    "market_per_zscore_252d", "market_pbr_zscore_252d",
]


def build_triple_barrier_dataset_combined_market(
    ticker: str,
    ticker_krx: str,
    pt_sl: tuple = (2, 1),
    num_days: int = 20,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    market: str = "KOSPI",  # 추가
) -> pd.DataFrame:
    df = build_triple_barrier_dataset_combined(
        ticker=ticker, ticker_krx=ticker_krx, pt_sl=pt_sl, num_days=num_days, start=start, end=end,
    )

    market_df = load_market_valuation(start, end, market=market)  # market 전달
    df = add_market_valuation_features(df, market_df)

    keep_cols = (
        ["Close", "Volume"] + FEATURE_COLS_BASE + FEATURE_COLS_VALUATION + FEATURE_COLS_MARKET_VAL +
        ["future_return", "label_fixed", "label_tb", "ret_tb", "holding_rows_tb"]
    )
    result = df[keep_cols].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=FEATURE_COLS_BASE + FEATURE_COLS_VALUATION + FEATURE_COLS_MARKET_VAL + ["label_tb"]
    )
    return result


if __name__ == "__main__":
    VALIDATED_TICKERS = {
        "064350": ("064350.KS", "KOSPI"),   # 현대로템
        "052690": ("052690.KS", "KOSPI"),   # 한전기술
        "118990": ("118990.KQ", "KOSDAQ"),  # 모트렉스
    }
    PT_SL = (2, 1)
    NUM_DAYS = 20
    CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"

    for TICKER_KRX, (TICKER, MARKET) in VALIDATED_TICKERS.items():
        print(f"\n{'#' * 60}\n# {TICKER_KRX} ({TICKER}, {MARKET})\n{'#' * 60}")

        dataset = build_triple_barrier_dataset_combined_market(
            ticker=TICKER, ticker_krx=TICKER_KRX, pt_sl=PT_SL, num_days=NUM_DAYS, market=MARKET,
        )
        print(f"생성된 데이터셋 shape: {dataset.shape}")

        out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_valuation_market.csv"
        dataset.to_csv(out_path)
        print(f"저장 완료: {out_path}")

    print("\n\n3종목 전부 생성 완료 -- backtest_triple_barrier_market_pooled.py로 검증할 것.")