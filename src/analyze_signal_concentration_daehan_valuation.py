"""
진짜 대한제강(084010) h=20 백테스트에서 BASE/VALUATION_ONLY/COMBINED 셋 다 Buy & Hold를
이겼는데, 국면(특정 연도) 우연에 기댄 결과인지 확인하기 위한 연도별 분해.
(analyze_signal_concentration.py 패턴 + analyze_signal_concentration_foreign_own.py의
 "여러 feature set 비교" 패턴을 합쳐서 재사용)

사용법:
    python analyze_signal_concentration_daehan_valuation.py

전제:
    backtest_valuation_comparison.py, train_xgboost_valuation_ablation.py와 같은 폴더에 있어야 함.
"""

import pandas as pd
import numpy as np

from backtest_valuation_comparison import generate_trades
from train_xgboost_valuation_ablation import (
    load_dataset, FEATURE_COLS_BASE, FEATURE_COLS_VALUATION_ONLY, FEATURE_COLS_COMBINED,
)


def analyze_by_year(ticker_krx: str, feature_cols: list, label: str, horizon: int, threshold: float = 0.55):
    df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
    trades, _ = generate_trades(df, feature_cols, horizon=horizon, threshold=threshold)

    if trades.empty:
        print(f"{ticker_krx} ({label}): 거래 없음")
        return None

    trades = trades.copy()
    trades["entry_year"] = pd.to_datetime(trades["entry_date"]).dt.year

    rows = []
    for year, group in trades.groupby("entry_year"):
        n_trades = len(group)
        avg_net_return = group["net_return"].mean()
        year_compound = (1 + group["net_return"]).prod() - 1

        year_prices = df["Close"][df.index.year == year]
        if len(year_prices) > 1:
            stock_year_return = year_prices.iloc[-1] / year_prices.iloc[0] - 1
        else:
            stock_year_return = np.nan

        rows.append({
            "year": year,
            "n_trades": n_trades,
            "avg_net_return": avg_net_return,
            "year_compound_contribution": year_compound,
            "stock_price_return_that_year": stock_year_return,
        })

    year_df = pd.DataFrame(rows).sort_values("year")

    total_compound = (1 + trades["net_return"]).prod() - 1
    year_df["share_of_total_log_return"] = (
        np.log1p(year_df["year_compound_contribution"]) / np.log1p(total_compound)
    )

    print(f"\n=== {ticker_krx} ({label}, horizon={horizon}) 연도별 분해 ===")
    print(f"전체 복리수익률: {total_compound:.1%}, 전체 거래 수: {len(trades)}\n")
    print(year_df.round(4).to_string(index=False))

    top_year = year_df.loc[year_df["share_of_total_log_return"].idxmax()]
    print(f"\n최대 기여 연도: {int(top_year['year'])}년 "
          f"(전체 log-return 기여도의 {top_year['share_of_total_log_return']:.1%}, "
          f"그 해 주가 자체 등락률 {top_year['stock_price_return_that_year']:.1%})")

    if abs(top_year["share_of_total_log_return"]) > 0.5:
        print("→ ⚠️ 특정 연도 하나가 전체 수익의 절반 이상을 차지함 -- 국면 의존적 신호일 가능성 높음")
    else:
        print("→ 여러 연도에 고르게 분산됨 -- 특정 구간 우연에 기댄 결과는 아닌 것으로 보임")

    negative_years = (year_df["year_compound_contribution"] < 0).sum()
    positive_years = (year_df["year_compound_contribution"] > 0).sum()
    print(f"손실 연도: {negative_years}개 / 이익 연도: {positive_years}개 (총 {len(year_df)}개 연도)")

    return year_df


if __name__ == "__main__":
    TICKER_KRX = "084010"
    HORIZON = 20

    for label, feature_cols in [
        ("BASE", FEATURE_COLS_BASE),
        ("VALUATION_ONLY", FEATURE_COLS_VALUATION_ONLY),
        ("COMBINED", FEATURE_COLS_COMBINED),
    ]:
        print("#" * 70)
        print(f"# {label}")
        print("#" * 70)
        analyze_by_year(TICKER_KRX, feature_cols, label, horizon=HORIZON)
        print("\n")