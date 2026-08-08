"""
COMBINED(BASE+외국인 보유율) 전략의 거래를 연도별로 쪼개서 확인:
    - 총 복리수익의 대부분이 특정 연도 하나에 몰려있는지
    - 그 연도가 실제로 강한 상승장/하락장(벤치마크 대비)이었는지
    - 두 종목 다 COMBINED가 BASE보다 못한 게 특정 국면 우연인지,
      아니면 여러 해에 걸쳐 골고루 나쁜(feature 자체의 문제) 것인지 확인하기 위함

사용법:
    python analyze_signal_concentration_foreign_own.py

전제:
    backtest_comparison_foreign_own.py, train_xgboost_ablation_foreign_own.py와 같은 폴더에 있어야 함.
"""

import pandas as pd
import numpy as np

from backtest_comparison_foreign_own import generate_trades, HORIZON
from train_xgboost_ablation_foreign_own import load_dataset, FEATURE_COLS_COMBINED, FEATURE_COLS_BASE


def analyze_by_year(ticker_krx: str, feature_cols: list, label: str, horizon: int = HORIZON, threshold: float = 0.55):
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
        print("→ 여러 연도에 고르게 분산됨 -- 특정 구간 우연에 기댄 결과는 아닌 것으로 보임 "
              "(즉, COMBINED가 BASE보다 못한 게 특정 해 하나 때문이 아니라 구조적일 가능성)")

    # 손실/이익에 기여한 해의 개수 확인 -- 골고루 나쁜지, 특정 해만 나쁜지
    negative_years = (year_df["year_compound_contribution"] < 0).sum()
    positive_years = (year_df["year_compound_contribution"] > 0).sum()
    print(f"손실 연도: {negative_years}개 / 이익 연도: {positive_years}개 (총 {len(year_df)}개 연도)")

    return year_df


if __name__ == "__main__":
    for ticker_krx, ticker_name in [("064350", "현대로템"), ("005930", "삼성전자")]:
        print("#" * 70)
        print(f"# {ticker_name} ({ticker_krx})")
        print("#" * 70)

        print("\n--- BASE ---")
        analyze_by_year(ticker_krx, FEATURE_COLS_BASE, "BASE")

        print("\n--- COMBINED ---")
        analyze_by_year(ticker_krx, FEATURE_COLS_COMBINED, "COMBINED")

        print("\n")