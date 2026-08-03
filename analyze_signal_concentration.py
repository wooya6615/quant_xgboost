"""
COMBINED(BASE+밸류에이션) 전략의 거래를 연도별로 쪼개서 확인:
    - 총 복리수익의 대부분이 특정 연도 하나에 몰려있는지
    - 그 연도가 실제로 강한 상승장(벤치마크 대비)이었는지
    - 두 종목(현대로템/삼성전자)의 결과가 반대로 나온 게 "국면 우연"인지 확인하기 위함

사용법:
    python analyze_signal_concentration.py
"""

import pandas as pd
import numpy as np

from backtest_valuation_comparison import generate_trades
from train_xgboost_valuation_ablation import load_dataset, FEATURE_COLS_COMBINED


def analyze_by_year(ticker_krx: str, horizon: int = 10, threshold: float = 0.55):
    df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
    trades, _ = generate_trades(df, FEATURE_COLS_COMBINED, horizon=horizon, threshold=threshold)

    if trades.empty:
        print(f"{ticker_krx}: 거래 없음")
        return None

    trades = trades.copy()
    trades["entry_year"] = pd.to_datetime(trades["entry_date"]).dt.year

    # 연도별 거래 수 / 평균 순수익률 / 그 해 주가 자체 등락률(bull/bear 국면 확인용)
    rows = []
    for year, group in trades.groupby("entry_year"):
        n_trades = len(group)
        avg_net_return = group["net_return"].mean()
        # 이 연도의 복리 기여도 (그 해 거래들만 순차 복리로 가정)
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

    # 전체 복리수익 대비 각 연도의 상대적 기여도 (연도별 compound를 다 곱하면 전체 복리가 나옴)
    total_compound = (1 + trades["net_return"]).prod() - 1
    year_df["share_of_total_log_return"] = (
        np.log1p(year_df["year_compound_contribution"]) / np.log1p(total_compound)
    )

    print(f"\n=== {ticker_krx} (horizon={horizon}) 연도별 분해 ===")
    print(f"전체 복리수익률: {total_compound:.1%}, 전체 거래 수: {len(trades)}\n")
    print(year_df.round(4).to_string(index=False))

    # 집중도 판정: 특정 연도 하나가 log-return 기여도의 50% 이상이면 "집중"으로 표시
    top_year = year_df.loc[year_df["share_of_total_log_return"].idxmax()]
    print(f"\n최대 기여 연도: {int(top_year['year'])}년 "
          f"(전체 log-return 기여도의 {top_year['share_of_total_log_return']:.1%}, "
          f"그 해 주가 자체 등락률 {top_year['stock_price_return_that_year']:.1%})")

    if top_year["share_of_total_log_return"] > 0.5:
        print("→ ⚠️ 특정 연도 하나가 전체 수익의 절반 이상을 차지함 -- 국면 의존적 신호일 가능성 높음")
    else:
        print("→ 여러 연도에 고르게 분산됨 -- 특정 구간 우연에 기댄 결과는 아닌 것으로 보임")

    return year_df


if __name__ == "__main__":
    print("#" * 70)
    print("# 현대로템 (064350)")
    print("#" * 70)
    analyze_by_year("064350", horizon=10)

    print("\n\n" + "#" * 70)
    print("# 삼성전자 (005930)")
    print("#" * 70)
    analyze_by_year("005930", horizon=10)