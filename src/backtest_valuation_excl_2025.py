"""
2025년(이례적 강세장) 구간을 제외하고, 2015~2024년까지만으로 COMBINED vs BASE vs Buy&Hold 재검증.

목적: 아까 나온 "COMBINED가 Buy&Hold를 이겼다"는 결과가 2025년 이상급등장 하나의 국면 덕인지,
     아니면 그 구간을 빼도 재현되는 진짜 엣지인지 가리기 위함.

사용법:
    python backtest_valuation_excl_2025.py
"""

import pandas as pd
import numpy as np

from feature_engineering_valuation import build_feature_dataset_with_valuation
from train_xgboost_valuation_ablation import FEATURE_COLS_BASE, FEATURE_COLS_COMBINED
from backtest_valuation_comparison import (
    generate_trades, evaluate_strategy, buy_and_hold_benchmark, walk_forward_splits,
)


def run_excl_2025(ticker: str, ticker_krx: str, horizon: int = 10, threshold: float = 0.55,
                   cost_threshold: float = 0.008, end: str = "2024-12-31"):
    print(f"\n{'#' * 70}")
    print(f"# {ticker_krx} -- {end}까지만 (2025년 이례적 강세장 제외)")
    print(f"{'#' * 70}")

    df = build_feature_dataset_with_valuation(
        ticker=ticker, ticker_krx=ticker_krx, horizon=horizon,
        cost_threshold=cost_threshold, end=end,
    )
    print(f"전체 데이터: {df.shape[0]}행, 기간: {df.index.min().date()} ~ {df.index.max().date()}")

    feature_sets = {"BASE": FEATURE_COLS_BASE, "COMBINED": FEATURE_COLS_COMBINED}
    results = []
    for label, feature_cols in feature_sets.items():
        trades, n_signals = generate_trades(df, feature_cols, horizon=horizon, threshold=threshold)
        print(f"  {label}: 신호 {n_signals}개 / 거래 성사 {len(trades)}건")
        results.append(evaluate_strategy(trades, label))

    bh_result = buy_and_hold_benchmark(df, horizon=horizon)
    results.append(bh_result)

    summary = pd.DataFrame(results).set_index("label")
    cols = ["n_trades", "win_rate", "avg_net_return", "total_compound_return", "sharpe_per_trade", "mdd", "return_per_risk"]

    print(f"\n검증 기간: {bh_result['period']}")
    print(summary[cols].round(4).to_string())

    return summary


if __name__ == "__main__":
    summary_rotem = run_excl_2025(ticker="064350.KS", ticker_krx="064350")
    summary_ss = run_excl_2025(ticker="005930.KS", ticker_krx="005930")

    print("\n\n" + "=" * 70)
    print("=== 2025년 제외 후 COMBINED vs Buy&Hold 요약 ===")
    print("=" * 70)
    for label, summary in [("현대로템", summary_rotem), ("삼성전자", summary_ss)]:
        combined_ret = summary.loc["COMBINED", "total_compound_return"]
        base_ret = summary.loc["BASE", "total_compound_return"]
        bh_ret = summary.loc["Buy & Hold", "total_compound_return"]
        print(f"{label}: BASE {base_ret:.1%} / COMBINED {combined_ret:.1%} / Buy&Hold {bh_ret:.1%}")
        if combined_ret > bh_ret:
            print("  → 2025년 없이도 COMBINED가 Buy&Hold를 이김 -- 국면 우연이 아닐 가능성 있음")
        else:
            print("  → 2025년 빼니 COMBINED가 Buy&Hold를 못 이김 -- 아까 결과는 국면(2025 강세장) 의존적이었을 가능성 높음")