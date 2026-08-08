"""
분석에서 드러난 국면 지배 연도(현대로템 2025, 대한제강 2018)의 거래를 사후적으로 제거하고
나머지 거래들만으로 복리수익을 재계산.

목적:
    analyze_signal_concentration_fx.py에서 "특정 연도 하나가 log-return의 절반 이상을
    설명한다"는 게 확인됐음. 근데 그 연도가:
        - 현대로템: 데이터 맨 끝(2025)이라 build_feature_dataset(end=...)로 잘라낼 수 있음
        - 대한제강: 데이터 중간(2018)이라 끝만 잘라내는 방식이 안 통함
    그래서 이 스크립트는 재학습 없이, 이미 생성된 거래 목록에서 해당 연도의 거래만
    제거하고 나머지로 복리수익을 재계산하는 방식으로 통일함 -- 두 케이스에 동일하게 적용 가능.

    ⚠️ 주의: 이건 "그 연도에 모델이 아예 없었다면 어땠을지"를 근사하는 게 아니라
    "그 연도 거래들의 손익 기여분만 빼면 나머지는 어떤지"를 보는 거라 완벽한 재검증은 아님.
    (모델은 여전히 그 연도를 포함한 데이터로 학습됐을 수 있음 -- walk-forward 특성상
    학습 구간과 그 연도가 겹치지 않는 fold도 있어서 완전히 동일하진 않지만, 엄밀한 인과
    분리는 아니라는 점을 감안하고 봐야 함)

사용법:
    python backtest_fx_excl_regime_year.py
"""

import pandas as pd
import numpy as np

from backtest_comparison_fx import generate_trades, evaluate_strategy, HORIZON
from train_xgboost_ablation_fx import load_dataset, FEATURE_COLS_BASE, FEATURE_COLS_COMBINED


def run_excl_year(ticker_krx: str, ticker_name: str, excl_year: int, threshold: float = 0.55, horizon: int = HORIZON):
    print(f"\n{'#' * 70}")
    print(f"# {ticker_name} ({ticker_krx}) -- {excl_year}년 거래 제외")
    print(f"{'#' * 70}")

    df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)

    rows = []
    for label, feature_cols in [("BASE", FEATURE_COLS_BASE), ("COMBINED", FEATURE_COLS_COMBINED)]:
        trades, _ = generate_trades(df, feature_cols, threshold=threshold, horizon=horizon, embargo=horizon)
        trades = trades.copy()
        trades["entry_year"] = pd.to_datetime(trades["entry_date"]).dt.year

        full_result = evaluate_strategy(trades, f"{label}_전체")

        trades_excl = trades[trades["entry_year"] != excl_year]
        excl_result = evaluate_strategy(trades_excl, f"{label}_{excl_year}제외")

        n_excluded = len(trades) - len(trades_excl)
        print(f"  {label}: 전체 {len(trades)}건 중 {excl_year}년 {n_excluded}건 제외 -> {len(trades_excl)}건 남음")

        rows.append(full_result)
        rows.append(excl_result)

    summary = pd.DataFrame(rows).set_index("label")
    cols = ["n_trades", "win_rate", "avg_net_return", "total_compound_return", "sharpe_per_trade", "mdd"]

    print(f"\n{summary[cols].round(4).to_string()}")

    base_full = summary.loc["BASE_전체", "total_compound_return"]
    base_excl = summary.loc[f"BASE_{excl_year}제외", "total_compound_return"]
    comb_full = summary.loc["COMBINED_전체", "total_compound_return"]
    comb_excl = summary.loc[f"COMBINED_{excl_year}제외", "total_compound_return"]

    print(f"\n--- {excl_year}년 제외 전/후 비교 ---")
    print(f"BASE:     {base_full:.1%} -> {base_excl:.1%}")
    print(f"COMBINED: {comb_full:.1%} -> {comb_excl:.1%}")

    if comb_excl > base_excl:
        print(f"→ {excl_year}년 빼도 COMBINED가 BASE보다 나음 -- 완전히 그 해 하나에만 기댄 결과는 아님")
    else:
        print(f"→ {excl_year}년 빼면 COMBINED가 BASE보다 못함(또는 역전) -- 우위가 그 해 하나에 사실상 전적으로 의존했다는 뜻")

    return summary


if __name__ == "__main__":
    summary_rotem = run_excl_year(ticker_krx="064350", ticker_name="현대로템", excl_year=2025)
    summary_daehan = run_excl_year(ticker_krx="084010", ticker_name="대한제강", excl_year=2018)