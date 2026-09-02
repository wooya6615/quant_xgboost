"""
[확인] 118990 threshold=0.55, seed=123에서 순수익률이 다른 시드 대비 3~10배로
튄 이유 확인. 예전 밸류에이션 COMBINED 실험에서 118990 수익의 74.9%가 2020년
한 해에 집중됐던 전례가 있어서, 이번에도 특정 거래/연도 하나가 결과를 좌우하는지
PBO로 넘어가기 전에 반드시 확인.

방법: backtest_base_only_triple_barrier.py의 generate_trades()를 그대로 재사용해서
    (a) 거래 단위 기여도: 상위 1/3/5개 거래를 제외했을 때 총수익률이 얼마나 빠지는지
    (b) 연도 단위 기여도: 연도별 복리수익률 분해

전제:
    backtest_base_only_triple_barrier.py와 같은 폴더(src/)에 있고,
    data/{ticker}_features_triple_barrier_pt2sl1_nd20_hl_base.csv가 있어야 함.

사용법 (레포 루트에서):
    python src/check_trade_concentration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_base_only_triple_barrier import generate_trades, DATA_DIR

# 확인 대상: (종목, threshold, seed) -- 의심스러운 조합 + 비교용 정상 조합
TARGETS = [
    ("118990", 0.55, 123),   # 이상치로 의심되는 조합
    ("118990", 0.55, 42),    # 같은 threshold, 다른(정상으로 보이는) 시드 -- 비교용
    ("064350", 0.60, 42),    # 통과 확정된 조합 -- 대조군으로 같이 확인
]


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def analyze(ticker_krx: str, threshold: float, seed: int):
    df = load_dataset(ticker_krx)
    trades = generate_trades(df, threshold, seed)
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    total_return = (1 + trades["net_return"]).prod() - 1
    print(f"\n{'#' * 60}\n# {ticker_krx}, threshold={threshold}, seed={seed}\n{'#' * 60}")
    print(f"총 거래 수: {len(trades)}, 총 순수익률: {total_return:+.2%}")

    # --- (a) 거래 단위 기여도: 상위 K개 거래 제외했을 때 총수익률 변화 ---
    by_size = trades.sort_values("net_return", ascending=False)
    for k in [1, 3, 5]:
        remaining = trades.drop(by_size.head(k).index)
        remaining_return = (1 + remaining["net_return"]).prod() - 1
        top_k_returns = by_size.head(k)["net_return"].tolist()
        print(f"  상위 {k}개 거래 제외 시 총수익률: {remaining_return:+.2%} "
              f"(제외 전 {total_return:+.2%} 대비 낙폭 {total_return - remaining_return:+.2%}p) "
              f"-- 제외된 거래 수익률: {[f'{r:+.1%}' for r in top_k_returns]}")

    # --- (b) 연도 단위 기여도 ---
    trades["year"] = trades["entry_date"].dt.year
    print("\n  연도별 복리수익률 분해:")
    yearly_rows = []
    for year, g in trades.groupby("year"):
        yr_return = (1 + g["net_return"]).prod() - 1
        yearly_rows.append({"year": year, "n_trades": len(g), "yearly_return": yr_return})
    yearly_df = pd.DataFrame(yearly_rows)
    print(yearly_df.to_string(index=False))

    max_year_row = yearly_df.loc[yearly_df["yearly_return"].abs().idxmax()]
    print(f"\n  최대 기여 연도: {int(max_year_row['year'])} "
          f"(그 해 수익률 {max_year_row['yearly_return']:+.2%}, "
          f"거래 {int(max_year_row['n_trades'])}건)")

    return trades, yearly_df


if __name__ == "__main__":
    for ticker_krx, threshold, seed in TARGETS:
        analyze(ticker_krx, threshold, seed)

    print(f"\n{'=' * 60}")
    print("[판정 가이드]")
    print("- 상위 1개 거래 제외만으로 총수익률이 크게 꺾이면(예: 50% 이상 빠지면):")
    print("  단일 거래 우연에 기댄 결과 -- 그 (threshold, seed) 조합은 신뢰하지 말 것")
    print("- 특정 한 해가 전체 수익의 대부분(예: 70%+)을 차지하면: 국면 쏠림 재확인 필요")
    print("  (118990은 예전 COMBINED 실험에서 2020년 74.9% 집중 전례가 있었음)")
    print("- 118990의 threshold=0.55/seed=123 대 seed=42를 나란히 비교해서, 정상 시드에도")
    print("  없는 극단적 단일거래/단일연도 쏠림이 123에만 있다면 그 결과는 버리고")
    print("  나머지 4개 시드 기준으로 재판정할 것")