"""
[재검증] 고정된 '2025년 제외' 대신, 종목별로 실제 최대 기여 연도를 찾아서 제외.

check_trade_concentration.py에서 확인됨: 118990의 최대 기여 연도는 2020년인데
(618350은 2025년), 지금까지 국면검증은 모든 종목에 똑같이 '2025년 제외'를
적용해서 118990의 진짜 쏠림 연도(2020년)를 한 번도 걸러낸 적이 없었음.
예전 COMBINED 실험에서 118990이 '2020년 74.9% 집중'으로 문제됐던 것과
동일한 패턴이 BASE 전용에서도 재현되는지 여기서 직접 확인.

방법: 각 (종목, threshold, seed) 조합의 거래를 연도별로 분해해서 최대 기여
연도를 자동으로 찾고, 그 연도의 거래를 제외한 뒤 남은 거래들의 복리수익률이
같은 기간 Buy & Hold를 여전히 이기는지 재확인.

주의: '행 삭제'가 아니라 그 연도에 발생한 거래만 제외하고 나머지 거래는 그대로
복리로 이어붙임 (Buy & Hold도 같은 방식으로 그 연도를 건너뛴 값으로 재계산--
과거에 '행 삭제 vs 기간 절단' 실수가 있었던 부분이라 이번엔 거래 단위로 정확히 맞춤).

전제:
    backtest_base_only_triple_barrier.py, check_trade_concentration.py와
    같은 폴더(src/)에 있어야 함.

사용법 (레포 루트에서):
    python src/regime_exclude_dominant_year.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_base_only_triple_barrier import generate_trades, buy_and_hold_return, DATA_DIR

TARGETS = [
    ("118990", 0.55), ("118990", 0.60), ("118990", 0.65),
    ("064350", 0.60),
]
SEEDS = [42, 1, 7, 123, 2024]


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def find_dominant_year(df: pd.DataFrame, threshold: float, seeds: list) -> int:
    """seed별로 최대 기여 연도를 구하고, 그중 가장 자주 나온 연도를 그 종목의 대표 쏠림 연도로 채택."""
    year_votes = {}
    for seed in seeds:
        trades = generate_trades(df, threshold, seed)
        if trades.empty:
            continue
        trades["year"] = trades["entry_date"].dt.year
        yearly = trades.groupby("year")["net_return"].apply(lambda r: (1 + r).prod() - 1)
        top_year = yearly.abs().idxmax()
        year_votes[top_year] = year_votes.get(top_year, 0) + 1
    return max(year_votes, key=year_votes.get)


def bh_return_excluding_year(df: pd.DataFrame, exclude_year: int) -> float:
    """제외 연도를 '건너뛰고' 이어붙인 Buy & Hold -- 그 해의 시작/종료 구간 수익률만 뺌."""
    before = df[df.index.year < exclude_year]
    after = df[df.index.year > exclude_year]
    ret = 1.0
    if len(before) > 1:
        ret *= before["Close"].iloc[-1] / before["Close"].iloc[0]
    if len(after) > 1:
        ret *= after["Close"].iloc[-1] / after["Close"].iloc[0]
    return ret - 1


def evaluate_excluding_year(ticker_krx: str, threshold: float, exclude_year: int):
    df = load_dataset(ticker_krx)
    bh_excl = bh_return_excluding_year(df, exclude_year)

    rows = []
    for seed in SEEDS:
        trades = generate_trades(df, threshold, seed)
        trades["year"] = trades["entry_date"].dt.year
        remaining = trades[trades["year"] != exclude_year]

        net_total = (1 + remaining["net_return"]).prod() - 1
        rows.append({
            "seed": seed,
            "n_trades_excl": len(remaining),
            "n_trades_removed": len(trades) - len(remaining),
            "net_total_return_excl": net_total,
            "bh_return_excl": bh_excl,
            "beats_bh": net_total > bh_excl,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for ticker_krx, threshold in TARGETS:
        df = load_dataset(ticker_krx)
        dominant_year = find_dominant_year(df, threshold, SEEDS)

        print(f"\n{'#' * 60}\n# {ticker_krx}, threshold={threshold} "
              f"-- 최대기여연도({dominant_year}) 제외 재검증\n{'#' * 60}")

        result = evaluate_excluding_year(ticker_krx, threshold, dominant_year)
        print(result.round(4).to_string(index=False))

        win_seeds = int(result["beats_bh"].sum())
        print(f"\n{dominant_year}년 제외 시 5-seed 결과: {win_seeds}/5 통과")
        if win_seeds >= 4:
            print(f"-> {dominant_year}년 없이도 버팀: 진짜 신호일 가능성 높음")
        elif win_seeds == 0:
            print(f"-> {dominant_year}년 하나에 전적으로 기댄 결과: [실패]로 재분류 권장")
        else:
            print(f"-> 애매함({win_seeds}/5): 좀 더 보수적으로 판단할 것")