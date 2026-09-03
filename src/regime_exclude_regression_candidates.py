"""
회귀(ret_tb) 백테스트 후보들의 국면검증 -- 자체 최대기여연도 제외 + 거래집중도.

backtest_regression_entry_ab.py 결과에서 win_seeds==5(또는 근접)로 나온 후보 중
바로 채택하면 안 되는 것들을 직접 검증:
    118990, A_threshold=0.01 (5/5, mean_net_return +918.66%) -- 절대수익률이 커서
        진짜일 가능성 있어 보이지만, 118990 자체가 Buy&Hold=-28.94%(벤치마크 붕괴)라
        118990 관련 결과는 전부 국면검증 없이는 못 믿음(과거 COMBINED 2020년
        74.9% 집중 전례와 동일 원칙)
    118990, B_top_pct=10% (5/5이지만 mean_net_return +2.76% -- 거의 본전, 이 5/5는
        "벤치마크가 더 많이 무너져서 상대적으로 이긴" 착시일 가능성이 높음. 그래도
        원칙대로 직접 확인)
    052690, B_top_pct=10% (4/5, mean_net_return +378.13% -- 052690은 분류에서
        전체기간부터 0/5였던 종목이라 국면검증까지 반드시 거쳐야 함)

방법론은 regime_exclude_dominant_year.py(분류용)와 동일 -- seed별 최대 기여 연도를
찾아 다수결로 대표 연도를 정하고, 그 연도 거래만 제외한 뒤 남은 거래가 여전히
(그 연도를 제외한) Buy & Hold를 이기는지 확인. 거래 단위 제외라 '행 삭제'가 아님.

사용법 (레포 루트에서):
    python src/regime_exclude_regression_candidates.py

전제:
    backtest_regression_entry_ab.py와 같은 폴더(src/)에 있어야 함(함수 재사용).
"""

import numpy as np
import pandas as pd

from backtest_regression_entry_ab import (
    TICKERS, SEEDS, load_or_build_dataset, buy_and_hold_return,
    compute_fold_predictions, generate_trades_threshold, generate_trades_top_pct,
)

# (ticker_krx, method, candidate)
TARGETS = [
    ("118990", "A_threshold", 0.01),
    ("118990", "B_top_pct", 10),
    ("052690", "B_top_pct", 10),
]


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


def generate_trades_for_method(df, fold_results, method, candidate):
    if method == "A_threshold":
        return generate_trades_threshold(df, fold_results, candidate)
    elif method == "B_top_pct":
        return generate_trades_top_pct(df, fold_results, candidate)
    raise ValueError(method)


def find_dominant_year(df, fold_results_by_seed, method, candidate, seeds):
    year_votes = {}
    for seed in seeds:
        trades = generate_trades_for_method(df, fold_results_by_seed[seed], method, candidate)
        if trades.empty:
            continue
        trades["year"] = trades["entry_date"].dt.year
        yearly = trades.groupby("year")["net_return"].apply(lambda r: (1 + r).prod() - 1)
        top_year = yearly.abs().idxmax()
        year_votes[top_year] = year_votes.get(top_year, 0) + 1
    return max(year_votes, key=year_votes.get) if year_votes else None


if __name__ == "__main__":
    for ticker_krx, method, candidate in TARGETS:
        ticker = TICKERS[ticker_krx]
        df = load_or_build_dataset(ticker_krx, ticker)
        bh_full = buy_and_hold_return(df)

        print(f"\n{'#' * 60}")
        print(f"# {ticker_krx}, {method}={candidate} -- Buy&Hold(전체기간)={bh_full:+.2%}")
        print(f"{'#' * 60}")

        fold_results_by_seed = {seed: compute_fold_predictions(df, seed) for seed in SEEDS}

        dominant_year = find_dominant_year(df, fold_results_by_seed, method, candidate, SEEDS)
        print(f"최대 기여 연도: {dominant_year}")

        bh_excl = bh_return_excluding_year(df, dominant_year)
        excl_rows = []
        for seed in SEEDS:
            trades = generate_trades_for_method(df, fold_results_by_seed[seed], method, candidate)
            trades["year"] = trades["entry_date"].dt.year
            remaining = trades[trades["year"] != dominant_year]
            net_total = (1 + remaining["net_return"]).prod() - 1 if not remaining.empty else np.nan
            excl_rows.append({
                "seed": seed, "n_trades_excl": len(remaining),
                "net_total_return_excl": net_total, "bh_return_excl": bh_excl,
                "beats_bh": (net_total > bh_excl) if pd.notna(net_total) else False,
            })
        excl_df = pd.DataFrame(excl_rows)
        print(f"\n=== {dominant_year}년 제외 재검증 ===")
        print(excl_df.round(4).to_string(index=False))

        win_seeds = int(excl_df["beats_bh"].sum())
        print(f"\nBuy&Hold({dominant_year}년 제외): {bh_excl:+.2%} "
              f"{'(벤치마크 붕괴 -- 절대수익률 부호를 볼 것)' if bh_excl < -0.3 else ''}")
        print(f"{dominant_year}년 제외 시 결과: {win_seeds}/{len(SEEDS)} 통과")
        if win_seeds >= 4:
            print("-> 지배 연도 없이도 버팀: 진짜 신호일 가능성 높음")
        elif win_seeds == 0:
            print("-> 지배 연도 하나에 전적으로 기댄 결과: [실패]로 재분류 권장")
        else:
            print(f"-> 애매함({win_seeds}/{len(SEEDS)}): 좀 더 보수적으로 판단할 것")

        # 거래 집중도 (seed=42/1)
        print(f"\n=== 거래 집중도 (seed=42/1) ===")
        for seed in [42, 1]:
            trades = generate_trades_for_method(
                df, fold_results_by_seed[seed], method, candidate
            ).sort_values("entry_date")
            if trades.empty:
                print(f"  seed={seed}: 거래 없음")
                continue
            total = (1 + trades["net_return"]).prod() - 1
            by_size = trades.sort_values("net_return", ascending=False)
            top5_returns = by_size.head(5)["net_return"].tolist()
            remaining5 = trades.drop(by_size.head(5).index)
            remaining5_return = (1 + remaining5["net_return"]).prod() - 1 if not remaining5.empty else np.nan
            print(f"  seed={seed}: 총 {len(trades)}건, 총수익 {total:+.2%}, "
                  f"상위5개 제외시 {remaining5_return:+.2%} -- {[f'{r:+.1%}' for r in top5_returns]}")

    print(f"\n{'=' * 60}")
    print("[판정 가이드]")
    print("- 국면제외 4/5+ & 거래집중도(상위5개 제외해도 부호 유지) & 시드 간 상위거래")
    print("  수익률이 비슷하면: 진짜 후보로 승격, threshold/top-pct PBO까지 진행")
    print("- 국면제외에서 무너지거나 절대수익률이 미미하면: 원래 IC 결과가 착시였다는")
    print("  쪽으로 결론, 이 티커/방식은 [실패]")