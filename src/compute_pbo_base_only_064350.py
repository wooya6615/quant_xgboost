"""
PBO(Probability of Backtest Overfitting) -- 064350 BASE 전용, threshold 축.

기존 compute_pbo.py(3종목 풀링 + VALUATION-COMBINED)와 방법론(CSCV, Bailey et
al. 2014)은 동일하되, 이번엔 (a) 064350 단독 종목, (b) BASE 전용(밸류에이션
제외) 데이터로 검증. BASE 재검증 결과 052690/118990은 [실패]로 확정됐고
064350만 생존했으므로, 풀링 없이 단일 종목으로 threshold 축 과적합 여부만 확인.

threshold 후보 5개(0.50/0.55/0.60/0.65/0.70) -- backtest_base_only_triple_
barrier.py에서 스윕했던 0.55/0.60/0.65에 양쪽 끝(0.50/0.70)을 추가해서 탐색
범위를 실제보다 넓게 잡음 (PBO를 보수적으로 계산하기 위함).

사용법 (레포 루트에서):
    python src/compute_pbo_base_only_064350.py

전제:
    verify_base_only_triple_barrier.py로 064350_features_triple_barrier_
    pt2sl1_nd20_hl_base.csv가 생성돼 있어야 함.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_base_only_triple_barrier import generate_trades, DATA_DIR

TICKER_KRX = "064350"
CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SEED = 42  # PBO 계산 자체는 단일 시드로 -- 시드 강건성은 이미 5-seed 백테스트에서 별도 확인됨
N_SUBPERIODS = 10  # 짝수, C(10,5)=252


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def build_performance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """구간(row) x threshold(column) 행렬. 값 = 그 구간에서 그 threshold의 복리수익률."""
    start, end = df.index.min(), df.index.max()
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)

    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=CANDIDATE_THRESHOLDS, dtype=float)

    for threshold in CANDIDATE_THRESHOLDS:
        print(f"threshold={threshold} 거래 생성 중...")
        trades = generate_trades(df, threshold, SEED)
        if trades.empty:
            perf[threshold] = 0.0
            continue
        # generate_trades()는 entry_date만 반환하므로, holding_rows로 근사 청산일 계산
        trades = trades.copy()
        trades["exit_date"] = trades["entry_date"] + pd.to_timedelta(trades["holding_rows"], unit="D")

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
            perf.loc[s, threshold] = (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0

    return perf


def cscv_pbo(perf: pd.DataFrame) -> dict:
    n_periods = perf.shape[0]
    half = n_periods // 2
    all_periods = list(range(n_periods))

    logits = []
    best_is_candidates = []
    for is_periods in combinations(all_periods, half):
        oos_periods = [p for p in all_periods if p not in is_periods]

        is_perf = perf.loc[list(is_periods)].mean()
        oos_perf = perf.loc[oos_periods].mean()

        best_candidate_is = is_perf.idxmax()
        best_is_candidates.append(best_candidate_is)
        oos_rank = oos_perf.rank(pct=True)[best_candidate_is]

        oos_rank_clipped = np.clip(oos_rank, 0.01, 0.99)
        logit = np.log(oos_rank_clipped / (1 - oos_rank_clipped))
        logits.append(logit)

    logits = np.array(logits)
    pbo = (logits <= 0).mean()

    return {
        "n_combinations": len(logits),
        "pbo": pbo,
        "logit_mean": logits.mean(),
        "logit_median": np.median(logits),
        "best_is_candidates": pd.Series(best_is_candidates).value_counts(),
    }


if __name__ == "__main__":
    df = load_dataset()
    print(f"064350 BASE 전용 데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})\n")

    perf = build_performance_matrix(df)
    print("\n=== 구간별 threshold 후보 성과 (064350 BASE 전용, 복리수익률) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (threshold 축, {len(CANDIDATE_THRESHOLDS)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f} (양수면 IS 선택이 OOS에서도 유효했다는 뜻)")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 threshold ===")
    print(result["best_is_candidates"].to_string())
    print("\n(백테스트에서 채택한 threshold=0.60이 여기서 자주 1등이면 선택이 근거")
    print(" 있었다는 뜻. 안 보이면 0.60 선택 자체가 사후적으로 골라낸 것일 수 있음.)")

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- IS 최고 후보가 OOS에서 신뢰 어려움")