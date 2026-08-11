"""
PBO(Probability of Backtest Overfitting) 확장판 -- threshold뿐 아니라
"어떤 종목 조합을 포트폴리오로 쓸지"까지 후보에 포함해서 CSCV 검증.

배경: compute_pbo.py(1차)는 threshold 축 하나만 검증했음. 이번 세션에서 실제로
고민했던 또 다른 큰 선택은 "현대로템+한전기술+모트렉스 3종목을 다 쓸지, 일부만
쓸지"였음 -- 이것도 일종의 파라미터 선택이라 PBO 대상에 포함해야 함.
(pt_sl, num_days, 체결방식까지 포함한 완전한 PBO는 데이터 재생성이 필요해서 범위 밖)

후보 = threshold(4개) x 종목 부분집합(7개, 단일3+쌍3+전체1) = 28개 후보.

사용법 (레포 루트에서):
    python src/compute_pbo_v2.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

ROUND_TRIP_COST = 0.002
NUM_DAYS = 20
CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKERS = ["064350", "052690", "118990"]
CANDIDATE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
SEED = 42
N_SUBPERIODS = 10


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


def walk_forward_splits(n_rows, train_size, test_size, step, embargo):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


def generate_trades(df, threshold, train_size=300, test_size=60, step=60,
                     embargo=NUM_DAYS, random_state=SEED):
    X = df[FEATURE_COLS_COMBINED]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx])
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1
    return pd.DataFrame(trades)


def get_ticker_subsets(tickers: list) -> list:
    """단일 종목 3개 + 쌍 3개 + 전체 1개 = 7개 부분집합."""
    subsets = []
    for r in range(1, len(tickers) + 1):
        subsets.extend(combinations(tickers, r))
    return subsets


def build_performance_matrix():
    """
    행=구간, 열=(threshold, 종목조합) 후보. 값=그 구간에서 그 후보의 복리수익률.
    """
    dfs = {t: load_dataset(t) for t in TICKERS}
    start = max(df.index.min() for df in dfs.values())
    end = min(df.index.max() for df in dfs.values())
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)
    print(f"전체 비교 구간: {start.date()} ~ {end.date()}, {N_SUBPERIODS}개 구간으로 분할")

    subsets = get_ticker_subsets(TICKERS)
    print(f"종목 조합 후보: {len(subsets)}개, threshold 후보: {len(CANDIDATE_THRESHOLDS)}개 "
          f"-> 총 {len(subsets) * len(CANDIDATE_THRESHOLDS)}개 조합")

    ticker_threshold_trades = {}
    for t in TICKERS:
        for threshold in CANDIDATE_THRESHOLDS:
            print(f"  {t}, threshold={threshold} 거래 생성 중...")
            ticker_threshold_trades[(t, threshold)] = generate_trades(dfs[t], threshold=threshold)

    candidates = [(threshold, subset) for threshold in CANDIDATE_THRESHOLDS for subset in subsets]
    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=range(len(candidates)), dtype=float)
    candidate_labels = []

    for c_idx, (threshold, subset) in enumerate(candidates):
        candidate_labels.append(f"th{threshold}_{'+'.join(subset)}")
        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            subset_period_returns = []
            for t in subset:
                trades = ticker_threshold_trades[(t, threshold)]
                if trades.empty:
                    subset_period_returns.append(0.0)
                    continue
                in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
                if in_period.empty:
                    subset_period_returns.append(0.0)
                else:
                    subset_period_returns.append((1 + in_period["net_return"]).prod() - 1)
            perf.loc[s, c_idx] = np.mean(subset_period_returns)

    perf.columns = candidate_labels
    return perf, candidate_labels


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
    perf, labels = build_performance_matrix()

    print(f"\n=== 구간별 전체 후보 성과 요약 (전체 {len(labels)}개 후보 중 평균 상위 5개만 표시) ===")
    avg_perf = perf.mean().sort_values(ascending=False)
    print(avg_perf.head(5).round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (threshold x 종목조합, {len(labels)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 후보 상위 5개 ===")
    print(result["best_is_candidates"].head(5).to_string())
    print("\n(우리가 최종 채택한 'th0.65_064350+052690+118990'이 여기 상위권에 자주")
    print(" 등장하면, 우리 선택이 근거 있었다는 뜻. 안 보이면 다른 조합이 더 나았을")
    print(" 수 있다는 뜻이니 재검토 필요.)")

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- 신뢰 어려움")