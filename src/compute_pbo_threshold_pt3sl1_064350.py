"""
PBO(Probability of Backtest Overfitting) -- 064350 BASE 전용, pt_sl=(3,1) 고정,
threshold 축.

compute_pbo_base_only_064350.py(pt_sl=(2,1) 기준 threshold PBO, 19.4%)와 동일
방법론이되, pt_sl 축 PBO(compute_pbo_pt_sl_064350.py)에서 (3,1)이 IS 최고빈도
63%로 압도적이었고 실제 5-seed 백테스트(backtest_pt1sl1_064350.py)에서도
threshold=0.50 하나가 5/5로 통과한 상태 -- 근데 "5/5 통과하는 유일한 threshold"
라는 것 자체가 우연히 골라낸 것일 수 있으므로, (2,1) 때와 동일하게 threshold
자체의 탐색 위험을 여기서 확인.

사용법 (레포 루트에서):
    python src/compute_pbo_threshold_pt3sl1_064350.py

전제:
    compute_pbo_pt_sl_064350.py 또는 backtest_pt1sl1_064350.py로
    data/064350_features_triple_barrier_pt3sl1_nd20_hl_base.csv가 이미
    생성돼 있어야 함.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKER_KRX = "064350"
PT_SL_LABEL = "pt3sl1_nd20_hl"
CANDIDATE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]  # 0.50 주변으로 좀 더 촘촘하게
SEED = 42
N_SUBPERIODS = 10
NUM_DAYS = 20
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{PT_SL_LABEL}_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
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


def generate_trades(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=SEED,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": entry_date + pd.Timedelta(days=holding),
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1

    return pd.DataFrame(trades)


def build_performance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    start, end = df.index.min(), df.index.max()
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)

    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=CANDIDATE_THRESHOLDS, dtype=float)
    for threshold in CANDIDATE_THRESHOLDS:
        print(f"threshold={threshold} 거래 생성 중...")
        trades = generate_trades(df, threshold)

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            if trades.empty:
                perf.loc[s, threshold] = 0.0
                continue
            in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
            perf.loc[s, threshold] = (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0

    return perf


def cscv_pbo(perf: pd.DataFrame) -> dict:
    n_periods = perf.shape[0]
    half = n_periods // 2
    all_periods = list(range(n_periods))

    logits, best_is_candidates = [], []
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
    return {
        "n_combinations": len(logits),
        "pbo": (logits <= 0).mean(),
        "logit_mean": logits.mean(),
        "logit_median": np.median(logits),
        "best_is_candidates": pd.Series(best_is_candidates).value_counts(),
    }


if __name__ == "__main__":
    df = load_dataset()
    print(f"064350 pt_sl=(3,1) BASE 전용: {df.shape[0]}행\n")

    perf = build_performance_matrix(df)
    print("\n=== 구간별 threshold 후보 성과 (pt_sl=(3,1) 고정) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (threshold 축, pt_sl=(3,1) 고정, {len(CANDIDATE_THRESHOLDS)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 threshold ===")
    print(result["best_is_candidates"].to_string())
    print("\n(백테스트에서 채택한 threshold=0.50이 여기서 자주 1등이면 선택이 근거 있음)")

    print("\n판정 기준: PBO<20% 낮음 / 20~50% 중간 / >=50% 높음")