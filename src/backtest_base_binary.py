"""
[재판정] BASE XGBoost(horizon=10)를 vs_base_rate 대신 실제 백테스트로 판정.

diagnose_base_rate_drift_binary.py에서 확인됨: drift 절대값 평균 0.121,
test_base_rate std 0.140, |drift|>0.15인 fold 15/41 (범위 [0.233, 0.867]) --
triple-barrier 재검증 때 vs_base_rate 게이트를 무효화시켰던 것과 같은 수준의
fold 간 라벨 비율 드리프트. backtest_base_only_triple_barrier.py가 이미 확립한
이 프로젝트의 최종 판정 기준("거래비용 반영 후 Buy & Hold를 이기는가")을 그대로
따라서 BASE XGBoost부터 재판정함. 여기서 통과/실패가 나오면 lag/GRU/TCN도
같은 기준으로 순서대로 재판정할 것.

방법: proba >= threshold인 날 진입, horizon(10)일 고정 보유 후 청산 (겹치는
진입은 보유기간만큼 건너뛰어서 중복 방지), 거래비용 반영 net_return을 복리로
누적, 거래가 발생한 기간 동안의 Buy & Hold 수익률과 비교.
(triple-barrier와 달리 우리 라벨은 holding_rows가 가변이 아니라 horizon으로
고정이라 보유기간 로직이 더 단순함.)

⚠️ 사전 등록: threshold [0.50, 0.55, 0.60, 0.65] 스윕, ROUND_TRIP_COST=0.002
(backtest_base_only_triple_barrier.py와 동일 값), 5-seed(42/1/7/123/2024).
horizon/cost_threshold/walk-forward 파라미터는 지금까지 쓰던 것과 동일
(horizon=10, cost_threshold=0.005, train=300/test=60/step=60/embargo=10).

사용법 (레포 루트에서):
    python src/backtest_base_binary.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering import build_feature_dataset

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

TICKER = "064350.KS"
TICKER_KRX = "064350"
HORIZON = 10
COST_THRESHOLD = 0.005
SEEDS = [42, 1, 7, 123, 2024]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, HORIZON
ROUND_TRIP_COST = 0.002

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. Walk-Forward 분할 (기존과 동일)
# ------------------------------------------------------------------
def walk_forward_splits(n_rows: int, train_size: int, test_size: int, step: int, embargo: int):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 2. 실제 거래 생성 (accuracy가 아니라 진입/청산/수익률 단위로)
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, threshold: float, random_state: int) -> pd.DataFrame:
    X = df[FEATURE_COLS_BASE]
    y = df["label"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=random_state)
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["future_return"].iloc[row_idx]

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "proba": proba[i],
                        "gross_return": gross_return,
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += HORIZON  # 보유기간(고정 horizon)만큼 건너뛰어서 중복 진입 방지
            else:
                i += 1

    return pd.DataFrame(trades)


# ------------------------------------------------------------------
# 3. Buy & Hold 비교
# ------------------------------------------------------------------
def get_full_test_period(df: pd.DataFrame):
    """41개 fold 전체가 커버하는 out-of-sample 구간의 시작/끝 날짜.
    이 구간 하나로 고정해서 모든 threshold·seed가 같은 B&H 기준선과 비교하게 함
    (seed마다 거래 시작/종료일이 달라져서 B&H 자체가 흔들리는 문제 방지)."""
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return df.index[first_test_idx], df.index[last_test_idx]


def buy_and_hold_return(df: pd.DataFrame, start_date, end_date) -> float:
    period = df.loc[start_date:end_date]
    if len(period) < 2:
        return np.nan
    return period["Close"].iloc[-1] / period["Close"].iloc[0] - 1


def summarize_trades(trades: pd.DataFrame, bh_return: float) -> dict:
    if trades.empty:
        return {
            "n_trades": 0, "total_net_return": np.nan, "win_rate": np.nan,
            "bh_return": bh_return, "excess_vs_bh": np.nan,
        }

    total_net_return = (1 + trades["net_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()

    return {
        "n_trades": len(trades),
        "total_net_return": total_net_return,
        "win_rate": win_rate,
        "bh_return": bh_return,
        "excess_vs_bh": total_net_return - bh_return,
    }


if __name__ == "__main__":
    df = build_feature_dataset(ticker=TICKER, horizon=HORIZON, cost_threshold=COST_THRESHOLD)
    df = df.sort_index()
    print(f"{TICKER_KRX}: {df.shape[0]}행, {df.index.min().date()} ~ {df.index.max().date()}\n")

    full_start, full_end = get_full_test_period(df)
    fixed_bh_return = buy_and_hold_return(df, full_start, full_end)
    print(f"고정 B&H 비교 구간: {full_start.date()} ~ {full_end.date()} (수익률 {fixed_bh_return:+.2%})\n")

    rows = []
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            trades = generate_trades(df, threshold=threshold, random_state=seed)
            summary = summarize_trades(trades, fixed_bh_return)
            summary["threshold"] = threshold
            summary["seed"] = seed
            rows.append(summary)

            if summary["n_trades"]:
                print(f"threshold={threshold}, seed={seed}: {summary['n_trades']}건, "
                      f"누적 순수익률={summary['total_net_return']:+.2%}, "
                      f"승률={summary['win_rate']:.1%}, "
                      f"B&H={summary['bh_return']:+.2%}, "
                      f"초과수익={summary['excess_vs_bh']:+.2%}")
            else:
                print(f"threshold={threshold}, seed={seed}: 거래 없음")

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("=== threshold별 5-seed 요약 ===")
    print("=" * 70)
    grouped = result_df.groupby("threshold").agg(
        mean_n_trades=("n_trades", "mean"),
        mean_net_return=("total_net_return", "mean"),
        mean_excess_vs_bh=("excess_vs_bh", "mean"),
        seeds_beat_bh=("excess_vs_bh", lambda s: (s > 0).sum()),
    )
    print(grouped.round(4).to_string())

    print("\n5-seed 전부(5/5) excess_vs_bh > 0인 threshold가 있으면 -- vs_base_rate")
    print("아티팩트를 걷어내니 실전 가능성이 있다는 뜻. 하나도 없으면 -- 아티팩트를")
    print("걷어내고 봐도 여전히 실패 -- [실패] 판정 유지하고 priority #1(BASE-only")
    print("재검증) 완료로 처리, lag/GRU/TCN 재판정도 같은 결론으로 수렴할 가능성 높음.")