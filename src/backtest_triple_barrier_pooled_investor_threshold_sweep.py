"""
BASE+INVESTOR 3종목 풀링 백테스트 -- threshold sweep.

배경: train_xgboost_triple_barrier_investor_ablation.py 결과, AUC diff가 3종목
평균 -0.01~+0.01 범위로 노이즈 수준이었음 (수급 feature가 판별력 자체를 깎아먹은
게 아님). 그런데 backtest_triple_barrier_pooled_investor.py의 5-seed 백테스트는
0/5로 Buy&Hold를 크게 못 이겼음 (블렌딩 net 79~218% vs BH 547%).

AUC는 거의 안 변했는데 백테스트가 크게 나빠졌다는 건 판별력 문제가 아니라
threshold=0.65(BASE+VALUATION 때 고른 값)가 수급 feature 조합의 확률분포에는
안 맞았을 가능성을 가리킴 -- 이 스크립트로 그 가설을 직접 확인.

사용법 (레포 루트에서):
    python src/backtest_triple_barrier_pooled_investor_threshold_sweep.py

전제:
    feature_engineering_triple_barrier_investor.py로 3종목 다
    {ticker}_features_triple_barrier_pt2sl1_nd20_hl_investor.csv가 생성돼 있어야 함.
"""

from pathlib import Path

import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR

ROUND_TRIP_COST = 0.002
NUM_DAYS = 20
CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKERS = ["064350", "052690", "118990"]
CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
SEEDS = [42, 1, 7, 123, 2024]
WEIGHT = 1 / len(TICKERS)


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_investor.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


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


def generate_trades(df: pd.DataFrame, threshold: float, train_size: int = 300,
                     test_size: int = 60, step: int = 60, embargo: int = NUM_DAYS,
                     random_state: int = 42) -> pd.DataFrame:
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
                    trades.append({"net_return": net_return})
                i += max(holding, 1)
            else:
                i += 1
    return pd.DataFrame(trades)


def get_fixed_bh_window(df: pd.DataFrame, train_size: int = 300, test_size: int = 60,
                         step: int = 60, embargo: int = NUM_DAYS) -> tuple:
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return df.index[first_test_idx], df.index[min(last_test_idx, len(df) - 1)]


if __name__ == "__main__":
    dfs = {t: load_dataset(t) for t in TICKERS}
    bh_windows = {t: get_fixed_bh_window(dfs[t]) for t in TICKERS}
    bh_returns = {
        t: dfs[t].loc[bh_windows[t][1], "Close"] / dfs[t].loc[bh_windows[t][0], "Close"] - 1
        for t in TICKERS
    }
    blended_bh = sum(WEIGHT * bh_returns[t] for t in TICKERS)
    print(f"블렌딩 Buy & Hold (고정): {blended_bh:.1%}\n")

    print("=" * 70)
    print("=== threshold별 5-seed 평균 블렌딩 net ===")
    print("=" * 70)

    summary_rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        seed_nets = []
        seed_n_trades = []
        for seed in SEEDS:
            per_ticker_net = {}
            n_trades_total = 0
            for t in TICKERS:
                trades = generate_trades(dfs[t], threshold=threshold, random_state=seed)
                n_trades_total += len(trades)
                per_ticker_net[t] = (1 + trades["net_return"]).prod() - 1 if not trades.empty else 0.0
            blended_net = sum(WEIGHT * per_ticker_net[t] for t in TICKERS)
            seed_nets.append(blended_net)
            seed_n_trades.append(n_trades_total)

        avg_net = sum(seed_nets) / len(seed_nets)
        win_count = sum(1 for n in seed_nets if n > blended_bh)
        avg_n_trades = sum(seed_n_trades) / len(seed_n_trades)

        summary_rows.append({
            "threshold": threshold, "avg_n_trades": avg_n_trades,
            "avg_blended_net": avg_net, "win_count_vs_bh": f"{win_count}/5",
        })
        print(f"threshold={threshold}: 평균 거래수={avg_n_trades:.0f}, "
              f"5-seed 평균 net={avg_net:.1%}, BH 이긴 시드={win_count}/5")

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("=== 요약 테이블 ===")
    print("=" * 70)
    print(summary_df.round(4).to_string(index=False))

    print("\n[해석 가이드]")
    print("- 0.65 근처에서 특별히 나쁘고 다른 threshold(예: 0.55, 0.70)에서 나아지면:")
    print("  '전략 자체는 살아있는데 threshold 하나를 잘못 골랐다'는 뜻 -- 다만 이 경우")
    print("  최적 threshold를 여기서 새로 고르는 것 자체가 또 다른 탐색이라, PBO 검증을")
    print("  반드시 threshold 축까지 다시 거쳐야 함 (사후 최적화 위험 반복 방지).")
    print("- 모든 threshold에서 고르게 BH를 못 이기면: threshold 문제가 아니라 이 feature")
    print("  조합 자체가 이 3종목/전략 구조와 안 맞는다는 뜻 -- 여기서 폐기가 맞음.")