"""
BASE(13개) XGBoost가 horizon 1/3/5/10/20 중 어디서든 동일가중(base rate)을
이기는지 확인.

배경: quant_seq_model에서 lag feature / GRU / TCN(3종목, 50종목 풀링) 전부
horizon=10, cost_threshold=0.005 기준으로 vs_base_rate 5/5 실패. 모델을 계속
바꿔도 안 통과되니, "이 horizon/threshold 조합 자체가 이 피처들로는 근본적으로
약한 신호"인지를 가장 싼 모델(XGBoost)로 먼저 확인. 여기서 어느 horizon에서도
안 통과되면 모델 문제가 아니라 라벨/피처 조합 자체의 문제로 봐야 함.

⚠️ 사전 등록: horizon만 바꾸고 cost_threshold=0.005, FEATURE_COLS_BASE(13개),
walk-forward 파라미터(train=300/test=60/step=60)는 전부 고정. embargo=horizon으로
각 horizon에 맞게만 조정. 결과 보고 나서 이 값들 바꾸지 않음.

사용법 (레포 루트에서):
    python src/train_xgboost_horizon_sweep.py
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features, add_label,
)

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

HORIZONS = [1, 3, 5, 10, 20]
COST_THRESHOLD = 0.005
SEEDS = (42, 1, 7, 123, 2024)
TRAIN_SIZE = 300
TEST_SIZE = 60
STEP = 60

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. Walk-Forward 분할 (기존 ablation 스크립트들과 동일)
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


def run_walk_forward(df: pd.DataFrame, embargo: int, random_state: int) -> pd.DataFrame:
    X = df[FEATURE_COLS_BASE]
    y = df["label"]

    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, embargo)
    rows = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=random_state)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        base_rate = y_test.mean()

        rows.append({
            "auc": roc_auc_score(y_test, proba),
            "vs_base_rate": accuracy_score(y_test, pred) - max(base_rate, 1 - base_rate),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"

    print(f"{TICKER_KRX} 원본 데이터 + BASE 피처 1회 생성 (horizon과 무관한 부분)...")
    df_raw, bench_raw = load_data(TICKER, "^KS11", "2015-01-01", "2026-07-18")
    df_raw = add_momentum_features(df_raw)
    df_raw = add_volatility_features(df_raw)
    df_raw = add_volume_features(df_raw)
    df_raw = add_relative_strength_features(df_raw, bench_raw)
    print(f"완료: {df_raw.shape[0]}행\n")

    summary_rows = []
    for horizon in HORIZONS:
        df = add_label(df_raw.copy(), horizon=horizon, cost_threshold=COST_THRESHOLD)
        feature_cols = ["Close", "Volume", *FEATURE_COLS_BASE, "future_return", "label"]
        df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

        print(f"=== horizon={horizon} ({df.shape[0]}행, 라벨 1비율={df['label'].mean():.3f}) ===")
        seed_results = []
        for seed in SEEDS:
            fold_df = run_walk_forward(df, embargo=horizon, random_state=seed)
            seed_results.append({
                "horizon": horizon,
                "seed": seed,
                "mean_auc": fold_df["auc"].mean(),
                "mean_vs_base_rate": fold_df["vs_base_rate"].mean(),
                "win_folds": int((fold_df["vs_base_rate"] > 0).sum()),
                "n_folds": len(fold_df),
            })
        seed_df = pd.DataFrame(seed_results)
        print(seed_df.round(4).to_string(index=False))
        wins = int((seed_df["mean_vs_base_rate"] > 0).sum())
        print(f"horizon={horizon}: vs_base_rate 5/5 양수 {wins}/5 {'(통과)' if wins == 5 else ''}\n")

        summary_rows.append({
            "horizon": horizon,
            "mean_vs_base_rate": seed_df["mean_vs_base_rate"].mean(),
            "mean_auc": seed_df["mean_auc"].mean(),
            "seeds_passed": wins,
        })

    print("=" * 60)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.round(4).to_string(index=False))

    if (summary_df["seeds_passed"] == 5).any():
        best = summary_df.loc[summary_df["mean_vs_base_rate"].idxmax()]
        print(f"\nhorizon={int(best['horizon'])}에서 통과 -- 이 horizon으로 시퀀스 모델 재시도 고려")
    else:
        print("\n모든 horizon 5/5 실패 -- 라벨/threshold 문제가 아니라 BASE 피처 자체의")
        print("신호 부족일 가능성이 높음. quant_xgboost priority #1(원래 PER 버그 이후")
        print("BASE-only 재검증)로 복귀하는 걸 고려할 것.")