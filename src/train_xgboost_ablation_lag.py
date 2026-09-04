"""
BASE(13개) vs LAG_ONLY(15개, lag만) vs COMBINED(BASE+LAG) 3-way ablation.
(train_xgboost_ablation_fx.py / _rate_spread.py와 동일한 패턴)

목적: "과거 며칠치 피처 값을 오늘 row에 나란히 붙여주면" 신호가 생기는지 확인.
     진짜 시퀀스 모델(LSTM/TCN/Transformer) 들어가기 전, 제일 싼 사전 점검.

전제:
    feature_engineering_lag.py의 build_multi_horizon_datasets_lag()를 먼저 실행해서
    {ticker_krx}_features_with_lag_h{horizon}.csv가 있어야 함.

사용법 (레포 루트에서):
    python src/train_xgboost_ablation_lag.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from feature_engineering_lag import lag_feature_names

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_LAG_ONLY = lag_feature_names()  # 15개 (5개 피처 x lag 5/10/20일)
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_LAG_ONLY

DEFAULT_HORIZON = 10
SEEDS = (42, 1, 7, 123, 2024)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(ticker_krx: str = "064350", horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_with_lag_h{horizon}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED + ["label"])
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (기존 ablation 스크립트들과 동일, position 기준)
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
# 3. fold별 학습 + 평가 (feature_cols를 인자로 받아서 BASE/LAG_ONLY/COMBINED 재사용)
# ------------------------------------------------------------------
def run_walk_forward(df: pd.DataFrame, feature_cols: list, embargo: int, train_size=300, test_size=60,
                      step=60, threshold=0.5, random_state=42):
    X = df[feature_cols]
    y = df["label"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError(
            f"데이터가 부족해서 walk-forward split을 만들 수 없어요. "
            f"(n_rows={len(df)}, train_size={train_size}, test_size={test_size}, embargo={embargo}) "
            f"train_size/test_size를 줄이세요."
        )

    rows = []
    importances = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=random_state)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        base_rate = y_test.mean()
        rows.append({
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
            "vs_base_rate": accuracy_score(y_test, pred) - max(base_rate, 1 - base_rate),
        })
        importances.append(pd.Series(model.feature_importances_, index=feature_cols))

    fold_df = pd.DataFrame(rows)
    importance_df = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return fold_df, importance_df


# ------------------------------------------------------------------
# 4. BASE vs LAG_ONLY vs COMBINED 비교
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(
        df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    lag_fold_df, lag_importance = run_walk_forward(
        df, FEATURE_COLS_LAG_ONLY, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(
        df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "LAG_ONLY": lag_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["LAG_ONLY-BASE"] = summary["LAG_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    return {
        "summary": summary,
        "base_importance": base_importance,
        "lag_importance": lag_importance,
        "combined_importance": combined_importance,
        "base_win_folds": (base_fold_df["vs_base_rate"] > 0).sum(),
        "combined_win_folds": (combined_fold_df["vs_base_rate"] > 0).sum(),
        "n_folds": len(base_fold_df),
    }


# ------------------------------------------------------------------
# 5. 멀티 시드 검증
# ------------------------------------------------------------------
def run_multi_seed(df: pd.DataFrame, horizon: int, seeds=SEEDS, **wfo_kwargs):
    rows = []
    for seed in seeds:
        result = compare_feature_sets(df, horizon=horizon, random_state=seed, **wfo_kwargs)
        rows.append({
            "seed": seed,
            "BASE_auc": result["summary"].loc["auc", "BASE"],
            "COMBINED_auc": result["summary"].loc["auc", "COMBINED"],
            "COMBINED_auc_diff": result["summary"].loc["auc", "COMBINED-BASE"],
            "BASE_vs_base_rate": result["summary"].loc["vs_base_rate", "BASE"],
            "COMBINED_vs_base_rate": result["summary"].loc["vs_base_rate", "COMBINED"],
            "COMBINED_vsbase_diff": result["summary"].loc["vs_base_rate", "COMBINED-BASE"],
            "LAG_ONLY_auc": result["summary"].loc["auc", "LAG_ONLY"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    TICKER_KRX = "064350"
    HORIZON = 10

    df = load_dataset(ticker_krx=TICKER_KRX, horizon=HORIZON)
    print(f"{TICKER_KRX} lag 데이터셋 로드 완료: {df.shape[0]}행, "
          f"{df.index.min().date()} ~ {df.index.max().date()}\n")
    print(f"BASE {len(FEATURE_COLS_BASE)}개 / LAG_ONLY {len(FEATURE_COLS_LAG_ONLY)}개 / "
          f"COMBINED {len(FEATURE_COLS_COMBINED)}개 피처\n")

    seed_df = run_multi_seed(df, horizon=HORIZON)
    print(seed_df.round(4).to_string(index=False))

    combined_diffs = seed_df["COMBINED_vsbase_diff"].values
    print(f"\nCOMBINED vs BASE (vs_base_rate) 5/5 양수: "
          f"{int((combined_diffs > 0).sum())}/5 "
          f"{'(통과)' if (combined_diffs > 0).all() else '(일관성 실패)'}")
    print("여기서 5/5 통과 못하면 lag feature 라인은 접고 GRU/TCN으로 넘어가는 게 맞음")
    print("(과거 값을 '나란히 붙이는' 방식 자체의 한계일 수 있음 -- 진짜 시퀀스로")
    print(" 모델링해야만 잡히는 신호일 가능성).")