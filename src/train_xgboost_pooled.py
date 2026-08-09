"""
여러 종목을 풀링한 데이터로 XGBoost 방향성 분류 학습
핵심: walk-forward 분할을 '행 개수'가 아니라 '고유 날짜' 기준으로 함
      (같은 날짜의 여러 종목 데이터가 train/test로 쪼개지면 날짜 기준 누수가 생기므로)

사용법:
    python train_xgboost_pooled.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


FEATURE_COLS = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
    "ticker",  # 범주형 feature로 추가 -- 종목 자체도 정보가 될 수 있음
]

HORIZON = 10


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_pooled_dataset(path: str = None) -> pd.DataFrame:
    path = path or DATA_DIR / "pooled_features.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in FEATURE_COLS if c != "ticker"] + ["label"])
    df["ticker"] = df["ticker"].astype("category")
    df = df.sort_index()
    return df


# ------------------------------------------------------------------
# 2. 날짜 기준 Walk-Forward 분할
#    (행 인덱스가 아니라 '고유 날짜' 리스트를 슬라이딩 -> 같은 날짜의 여러 종목이 항상 같은 쪽에 속함)
# ------------------------------------------------------------------
def walk_forward_splits_by_date(df: pd.DataFrame, train_days: int, test_days: int, step_days: int, embargo_days: int):
    unique_dates = df.index.unique().sort_values()
    n = len(unique_dates)

    splits = []
    start = 0
    while start + train_days + embargo_days + test_days <= n:
        train_dates = unique_dates[start: start + train_days]
        test_start = start + train_days + embargo_days
        test_dates = unique_dates[test_start: test_start + test_days]
        splits.append((train_dates, test_dates))
        start += step_days
    return splits


# ------------------------------------------------------------------
# 3. fold별 학습 + 평가
# ------------------------------------------------------------------
def run_walk_forward(df: pd.DataFrame, train_days=300, test_days=60, step_days=60,
                      embargo_days=HORIZON, threshold=0.5):
    splits = walk_forward_splits_by_date(df, train_days, test_days, step_days, embargo_days)
    if not splits:
        raise ValueError("데이터가 부족해서 split을 만들 수 없어요.")

    fold_rows = []
    importances = []

    for fold, (train_dates, test_dates) in enumerate(splits):
        train = df.loc[df.index.isin(train_dates)]
        test = df.loc[df.index.isin(test_dates)]

        X_train, y_train = train[FEATURE_COLS], train["label"]
        X_test, y_test = test[FEATURE_COLS], test["label"]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            enable_categorical=True,   # ticker를 범주형으로 직접 처리
            tree_method="hist",        # 범주형 지원에 필요
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        base_rate = y_test.mean()
        acc = accuracy_score(y_test, pred)

        fold_rows.append({
            "fold": fold,
            "test_start": test_dates[0].date(),
            "test_end": test_dates[-1].date(),
            "n_rows": len(test),
            "base_rate": base_rate,
            "accuracy": acc,
            "vs_base_rate": acc - max(base_rate, 1 - base_rate),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        })
        importances.append(model.feature_importances_)

    fold_df = pd.DataFrame(fold_rows)
    importance = pd.Series(np.mean(importances, axis=0), index=FEATURE_COLS).sort_values(ascending=False)
    return fold_df, importance


if __name__ == "__main__":
    df = load_pooled_dataset()
    print(f"전체 데이터: {df.shape[0]}행, 종목: {df['ticker'].unique().tolist()}")
    print(f"전체 라벨 분포:\n{df['label'].value_counts(normalize=True)}\n")

    fold_df, importance = run_walk_forward(df, threshold=0.5)

    print("=== Fold별 결과 (threshold=0.5) ===")
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== 전체 평균 ===")
    print(fold_df[["base_rate", "accuracy", "vs_base_rate", "precision", "recall", "auc"]].mean())

    n_positive_folds = (fold_df["vs_base_rate"] > 0).sum()
    print(f"\n베이스라인 이긴 fold: {n_positive_folds} / {len(fold_df)}")

    print("\n=== Feature Importance (상위순) ===")
    print(importance)