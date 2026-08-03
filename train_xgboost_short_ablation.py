"""
공매도 feature 추가 전/후 성능을 비교하는 ablation 스크립트
(train_xgboost_ablation.py와 동일한 구조 -- BASE/SHORT_ONLY/COMBINED + 멀티 시드 검증)

사용법:
    python train_xgboost_short_ablation.py

전제:
    feature_engineering_short.py를 먼저 실행해서
    {ticker}_features_with_short_h{horizon}.csv가 만들어져 있어야 함.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

FEATURE_COLS_SHORT = [
    "short_qty_3d", "short_qty_5d", "short_weight_5d_avg", "short_covering_signal",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_SHORT

HORIZON = 5  # feature_engineering_short.py의 horizon과 반드시 동일 (embargo에 사용)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(path: str = "005930_features_with_short_h5.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED + ["label"])
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할
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
# 3. fold별 학습 + 평가
# ------------------------------------------------------------------
def run_walk_forward(df: pd.DataFrame, feature_cols: list, train_size=100, test_size=20,
                      step=20, embargo=HORIZON, threshold=0.5, random_state=42):
    X = df[feature_cols]
    y = df["label"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    fold_rows = []
    importances = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        base_rate = y_test.mean()
        acc = accuracy_score(y_test, pred)

        fold_rows.append({
            "fold": fold,
            "test_start": df.index[list(test_idx)[0]].date(),
            "test_end": df.index[list(test_idx)[-1]].date(),
            "base_rate": base_rate,
            "accuracy": acc,
            "vs_base_rate": acc - max(base_rate, 1 - base_rate),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        })
        importances.append(model.feature_importances_)

    fold_df = pd.DataFrame(fold_rows)
    importance = pd.Series(np.mean(importances, axis=0), index=feature_cols).sort_values(ascending=False)
    return fold_df, importance


# ------------------------------------------------------------------
# 4. BASE vs SHORT_ONLY vs COMBINED 비교
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, random_state=random_state, **wfo_kwargs)
    short_fold_df, short_importance = run_walk_forward(df, FEATURE_COLS_SHORT, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "SHORT_ONLY": short_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["SHORT_ONLY-BASE"] = summary["SHORT_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    return {
        "base_fold_df": base_fold_df,
        "short_fold_df": short_fold_df,
        "combined_fold_df": combined_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "short_importance": short_importance,
        "combined_importance": combined_importance,
        "base_win_folds": (base_fold_df["vs_base_rate"] > 0).sum(),
        "short_win_folds": (short_fold_df["vs_base_rate"] > 0).sum(),
        "combined_win_folds": (combined_fold_df["vs_base_rate"] > 0).sum(),
        "n_folds": len(base_fold_df),
    }


# ------------------------------------------------------------------
# 5. 멀티 시드 검증
# ------------------------------------------------------------------
def run_multi_seed(df: pd.DataFrame, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
    rows = []
    for seed in seeds:
        print(f"--- seed={seed} ---")
        result = compare_feature_sets(df, random_state=seed, **wfo_kwargs)
        rows.append({
            "seed": seed,
            "BASE_auc": result["summary"].loc["auc", "BASE"],
            "COMBINED_auc": result["summary"].loc["auc", "COMBINED"],
            "COMBINED_auc_diff": result["summary"].loc["auc", "COMBINED-BASE"],
            "BASE_vs_base_rate": result["summary"].loc["vs_base_rate", "BASE"],
            "COMBINED_vs_base_rate": result["summary"].loc["vs_base_rate", "COMBINED"],
            "COMBINED_vsbase_diff": result["summary"].loc["vs_base_rate", "COMBINED-BASE"],
            "SHORT_ONLY_auc": result["summary"].loc["auc", "SHORT_ONLY"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()
    print(f"전체 데이터: {df.shape[0]}행\n")

    result = compare_feature_sets(df)

    print("=" * 60)
    print("=== 전체 평균 비교 (fold 평균, seed=42) ===")
    print("=" * 60)
    print(result["summary"].round(4))

    print(f"\n베이스라인 이긴 fold 수 (fold 총 {result['n_folds']}개)")
    print(f"  BASE:       {result['base_win_folds']} / {result['n_folds']}")
    print(f"  SHORT_ONLY: {result['short_win_folds']} / {result['n_folds']}")
    print(f"  COMBINED:   {result['combined_win_folds']} / {result['n_folds']}")

    print("\n" + "=" * 60)
    print("=== SHORT_ONLY Feature Importance ===")
    print("=" * 60)
    print(result["short_importance"])

    # ------------------------------------------------------------------
    # 멀티 시드 검증
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 60)
    print("# 멀티 시드 검증 (5개 시드 반복)")
    print("#" * 60)

    seed_df = run_multi_seed(df)

    print("\n" + "=" * 60)
    print("=== 시드별 결과 ===")
    print("=" * 60)
    print(seed_df.round(4).to_string(index=False))

    n_seeds = len(seed_df)
    combined_beats_base_count = (seed_df["COMBINED_auc_diff"] > 0).sum()
    combined_vsbase_beats_count = (seed_df["COMBINED_vsbase_diff"] > 0).sum()

    print("\n" + "=" * 60)
    print("=== 최종 판정 (5개 시드 종합) ===")
    print("=" * 60)
    print(f"COMBINED가 BASE보다 AUC 높았던 시드:        {combined_beats_base_count} / {n_seeds}")
    print(f"COMBINED가 BASE보다 vs_base_rate 높았던 시드: {combined_vsbase_beats_count} / {n_seeds}")
    print(f"\nAUC 차이(COMBINED-BASE) 평균: {seed_df['COMBINED_auc_diff'].mean():+.4f} (표준편차 {seed_df['COMBINED_auc_diff'].std():.4f})")
    print(f"vs_base_rate 차이 평균:       {seed_df['COMBINED_vsbase_diff'].mean():+.4f} (표준편차 {seed_df['COMBINED_vsbase_diff'].std():.4f})")

    print()
    if combined_beats_base_count >= n_seeds - 1 and combined_vsbase_beats_count >= n_seeds - 1:
        print("→ 5개 시드 중 대부분에서 COMBINED가 BASE를 이김. 재현 가능한 신호로 볼 여지가 있음.")
    elif combined_beats_base_count <= 1 and combined_vsbase_beats_count <= 1:
        print("→ 5개 시드 중 대부분에서 COMBINED가 BASE보다 못함. 공매도 feature는 이 조건에서 엣지 없음.")
    else:
        print(f"→ 시드마다 결과가 갈림 ({combined_beats_base_count}/{n_seeds}가 개선). 노이즈에 가까울 가능성 있음.")