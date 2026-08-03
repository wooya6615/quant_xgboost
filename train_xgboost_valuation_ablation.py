"""
밸류에이션(PER/PBR/배당수익률) feature 추가 전/후 성능을 비교하는 ablation 스크립트
(investor/short ablation과 동일한 구조 -- 재사용성을 위해 패턴 통일)

사용법:
    python train_xgboost_valuation_ablation.py

전제:
    feature_engineering_valuation.py의 build_multi_horizon_valuation_datasets()로
    {ticker_krx}_features_with_valuation_h{horizon}.csv 들이 만들어져 있어야 함.
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

FEATURE_COLS_VALUATION = [
    "per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION
FEATURE_COLS_VALUATION_ONLY = FEATURE_COLS_VALUATION

DEFAULT_HORIZON = 5


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(ticker_krx: str = "064350", horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    path = f"{ticker_krx}_features_with_valuation_h{horizon}.csv"
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
def run_walk_forward(df: pd.DataFrame, feature_cols: list, embargo: int, train_size=300, test_size=60,
                      step=60, threshold=0.5, random_state=42):
    X = df[feature_cols]
    y = df["label"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError(
            f"데이터가 부족해서 walk-forward split을 만들 수 없어요. "
            f"(n_rows={len(df)}, train_size={train_size}, test_size={test_size}, embargo={embargo})"
        )

    fold_rows = []
    importances = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
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
# 4. BASE vs VALUATION_ONLY vs COMBINED 비교
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    valuation_fold_df, valuation_importance = run_walk_forward(df, FEATURE_COLS_VALUATION_ONLY, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "VALUATION_ONLY": valuation_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["VALUATION_ONLY-BASE"] = summary["VALUATION_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    return {
        "base_fold_df": base_fold_df,
        "valuation_fold_df": valuation_fold_df,
        "combined_fold_df": combined_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "valuation_importance": valuation_importance,
        "combined_importance": combined_importance,
        "base_win_folds": (base_fold_df["vs_base_rate"] > 0).sum(),
        "valuation_win_folds": (valuation_fold_df["vs_base_rate"] > 0).sum(),
        "combined_win_folds": (combined_fold_df["vs_base_rate"] > 0).sum(),
        "n_folds": len(base_fold_df),
    }


# ------------------------------------------------------------------
# 5. 멀티 시드 검증
# ------------------------------------------------------------------
def run_multi_seed(df: pd.DataFrame, horizon: int, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
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
            "VALUATION_ONLY_auc": result["summary"].loc["auc", "VALUATION_ONLY"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 6. Horizon 스윕
# ------------------------------------------------------------------
def run_horizon_sweep(ticker_krx: str, horizons: list[int] = None, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]   # [추가] horizon=10에서 강한 신호가 나와서 20일도 확인

    horizon_rows = []
    seed_details = {}

    for horizon in horizons:
        print(f"\n{'#' * 60}")
        print(f"# horizon = {horizon}")
        print(f"{'#' * 60}")

        try:
            df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
        except FileNotFoundError:
            print(f"  {ticker_krx}_features_with_valuation_h{horizon}.csv 없음 -- "
                  f"feature_engineering_valuation.py의 build_multi_horizon_valuation_datasets()를 먼저 실행하세요.")
            continue

        n_rows = len(df)
        label_rate = df["label"].mean()
        print(f"  데이터: {n_rows}행, label=1 비율 {label_rate:.3f}")

        try:
            seed_df = run_multi_seed(df, horizon=horizon, seeds=seeds, **wfo_kwargs)
        except ValueError as e:
            print(f"  {e}")
            continue

        seed_details[horizon] = seed_df
        n_seeds = len(seed_df)
        combined_beats_base_count = (seed_df["COMBINED_auc_diff"] > 0).sum()

        horizon_rows.append({
            "horizon": horizon,
            "n_rows": n_rows,
            "n_folds": seed_df["n_folds"].iloc[0] if n_seeds else np.nan,
            "label_rate": label_rate,
            "COMBINED_auc_diff_mean": seed_df["COMBINED_auc_diff"].mean(),
            "COMBINED_auc_diff_std": seed_df["COMBINED_auc_diff"].std(),
            "COMBINED_beats_BASE_seeds": f"{combined_beats_base_count}/{n_seeds}",
            "VALUATION_ONLY_auc_mean": seed_df["VALUATION_ONLY_auc"].mean(),
        })

    return pd.DataFrame(horizon_rows), seed_details


if __name__ == "__main__":
    TICKER_KRX = "064350"  # 현대로템

    print("=" * 60)
    print(f"=== 단일 horizon 확인 (horizon={DEFAULT_HORIZON}) ===")
    print("=" * 60)

    df = load_dataset(ticker_krx=TICKER_KRX, horizon=DEFAULT_HORIZON)
    print(f"전체 데이터: {df.shape[0]}행\n")

    result = compare_feature_sets(df, horizon=DEFAULT_HORIZON)
    print("=== 전체 평균 비교 (fold 평균) ===")
    print(result["summary"].round(4))
    print(f"\n베이스라인 이긴 fold 수 (fold 총 {result['n_folds']}개)")
    print(f"  BASE:            {result['base_win_folds']} / {result['n_folds']}")
    print(f"  VALUATION_ONLY:  {result['valuation_win_folds']} / {result['n_folds']}")
    print(f"  COMBINED:        {result['combined_win_folds']} / {result['n_folds']}")

    print("\n" + "#" * 60)
    print("# 멀티 시드 검증 (5개 시드, horizon=5)")
    print("#" * 60)
    seed_df = run_multi_seed(df, horizon=DEFAULT_HORIZON)
    print(seed_df.round(4).to_string(index=False))

    print("\n\n" + "#" * 60)
    print("# Horizon 스윕 (1/3/5/10일)")
    print("#" * 60)

    horizon_summary, seed_details = run_horizon_sweep(ticker_krx=TICKER_KRX)

    print("\n" + "=" * 60)
    print("=== Horizon별 요약 ===")
    print("=" * 60)
    print(horizon_summary.round(4).to_string(index=False))