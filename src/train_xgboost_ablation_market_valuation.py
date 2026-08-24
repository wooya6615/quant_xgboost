"""
코스피 지수 전체 PER/PBR feature 추가 전/후 성능을 비교하는 ablation 스크립트
(train_xgboost_ablation_rate_spread.py를 MARKET_VALUATION용으로 리네이밍)

핵심 설계:
    같은 데이터, 같은 fold 구성, 같은 하이퍼파라미터로
    BASE(가격 feature 13개) vs BASE + MARKET_VAL(코스피 지수 PER/PBR feature 4개 추가)만 비교.
    개별종목 밸류에이션(이미 통과)과는 다른 축 -- "시장 전체가 비싼 국면인가"를
    개별 feature와 별개로 넣었을 때 추가 정보가 있는지 확인.

사용법 (레포 루트에서):
    python src/train_xgboost_ablation_market_valuation.py

전제:
    feature_engineering_market_valuation.py를 먼저 실행해서
    {ticker_krx}_features_with_market_valuation_h{horizon}.csv 들이 만들어져 있어야 함.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

FEATURE_COLS_MARKET_VAL = [
    "market_per", "market_pbr",
    "market_per_zscore_252d", "market_pbr_zscore_252d",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_MARKET_VAL

FEATURE_COLS_MARKET_VAL_ONLY = FEATURE_COLS_MARKET_VAL

DEFAULT_HORIZON = 5


def load_dataset(ticker_krx: str = "064350", horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_with_market_valuation_h{horizon}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED + ["label"])
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

    fold_rows = []
    importances = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

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
            "n_rows": len(test_idx),
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


def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(
        df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    mval_fold_df, mval_importance = run_walk_forward(
        df, FEATURE_COLS_MARKET_VAL_ONLY, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(
        df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "MARKET_VAL_ONLY": mval_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["MARKET_VAL_ONLY-BASE"] = summary["MARKET_VAL_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    return {
        "base_fold_df": base_fold_df,
        "mval_fold_df": mval_fold_df,
        "combined_fold_df": combined_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "mval_importance": mval_importance,
        "combined_importance": combined_importance,
        "n_folds": len(base_fold_df),
    }


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
            "MARKET_VAL_auc": result["summary"].loc["auc", "MARKET_VAL_ONLY"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


def run_horizon_sweep(ticker_krx: str, horizons: list = None, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    horizon_rows = []
    seed_details = {}

    for horizon in horizons:
        print(f"\n{'#' * 60}")
        print(f"# horizon = {horizon}")
        print(f"{'#' * 60}")

        try:
            df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
        except FileNotFoundError:
            print(f"  {ticker_krx}_features_with_market_valuation_h{horizon}.csv 없음 -- "
                  f"feature_engineering_market_valuation.py를 먼저 실행하세요.")
            continue

        n_rows = len(df)
        label_rate = df["label"].mean()
        print(f"  데이터: {n_rows}행, label=1 비율 {label_rate:.3f}")

        seed_df = run_multi_seed(df, horizon=horizon, seeds=seeds, **wfo_kwargs)
        seed_details[horizon] = seed_df

        combined_beats_base_count = (seed_df["COMBINED_auc_diff"] > 0).sum()
        n_seeds = len(seed_df)

        horizon_rows.append({
            "horizon": horizon,
            "n_rows": n_rows,
            "n_folds": seed_df["n_folds"].iloc[0] if n_seeds else np.nan,
            "label_rate": label_rate,
            "COMBINED_auc_diff_mean": seed_df["COMBINED_auc_diff"].mean(),
            "COMBINED_auc_diff_std": seed_df["COMBINED_auc_diff"].std(),
            "COMBINED_beats_BASE_seeds": f"{combined_beats_base_count}/{n_seeds}",
            "MARKET_VAL_auc_mean": seed_df["MARKET_VAL_auc"].mean(),
        })

    horizon_summary = pd.DataFrame(horizon_rows)
    return horizon_summary, seed_details


if __name__ == "__main__":
    TICKER_KRX = "064350"

    print("=" * 60)
    print(f"=== 단일 horizon={DEFAULT_HORIZON} 빠른 확인 (seed=42) ===")
    print("=" * 60)
    df_default = load_dataset(ticker_krx=TICKER_KRX, horizon=DEFAULT_HORIZON)
    result_default = compare_feature_sets(df_default, horizon=DEFAULT_HORIZON, random_state=42)
    print(result_default["summary"].round(4).to_string())
    print(f"\nBASE feature importance top5:\n{result_default['base_importance'].head()}")
    print(f"\nMARKET_VAL_ONLY feature importance:\n{result_default['mval_importance']}")

    print("\n\n" + "=" * 60)
    print(f"=== 5-seed 검증 (horizon={DEFAULT_HORIZON}) ===")
    print("=" * 60)
    seed_df = run_multi_seed(df_default, horizon=DEFAULT_HORIZON)
    print(seed_df.round(4).to_string(index=False))

    combined_beats_base_count = (seed_df["COMBINED_auc_diff"] > 0).sum()
    n_seeds = len(seed_df)
    print(f"\nCOMBINED가 BASE보다 AUC 높았던 시드: {combined_beats_base_count} / {n_seeds}")
    print(f"AUC 차이 평균: {seed_df['COMBINED_auc_diff'].mean():+.4f} "
          f"(표준편차 {seed_df['COMBINED_auc_diff'].std():.4f})")

    print("\n\n" + "#" * 60)
    print("# Horizon 스윕 (1/3/5/10/20일)")
    print("#" * 60)

    horizon_summary, seed_details = run_horizon_sweep(ticker_krx=TICKER_KRX)

    print("\n" + "=" * 60)
    print("=== Horizon별 요약 ===")
    print("=" * 60)
    print(horizon_summary.round(4).to_string(index=False))

    if not horizon_summary.empty:
        best_row = horizon_summary.loc[horizon_summary["COMBINED_auc_diff_mean"].idxmax()]
        print(f"\nCOMBINED-BASE AUC 차이가 가장 큰 horizon: {int(best_row['horizon'])}일 "
              f"(평균 {best_row['COMBINED_auc_diff_mean']:+.4f}, 표준편차 {best_row['COMBINED_auc_diff_std']:.4f})")
        print("→ 표준편차가 평균의 절반을 넘으면 노이즈 가능성 높음. fold 개수도 같이 확인할 것.")
    else:
        print("사용 가능한 horizon 데이터셋이 없음")