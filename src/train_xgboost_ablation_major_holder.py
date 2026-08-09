"""
DART 대량보유상황보고서(5% Rule) feature 추가 전/후 성능을 비교하는 ablation 스크립트
(train_xgboost_ablation_dart.py를 리네이밍 -- 구조/로직은 100% 동일, DART -> MAJOR_HOLDER)

⚠️ 표본 희소성 주의:
    현대로템 기준 11.5년간 총 31건(연평균 2.7건)의 이벤트만 존재함. 이 정도 희소성이면
    walk-forward의 test_size=60(영업일)짜리 fold 중 상당수가 "이벤트가 아예 없는 기간"일
    가능성이 높음 -- 그 경우 major_holder_count_20d/60d, flag_20d는 거의 항상 0으로 고정된
    feature가 되어 모델이 학습할 변동성 자체가 부족할 수 있음. AUC가 안 나오거나 노이즈만
    나온다면 "신호가 없다"보다는 "표본이 부족해서 검증 자체가 어렵다"에 가까운 결론으로
    해석해야 함 (다른 실험들의 "신호 없음" 결론과는 성격이 다름).

사용법:
    python train_xgboost_ablation_major_holder.py

전제:
    feature_engineering_dart_major_holder.py를 먼저 실행해서
    {ticker_krx}_features_with_major_holder_h{horizon}.csv 들이 만들어져 있어야 함.
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

FEATURE_COLS_MAJOR_HOLDER = [
    "major_holder_count_20d", "major_holder_count_60d",
    "major_holder_flag_20d", "days_since_major_holder_filing",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_MAJOR_HOLDER

FEATURE_COLS_MAJOR_HOLDER_ONLY = FEATURE_COLS_MAJOR_HOLDER

DEFAULT_HORIZON = 5


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(ticker_krx: str = "064350", horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_with_major_holder_h{horizon}.csv"
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
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요. train_size/test_size를 줄이세요.")

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
# 4. BASE vs COMBINED 비교
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    mh_fold_df, mh_importance = run_walk_forward(df, FEATURE_COLS_MAJOR_HOLDER_ONLY, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "MAJOR_HOLDER_ONLY": mh_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["MAJOR_HOLDER_ONLY-BASE"] = summary["MAJOR_HOLDER_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    base_win_folds = (base_fold_df["vs_base_rate"] > 0).sum()
    mh_win_folds = (mh_fold_df["vs_base_rate"] > 0).sum()
    combined_win_folds = (combined_fold_df["vs_base_rate"] > 0).sum()

    return {
        "base_fold_df": base_fold_df,
        "mh_fold_df": mh_fold_df,
        "combined_fold_df": combined_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "mh_importance": mh_importance,
        "combined_importance": combined_importance,
        "base_win_folds": base_win_folds,
        "mh_win_folds": mh_win_folds,
        "combined_win_folds": combined_win_folds,
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
            "MAJOR_HOLDER_auc": result["summary"].loc["auc", "MAJOR_HOLDER_ONLY"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 6. Horizon 스윕
# ------------------------------------------------------------------
def run_horizon_sweep(ticker_krx: str, horizons: list = None, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
    if horizons is None:
        horizons = [1, 3, 5, 10]

    horizon_rows = []
    seed_details = {}

    for horizon in horizons:
        print(f"\n{'#' * 60}")
        print(f"# horizon = {horizon}")
        print(f"{'#' * 60}")

        try:
            df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
        except FileNotFoundError:
            print(f"  {ticker_krx}_features_with_major_holder_h{horizon}.csv 없음 -- "
                  f"feature_engineering_dart_major_holder.py를 먼저 실행하세요.")
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
            "MAJOR_HOLDER_auc_mean": seed_df["MAJOR_HOLDER_auc"].mean(),
        })

    horizon_summary = pd.DataFrame(horizon_rows)
    return horizon_summary, seed_details


if __name__ == "__main__":
    TICKER_KRX = "064350"  # 현대로템

    print("=" * 60)
    print(f"=== 단일 horizon 확인 (horizon={DEFAULT_HORIZON}) ===")
    print("=" * 60)

    df = load_dataset(ticker_krx=TICKER_KRX, horizon=DEFAULT_HORIZON)
    print(f"전체 데이터: {df.shape[0]}행\n")
    print(f"[표본 희소성 체크] flag_20d=1인 행 비율: {(df['major_holder_flag_20d'] == 1).mean():.1%}")

    result = compare_feature_sets(df, horizon=DEFAULT_HORIZON)
    print("\n=== 전체 평균 비교 (fold 평균) ===")
    print(result["summary"].round(4))
    print(f"\n베이스라인 이긴 fold 수 (fold 총 {result['n_folds']}개)")
    print(f"  BASE:               {result['base_win_folds']} / {result['n_folds']}")
    print(f"  MAJOR_HOLDER_ONLY:  {result['mh_win_folds']} / {result['n_folds']}")
    print(f"  COMBINED:           {result['combined_win_folds']} / {result['n_folds']}")

    print("\n\n" + "#" * 60)
    print("# Horizon 스윕 (1/3/5/10일 자동 비교)")
    print("#" * 60)

    horizon_summary, seed_details = run_horizon_sweep(ticker_krx=TICKER_KRX)

    print("\n" + "=" * 60)
    print("=== Horizon별 요약 ===")
    print("=" * 60)
    print(horizon_summary.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("=== 판정 ===")
    print("=" * 60)
    if not horizon_summary.empty:
        best_row = horizon_summary.loc[horizon_summary["COMBINED_auc_diff_mean"].idxmax()]
        print(f"COMBINED-BASE AUC 차이가 가장 큰 horizon: {int(best_row['horizon'])}일 "
              f"(평균 {best_row['COMBINED_auc_diff_mean']:+.4f}, 표준편차 {best_row['COMBINED_auc_diff_std']:.4f})")
        print("→ 표준편차가 평균의 절반을 넘으면 그 horizon 결과는 노이즈에 가까우니 주의.")
        print("→ 표본이 희소한 feature이므로(연평균 2.7건), 결과가 약하게 나와도 '신호 없음'이 아니라")
        print("  '표본 부족으로 검증력 자체가 낮음'일 수 있음을 감안할 것.")
    else:
        print("사용 가능한 horizon 데이터셋이 없음 -- 먼저 feature_engineering_dart_major_holder.py를 실행하세요.")