"""
수급 feature(외국인/기관 순매수) 추가 전/후 성능을 비교하는 ablation 스크립트

핵심 설계:
    같은 데이터, 같은 fold 구성, 같은 하이퍼파라미터로
    BASE(가격 feature 13개) vs BASE + INVESTOR(수급 feature 7개 추가)만 비교함.
    -- 그래야 성능 차이가 순수하게 "수급 feature 추가 효과"인지 확인 가능.
    (train_xgboost_wfo.py의 walk-forward 구조를 그대로 재사용)

[수정] horizon 파라미터화
    - 기존엔 HORIZON=5가 전역 상수로 박혀 있어서 embargo/파일 경로가 전부 거기 묶여있었음.
      horizon 스윕(1/3/5/10일)을 자동으로 돌리려면 함수 인자로 받아야 해서 구조 변경.
    - run_horizon_sweep(): feature_engineering_investor.py의 build_multi_horizon_datasets()가
      만든 {ticker_krx}_features_with_investor_h{horizon}.csv들을 순서대로 읽어서
      각 horizon마다 멀티시드 ablation을 돌리고, horizon별 COMBINED-BASE AUC 차이를
      한 표로 모아줌 -- "공매도/수급 신호가 며칠 후에 가장 잘 반영되는지" 한눈에 비교 가능.

사용법:
    python train_xgboost_ablation.py

전제:
    feature_engineering_investor.py를 먼저 실행해서
    {ticker_krx}_features_with_investor_h{horizon}.csv 들이 만들어져 있어야 함.
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

FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR

FEATURE_COLS_INVESTOR_ONLY = FEATURE_COLS_INVESTOR  # 가격 feature 없이 수급 feature 7개만

DEFAULT_HORIZON = 5  # 기존 실험과의 하위호환용 기본값 (단일 실행 시 사용)


# ------------------------------------------------------------------
# 1. 데이터 로드
#    [수정] 경로에 horizon을 넣어서 파일명을 동적으로 조합
# ------------------------------------------------------------------
def load_dataset(ticker_krx: str = "064350", horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_with_investor_h{horizon}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED + ["label"])
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (train_xgboost_wfo.py와 동일)
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
# 3. fold별 학습 + 평가 (feature_cols를 인자로 받아서 BASE/COMBINED 재사용)
#    [수정] embargo에 기본값 대신 horizon을 그대로 넘기도록 호출부에서 명시
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
#    [수정] embargo를 horizon 값으로 명시적으로 넘김
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    investor_fold_df, investor_importance = run_walk_forward(df, FEATURE_COLS_INVESTOR_ONLY, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "INVESTOR_ONLY": investor_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
    })
    summary["INVESTOR_ONLY-BASE"] = summary["INVESTOR_ONLY"] - summary["BASE"]
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]

    base_win_folds = (base_fold_df["vs_base_rate"] > 0).sum()
    investor_win_folds = (investor_fold_df["vs_base_rate"] > 0).sum()
    combined_win_folds = (combined_fold_df["vs_base_rate"] > 0).sum()

    return {
        "base_fold_df": base_fold_df,
        "investor_fold_df": investor_fold_df,
        "combined_fold_df": combined_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "investor_importance": investor_importance,
        "combined_importance": combined_importance,
        "base_win_folds": base_win_folds,
        "investor_win_folds": investor_win_folds,
        "combined_win_folds": combined_win_folds,
        "n_folds": len(base_fold_df),
    }


# ------------------------------------------------------------------
# 5. 멀티 시드 검증 -- 특정 horizon에서 나온 결과가 노이즈인지 진짜 신호인지 확인
#    같은 데이터/같은 fold 구성에서 XGBoost의 random_state만 바꿔가며 반복.
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
            "INVESTOR_auc": result["summary"].loc["auc", "INVESTOR_ONLY"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 6. [신규] Horizon 스윕 -- 여러 horizon에 대해 멀티시드 ablation을 자동 반복
#    "수급 신호가 며칠 후에 가장 잘 반영되는가"를 horizon별 AUC 차이로 비교
# ------------------------------------------------------------------
def run_horizon_sweep(ticker_krx: str, horizons: list[int] = None, seeds=(42, 1, 7, 123, 2024), **wfo_kwargs):
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
            print(f"  {ticker_krx}_features_with_investor_h{horizon}.csv 없음 -- "
                  f"feature_engineering_investor.py의 build_multi_horizon_datasets()를 먼저 실행하세요.")
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
            "INVESTOR_auc_mean": seed_df["INVESTOR_auc"].mean(),
        })

    horizon_summary = pd.DataFrame(horizon_rows)
    return horizon_summary, seed_details


if __name__ == "__main__":
    TICKER_KRX = "064350"  # 현대로템

    # ------------------------------------------------------------------
    # (A) 단일 horizon만 볼 때는 기존 방식 그대로 유지 (하위호환)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"=== 단일 horizon 확인 (horizon={DEFAULT_HORIZON}) ===")
    print("=" * 60)

    df = load_dataset(ticker_krx=TICKER_KRX, horizon=DEFAULT_HORIZON)
    print(f"전체 데이터: {df.shape[0]}행\n")

    result = compare_feature_sets(df, horizon=DEFAULT_HORIZON)
    print("=== 전체 평균 비교 (fold 평균) ===")
    print(result["summary"].round(4))
    print(f"\n베이스라인 이긴 fold 수 (fold 총 {result['n_folds']}개)")
    print(f"  BASE:          {result['base_win_folds']} / {result['n_folds']}")
    print(f"  INVESTOR_ONLY: {result['investor_win_folds']} / {result['n_folds']}")
    print(f"  COMBINED:      {result['combined_win_folds']} / {result['n_folds']}")

    # ------------------------------------------------------------------
    # (B) [신규] Horizon 스윕 -- 1/3/5/10일 자동 비교
    #     사전 조건: feature_engineering_investor.py의 build_multi_horizon_datasets()로
    #     {TICKER_KRX}_features_with_investor_h{horizon}.csv 들이 이미 생성돼 있어야 함
    # ------------------------------------------------------------------
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
        print("→ n_folds가 너무 작은 horizon(특히 짧은 horizon일수록 fold가 줄 수 있음)은 결론에서 가중치를 낮출 것.")
    else:
        print("사용 가능한 horizon 데이터셋이 없음 -- 먼저 feature_engineering_investor.py를 실행하세요.")