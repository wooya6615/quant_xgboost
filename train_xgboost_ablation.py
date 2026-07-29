"""
수급 feature(외국인/기관 순매수) 추가 전/후 성능을 비교하는 ablation 스크립트

핵심 설계:
    같은 데이터, 같은 fold 구성, 같은 하이퍼파라미터로
    BASE(가격 feature 13개) vs BASE + INVESTOR(수급 feature 7개 추가)만 비교함.
    -- 그래야 성능 차이가 순수하게 "수급 feature 추가 효과"인지 확인 가능.
    (train_xgboost_wfo.py의 walk-forward 구조를 그대로 재사용)

사용법:
    python train_xgboost_ablation.py

전제:
    feature_engineering_investor.py를 먼저 실행해서
    005930_features_with_investor.csv가 만들어져 있어야 함.
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

FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR

FEATURE_COLS_INVESTOR_ONLY = FEATURE_COLS_INVESTOR  # 가격 feature 없이 수급 feature 7개만

HORIZON = 5  # feature_engineering_investor.py에서 쓴 horizon과 반드시 동일 (embargo에 사용)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(path: str = "064350_features_with_investor_h5.csv") -> pd.DataFrame:
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
# ------------------------------------------------------------------
def run_walk_forward(df: pd.DataFrame, feature_cols: list, train_size=300, test_size=60,
                      step=60, embargo=HORIZON, threshold=0.5, random_state=42):
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
def compare_feature_sets(df: pd.DataFrame, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, random_state=random_state, **wfo_kwargs)
    investor_fold_df, investor_importance = run_walk_forward(df, FEATURE_COLS_INVESTOR_ONLY, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, random_state=random_state, **wfo_kwargs)

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
# 5. 멀티 시드 검증 -- h5에서 나온 소폭 개선이 노이즈인지 진짜 신호인지 확인
#    같은 데이터/같은 fold 구성에서 XGBoost의 random_state만 바꿔가며 반복.
#    방향(COMBINED > BASE)이 시드마다 계속 바뀌면 우연, 계속 같은 방향이면 신호일 가능성.
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
            "INVESTOR_auc": result["summary"].loc["auc", "INVESTOR_ONLY"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()
    print(f"전체 데이터: {df.shape[0]}행\n")

    result = compare_feature_sets(df)

    print("\n" + "=" * 60)
    print("=== 전체 평균 비교 (fold 평균) ===")
    print("=" * 60)
    print(result["summary"].round(4))

    print(f"\n베이스라인 이긴 fold 수 (fold 총 {result['n_folds']}개)")
    print(f"  BASE:          {result['base_win_folds']} / {result['n_folds']}")
    print(f"  INVESTOR_ONLY: {result['investor_win_folds']} / {result['n_folds']}")
    print(f"  COMBINED:      {result['combined_win_folds']} / {result['n_folds']}")

    print("\n" + "=" * 60)
    print("=== INVESTOR_ONLY Feature Importance (상위순) ===")
    print("=" * 60)
    print(result["investor_importance"])

    print("\n" + "=" * 60)
    print("=== 판정 ===")
    print("=" * 60)

    # INVESTOR_ONLY가 base_rate(즉 다수클래스 찍기)보다 나은지가 핵심 질문
    investor_vs_base = result["summary"].loc["vs_base_rate", "INVESTOR_ONLY"]
    investor_auc = result["summary"].loc["auc", "INVESTOR_ONLY"]
    print(f"INVESTOR_ONLY만으로 vs_base_rate = {investor_vs_base:+.4f}, AUC = {investor_auc:.4f}")
    print("(AUC 0.5 = 완전 무작위, 0.5보다 유의미하게 높아야 '수급 데이터 자체에 신호가 있다'는 근거)\n")

    if investor_vs_base > 0 and investor_auc > 0.52:
        print("→ 수급 데이터만으로도 base_rate를 이김. 수급 자체엔 신호가 있다는 뜻.")
        combined_diff = result["summary"].loc["vs_base_rate", "COMBINED-BASE"]
        if combined_diff <= 0:
            print("→ 근데 COMBINED에서는 개선이 없었으니, 가격 feature와 섞이면서 희석/과적합된 것으로 추정.")
            print("→ 다음 시도: feature selection으로 가격 feature 개수를 줄이거나, 두 모델을 앙상블(가격 모델 + 수급 모델 확률 평균)하는 방향을 검토.")
        else:
            print("→ COMBINED에서도 개선됐다면 순수하게 좋은 신호.")
    else:
        print("→ 수급 데이터만으로는 base_rate를 못 이김 (AUC도 0.5 근처).")
        print("→ COMBINED가 나빴던 게 '노이즈로 인한 희석'이 아니라 '애초에 이 종목/이 기간엔 수급 자체에 신호가 없었다'는 쪽에 무게가 실림.")
        print("→ 종목을 유동성 낮은 중형주로 바꾸거나 horizon을 3~5일로 낮춰서 재검증 권장.")

    # ------------------------------------------------------------------
    # 멀티 시드 검증 -- 단일 시드(42)에서 나온 결과가 우연인지 확인
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
    investor_beats_50_count = (seed_df["INVESTOR_auc"] > 0.5).sum()

    print("\n" + "=" * 60)
    print("=== 최종 판정 (5개 시드 종합) ===")
    print("=" * 60)
    print(f"COMBINED가 BASE보다 AUC 높았던 시드:        {combined_beats_base_count} / {n_seeds}")
    print(f"COMBINED가 BASE보다 vs_base_rate 높았던 시드: {combined_vsbase_beats_count} / {n_seeds}")
    print(f"INVESTOR_ONLY의 AUC가 0.5 넘었던 시드:        {investor_beats_50_count} / {n_seeds}")
    print(f"\nAUC 차이(COMBINED-BASE) 평균: {seed_df['COMBINED_auc_diff'].mean():+.4f} (표준편차 {seed_df['COMBINED_auc_diff'].std():.4f})")
    print(f"vs_base_rate 차이 평균:       {seed_df['COMBINED_vsbase_diff'].mean():+.4f} (표준편차 {seed_df['COMBINED_vsbase_diff'].std():.4f})")

    print()
    if combined_beats_base_count >= n_seeds - 1 and combined_vsbase_beats_count >= n_seeds - 1:
        print("→ 5개 시드 중 대부분에서 COMBINED가 BASE를 이김. 재현 가능한 신호로 볼 여지가 있음.")
        print("→ 다음 단계: out-of-sample 기간을 더 늘리거나 다른 종목으로 교차검증 권장.")
    elif combined_beats_base_count <= 1 and combined_vsbase_beats_count <= 1:
        print("→ 5개 시드 중 대부분에서 COMBINED가 BASE보다 못함. 이전 단일 시드(42) 결과가 우연이었을 가능성이 높음.")
        print("→ 결론: 이 조건(삼성전자/horizon 5일/이 7개 수급 feature)에서는 재현 가능한 엣지를 확인하지 못함.")
    else:
        print(f"→ 시드마다 결과가 갈림 ({combined_beats_base_count}/{n_seeds}가 개선). 표준편차가 평균보다 크면 노이즈에 가까움.")
        print("→ 표준편차와 평균을 비교해서, 표준편차가 평균의 절반 이상이면 신호로 보기 어려움.")