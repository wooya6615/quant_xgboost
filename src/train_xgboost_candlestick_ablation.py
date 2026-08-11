"""
캔들스틱 패턴 feature 추가 전/후 성능을 비교하는 ablation 스크립트 (현대로템, horizon=20).
(valuation/short/investor ablation과 동일한 구조 -- 재사용성을 위해 패턴 통일)

3자 비교:
    BASE            (13개, 가격/거래량 파생)
    COMBINED        (BASE + VALUATION 5개)      -- 지금까지 검증된 최선 조합
    COMBINED_CANDLE (BASE + VALUATION + CANDLE 11개) -- 여기에 캔들 패턴을 얹었을 때 변화 확인

horizon=20으로 고정한 이유: 밸류에이션 실험에서 h=20이 COMBINED-BASE AUC 단조증가
패턴의 정점이었고, 2025년(강세장) 제외해도 재현되는 게 확인된 유일한 조합
(PROJECT_SUMMARY.md 7절 참고) -- 여기에 캔들 패턴을 얹었을 때 개선/훼손 여부를 보는 게
가장 의미 있는 비교임.

사용법 (레포 루트에서):
    python src/train_xgboost_candlestick_ablation.py

전제:
    feature_engineering_candlestick.py로 064350_features_with_candlestick_h20.csv가
    만들어져 있어야 함 (없으면 이 스크립트가 자동으로 생성).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from feature_engineering_candlestick import (
    FEATURE_COLS_BASE, FEATURE_COLS_VALUATION, FEATURE_COLS_CANDLE,
    FEATURE_COLS_COMBINED, FEATURE_COLS_COMBINED_CANDLE,
    build_feature_dataset_with_candlestick,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HORIZON = 20
COST_THRESHOLD = 0.012
SEEDS = (42, 1, 7, 123, 2024)


# ------------------------------------------------------------------
# 1. 데이터 로드 (없으면 생성)
# ------------------------------------------------------------------
def load_dataset(ticker_krx: str = "064350", horizon: int = HORIZON) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_with_candlestick_h{horizon}.csv"
    if not path.exists():
        print(f"{path.name}이 없어서 새로 생성합니다...")
        build_feature_dataset_with_candlestick(
            horizon=horizon, cost_threshold=COST_THRESHOLD,
        ).to_csv(path)

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED_CANDLE + ["label"])
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
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

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
            "auc": roc_auc_score(y_test, proba),
        })
        importances.append(model.feature_importances_)

    fold_df = pd.DataFrame(fold_rows)
    importance = pd.Series(np.mean(importances, axis=0), index=feature_cols).sort_values(ascending=False)
    return fold_df, importance


# ------------------------------------------------------------------
# 4. BASE vs COMBINED vs COMBINED_CANDLE 비교
# ------------------------------------------------------------------
def compare_feature_sets(df: pd.DataFrame, horizon: int, random_state=42, **wfo_kwargs):
    base_fold_df, base_importance = run_walk_forward(df, FEATURE_COLS_BASE, embargo=horizon, random_state=random_state, **wfo_kwargs)
    combined_fold_df, combined_importance = run_walk_forward(df, FEATURE_COLS_COMBINED, embargo=horizon, random_state=random_state, **wfo_kwargs)
    candle_fold_df, candle_importance = run_walk_forward(df, FEATURE_COLS_COMBINED_CANDLE, embargo=horizon, random_state=random_state, **wfo_kwargs)

    metrics = ["accuracy", "vs_base_rate", "precision", "recall", "auc"]
    summary = pd.DataFrame({
        "BASE": base_fold_df[metrics].mean(),
        "COMBINED": combined_fold_df[metrics].mean(),
        "COMBINED_CANDLE": candle_fold_df[metrics].mean(),
    })
    summary["COMBINED-BASE"] = summary["COMBINED"] - summary["BASE"]
    summary["COMBINED_CANDLE-COMBINED"] = summary["COMBINED_CANDLE"] - summary["COMBINED"]
    summary["COMBINED_CANDLE-BASE"] = summary["COMBINED_CANDLE"] - summary["BASE"]

    return {
        "base_fold_df": base_fold_df,
        "combined_fold_df": combined_fold_df,
        "candle_fold_df": candle_fold_df,
        "summary": summary,
        "base_importance": base_importance,
        "combined_importance": combined_importance,
        "candle_importance": candle_importance,
        "base_win_folds": (base_fold_df["vs_base_rate"] > 0).sum(),
        "combined_win_folds": (combined_fold_df["vs_base_rate"] > 0).sum(),
        "candle_win_folds": (candle_fold_df["vs_base_rate"] > 0).sum(),
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
            "COMBINED_CANDLE_auc": result["summary"].loc["auc", "COMBINED_CANDLE"],
            "COMBINED-BASE_diff": result["summary"].loc["auc", "COMBINED-BASE"],
            "COMBINED_CANDLE-COMBINED_diff": result["summary"].loc["auc", "COMBINED_CANDLE-COMBINED"],
            "COMBINED_CANDLE-BASE_diff": result["summary"].loc["auc", "COMBINED_CANDLE-BASE"],
            "n_folds": result["n_folds"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()
    print(f"데이터셋: {df.shape[0]}행, 기간: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"라벨 분포: {df['label'].value_counts(normalize=True).to_dict()}\n")

    print(f"=== 5-seed 비교 (horizon={HORIZON}) ===")
    seed_df = run_multi_seed(df, horizon=HORIZON)
    print(seed_df.round(4).to_string(index=False))

    print("\n=== 요약 ===")
    for col, label in [
        ("COMBINED-BASE_diff", "COMBINED - BASE"),
        ("COMBINED_CANDLE-COMBINED_diff", "COMBINED_CANDLE - COMBINED (캔들 패턴 순수 기여)"),
        ("COMBINED_CANDLE-BASE_diff", "COMBINED_CANDLE - BASE"),
    ]:
        diffs = seed_df[col].values
        n_positive = int((diffs > 0).sum())
        print(f"{label}: 평균 {diffs.mean():+.4f}, {n_positive}/{len(diffs)} seed 양수 "
              f"{'(5/5 일관성 통과)' if n_positive == len(diffs) else '(일관성 실패 -- 노이즈일 가능성)'}")

    print("\n마지막 seed(2024) 기준 COMBINED_CANDLE feature importance 상위 10개:")
    last_result = compare_feature_sets(df, horizon=HORIZON, random_state=SEEDS[-1])
    print(last_result["candle_importance"].head(10))

    print("\n⚠️ 여기서 AUC가 개선돼도 반드시 backtest_valuation_comparison.py 패턴으로")
    print("   거래비용 반영 실전 손익까지 확인할 것 -- AUC 개선이 백테스트에서 무너진 전례가 반복됨.")