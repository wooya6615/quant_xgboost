"""
triple-barrier 라벨(label_tb_binary, 1:2 손익비)은 고정하고, feature만 BASE -> BASE+VALUATION
으로 바꿔서 비교. 현대로템은 기존 밸류에이션 ablation에서 h=20 기준 5/5 시드로 개선되고
2025년 제외해도 재현됐던 조합
이라, triple-barrier 라벨과 결합해도 그 개선이 재현되는지 확인하기 위함.

train_xgboost_triple_barrier_ablation.py(라벨 축 비교)와 이 스크립트(feature 축 비교)를
합치면 "라벨 x feature" 2x2를 다 확인하는 셈.

사용법 (레포 루트에서):
    python src/train_xgboost_triple_barrier_valuation_ablation.py

전제:
    feature_engineering_triple_barrier.py 실행 후
    064350_features_triple_barrier_pt2sl1.csv, 064350_features_triple_barrier_pt2sl1_valuation.csv
    둘 다 생성돼 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

SEEDS = [42, 1, 7, 123, 2024]
PT_SL_LABEL = "pt2sl1"
HORIZON = 20  # embargo용


def load_dataset(ticker_krx: str, combined: bool) -> pd.DataFrame:
    suffix = "_valuation" if combined else ""
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{PT_SL_LABEL}{suffix}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
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


def run_walk_forward(df: pd.DataFrame, feature_cols: list, embargo: int, train_size: int = 300,
                      test_size: int = 60, step: int = 60, random_state: int = 42) -> dict:
    X = df[feature_cols]
    y = df["label_tb_binary"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    aucs = []
    for train_idx, test_idx in splits:
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
        aucs.append(roc_auc_score(y_test, proba))

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc": float(np.std(aucs)) if aucs else float("nan"),
        "n_folds": len(aucs),
    }


if __name__ == "__main__":
    df_base = load_dataset("064350", combined=False)
    df_combined = load_dataset("064350", combined=True)
    print(f"BASE: {df_base.shape[0]}행 / COMBINED: {df_combined.shape[0]}행 "
          f"(밸류에이션 z-score 워밍업 구간만큼 COMBINED가 적은 게 정상)\n")

    results = {"BASE": [], "COMBINED": []}
    for name, df, cols in [("BASE", df_base, FEATURE_COLS_BASE), ("COMBINED", df_combined, FEATURE_COLS_COMBINED)]:
        print(f"=== {name} ({len(cols)}개 feature) ===")
        for seed in SEEDS:
            r = run_walk_forward(df, cols, embargo=HORIZON, random_state=seed)
            results[name].append(r["mean_auc"])
            print(f"  seed={seed}: AUC={r['mean_auc']:.4f} (std={r['std_auc']:.4f}, folds={r['n_folds']})")

    print("\n=== 5-seed 요약 ===")
    for name, aucs in results.items():
        aucs = np.array(aucs)
        print(f"{name}: 평균 AUC={aucs.mean():.4f}, 표준편차={aucs.std():.4f} "
              f"(std/mean={aucs.std() / aucs.mean():.1%})")

    diff = np.array(results["COMBINED"]) - np.array(results["BASE"])
    print(f"\nCOMBINED - BASE AUC 차이: 평균 {diff.mean():+.4f}, 5개 시드 중 양수 {int((diff > 0).sum())}/5")
    print("여기서 개선이 나와도 반드시 backtest_triple_barrier.py로 실전 손익까지 확인할 것 --")
    print("BASE만으로도 AUC는 좋았는데 백테스트에서 무너졌던 전례가 있음.")