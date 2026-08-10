"""
2차 모델(meta model) 검증: BASE+VALUATION feature로 meta_label(1차 베팅 성공 여부)을
예측할 수 있는지 5-seed walk-forward AUC로 확인.

train_xgboost_triple_barrier_ablation.py와 동일한 구조/파라미터 재사용 -- 라벨만
label_tb_binary -> meta_label로 바뀐 것.

사용법 (레포 루트에서):
    python src/train_xgboost_meta_labeling.py

전제:
    feature_engineering_meta_labeling.py로 064350_features_meta_labeling.csv가
    먼저 생성돼 있어야 함.
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
NUM_DAYS = 10  # feature_engineering_meta_labeling.py와 동일하게 맞출 것
HORIZON = NUM_DAYS  # embargo용


def load_dataset(ticker_krx: str = "064350") -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_meta_labeling_nd{NUM_DAYS}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


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


def run_walk_forward(df: pd.DataFrame, embargo: int, train_size: int = 300, test_size: int = 60,
                      step: int = 60, random_state: int = 42) -> dict:
    X = df[FEATURE_COLS_COMBINED]
    y = df["meta_label"]

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
    df = load_dataset("064350")
    print(f"데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})")
    print(f"meta_label 분포: {df['meta_label'].value_counts(normalize=True).to_dict()}\n")

    aucs = []
    for seed in SEEDS:
        r = run_walk_forward(df, embargo=HORIZON, random_state=seed)
        aucs.append(r["mean_auc"])
        print(f"seed={seed}: AUC={r['mean_auc']:.4f} (std={r['std_auc']:.4f}, folds={r['n_folds']})")

    aucs = np.array(aucs)
    print(f"\n5-seed 요약: 평균 AUC={aucs.mean():.4f}, 표준편차={aucs.std():.4f} "
          f"(std/mean={aucs.std() / aucs.mean():.1%})")
    print(f"5개 시드 중 0.5 초과: {int((aucs > 0.5).sum())}/5")
    print("판정: AUC가 0.5 근처면 메타 모델이 '이 베팅이 성공할지'를 구분 못 한다는 뜻 --")
    print("직접분류와 마찬가지로 백테스트까지 반드시 확인할 것 (다음 스크립트).")