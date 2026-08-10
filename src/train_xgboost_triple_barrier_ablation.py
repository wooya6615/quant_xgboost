"""
라벨 정의만 바꿔서 비교: 기존 고정 horizon 라벨(label_fixed) vs triple-barrier 라벨
(label_tb_binary) -- feature(BASE 13개)/모델(XGBoost)/walk-forward 구조는 전부 동일하게
유지해서 "라벨을 바꾸면 성능이 달라지는가"만 순수하게 봄. 지금까지의 ablation 원칙
(BASE/FEATURE_ONLY/COMBINED처럼 한 번에 하나만 바꾼다)을 라벨에도 그대로 적용.

사용법 (레포 루트에서):
    python src/train_xgboost_triple_barrier_ablation.py

전제:
    feature_engineering_triple_barrier.py로 064350_features_triple_barrier.csv가
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

SEEDS = [42, 1, 7, 123, 2024]  # 기존 컨벤션과 동일 -- 최소 5시드
HORIZON = 20  # embargo용 -- label_fixed(h=20)/label_tb(num_days=20) 둘 다 동일 기준으로 만들었으므로 공용


PT_SL_LABEL = "pt2sl1"  # feature_engineering_triple_barrier.py에서 저장한 파일명과 맞출 것 (1:2 손익비)


def load_dataset(ticker_krx: str = "064350") -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{PT_SL_LABEL}.csv"
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


def run_walk_forward(df: pd.DataFrame, label_col: str, embargo: int, train_size: int = 300,
                      test_size: int = 60, step: int = 60, random_state: int = 42) -> dict:
    X = df[FEATURE_COLS_BASE]
    y = df[label_col]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요. train_size/test_size를 줄이세요.")

    aucs = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue  # 해당 fold에 한 클래스만 있으면 AUC 계산 불가 -- 건너뜀 (건너뛴 fold 수는 n_folds로 확인)

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
    print(f"배리어 설정: {PT_SL_LABEL} (1:2 손익비 -- 손절 1배, 익절 2배)")
    print(f"데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})\n")

    results = {"label_fixed": [], "label_tb_binary": []}
    for label_col in results:
        print(f"=== {label_col} ===")
        for seed in SEEDS:
            r = run_walk_forward(df, label_col, embargo=HORIZON, random_state=seed)
            results[label_col].append(r["mean_auc"])
            print(f"  seed={seed}: AUC={r['mean_auc']:.4f} (std={r['std_auc']:.4f}, folds={r['n_folds']})")

    print("\n=== 5-seed 요약 ===")
    for label_col, aucs in results.items():
        aucs = np.array(aucs)
        print(f"{label_col}: 평균 AUC={aucs.mean():.4f}, 표준편차={aucs.std():.4f} "
              f"(std/mean={aucs.std() / aucs.mean():.1%})")

    diff = np.array(results["label_tb_binary"]) - np.array(results["label_fixed"])
    print(f"\ntriple-barrier - 기존 라벨 AUC 차이: 평균 {diff.mean():+.4f}, "
          f"5개 시드 중 양수 {int((diff > 0).sum())}/5")
    print("판정: 5/5 양수면 일관된 개선, 3/5 이하면 노이즈일 가능성 -- 기존 판정기준(std/mean 50% 이내,")
    print("방향 5/5 일치)과 동일 원칙. AUC 개선이 나와도 여기서 끝내지 말고 반드시 거래비용 반영")
    print("백테스트(ret_tb 컬럼 활용)로 실전 손익까지 확인할 것 -- 지금까지 매번 이 갭에서 무너졌음.")