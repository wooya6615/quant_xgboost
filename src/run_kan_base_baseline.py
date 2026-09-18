"""
KAN vs XGBoost(BASE) vs 동일가중 -- 064350, triple-barrier 프레임(pt_sl=(2,1),
num_days=30), 5-seed walk-forward.

배경: quant_cnn_chart(CNN), quant_seq_model(GRU/TCN) 전부 시퀀스/이미지로
표현력을 늘리는 방향이었지만 동일가중 대비 일관된 우위를 못 보임. KAN은
"엣지마다 학습되는 비선형 곡선"이라 시퀀스가 필요 없고, XGBoost와 동일한
BASE 13개 스칼라 피처에 그대로 붙일 수 있음 -- 같은 입력에서 아키텍처만
바꿨을 때 신호가 보이는지 가장 공정하게 확인하는 게 목적.

⚠️ 사전등록 (docs/prereg_kan_base_baseline.md, 결과 보고 나서 안 바꿈):
TRAIN=300/TEST=60/STEP=60/EMBARGO=30, 5-seed(42/1/7/123/2024), KAN=[13,16,2]
grid_size=5/spline_order=3, epoch=30/lr=1e-3/batch=64, 정규화손실 가중치 1e-4,
XGB_PARAMS는 기존 스크립트들과 동일.

1차 판정: vs_base_rate 5/5 양수인지만 먼저 봄. 둘 다 실패하면 -- 지금까지의
결론(신호 자체가 약함)을 재확인한 것으로 보고 접음. KAN만 통과하면 다음 단계로
triple-barrier 실제 백테스트(run_seq_experiment_triple_barrier.py와 동일 절차).

사용법 (레포 루트에서, feature_engineering_triple_barrier.py와 같은 src/ 안에 놓고):
    python src/run_kan_base_baseline.py
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE
from kan_classifier import KANClassifier

TICKER_KRX = "064350"
TICKER = "064350.KS"
TICKER_NAME = "현대로템"

PT_SL = (2, 1)
NUM_DAYS = 30
EMBARGO_DAYS = NUM_DAYS
TRAIN_SIZE, TEST_SIZE, STEP = 300, 60, 60
SEEDS = (42, 1, 7, 123, 2024)

# KAN 학습 설정 -- epoch=30은 GRU/TCN(epoch=10)보다 늘린 값. spline 계산이
# MLP/GRU보다 느려서 같은 epoch로는 수렴이 덜 될 수 있어 늘림 (사전등록에 고정).
KAN_EPOCHS = 30
KAN_LR = 1e-3
KAN_BATCH_SIZE = 64
KAN_REG_WEIGHT = 1e-4  # entropy+activation 정규화 -- spline이 노이즈에 과적합하는 것 억제

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 1. walk-forward 분할 (기존 스크립트들과 동일 로직)
# ------------------------------------------------------------------
def walk_forward_splits(n_rows: int, train_size: int, test_size: int, step: int, embargo: int):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = list(range(start, start + train_size))
        test_start = start + train_size + embargo
        test_idx = list(range(test_start, test_start + test_size))
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 2. KAN fold 학습 -- fold별 train 통계로 z-score 후 [-1,1] soft-clip
#    (grid 범위 밖으로 나가면 spline 외삽이 불안정해지므로 반드시 clip)
# ------------------------------------------------------------------
def normalize_fold_kan(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    z_train = np.clip((X_train - mean) / std, -3, 3) / 3
    z_test = np.clip((X_test - mean) / std, -3, 3) / 3
    return z_train, z_test


def train_eval_fold_kan(X: np.ndarray, y: np.ndarray, train_idx, test_idx, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    X_train, X_test = normalize_fold_kan(X_train_raw, X_test_raw)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    model = KANClassifier(n_features=X.shape[1], hidden=16, grid_size=5, spline_order=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=KAN_LR)
    criterion = nn.CrossEntropyLoss()

    n_train = len(X_train_t)
    model.train()
    for _ in range(KAN_EPOCHS):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, KAN_BATCH_SIZE):
            idx = perm[i:i + KAN_BATCH_SIZE]
            xb = X_train_t[idx].to(DEVICE)
            yb = y_train_t[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb) + KAN_REG_WEIGHT * model.regularization_loss()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(X_test_t), dim=1)[:, 1].cpu().numpy()
    pred = (proba >= 0.5).astype(int)
    base_rate = y_test.mean()

    return {
        "auc": roc_auc_score(y_test, proba),
        "vs_base_rate": accuracy_score(y_test, pred) - max(base_rate, 1 - base_rate),
        "n_test": len(y_test),
    }


# ------------------------------------------------------------------
# 3. XGBoost fold 학습 (기존 스크립트들과 동일 로직 -- 비교군)
# ------------------------------------------------------------------
def train_eval_fold_xgb(X: np.ndarray, y: np.ndarray, train_idx, test_idx, seed: int) -> dict:
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    base_rate = y_test.mean()

    return {
        "auc": roc_auc_score(y_test, proba),
        "vs_base_rate": accuracy_score(y_test, pred) - max(base_rate, 1 - base_rate),
        "n_test": len(y_test),
    }


# ------------------------------------------------------------------
# 4. 5-seed 검증
# ------------------------------------------------------------------
def run_multi_seed(X: np.ndarray, y: np.ndarray, model_name: str, seeds=SEEDS) -> pd.DataFrame:
    splits = walk_forward_splits(len(X), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO_DAYS)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    train_eval_fn = train_eval_fold_kan if model_name == "KAN" else train_eval_fold_xgb

    rows = []
    for seed in seeds:
        t0 = time.time()
        fold_rows = []
        for train_idx, test_idx in splits:
            result = train_eval_fn(X, y, train_idx, test_idx, seed)
            if result is not None:
                fold_rows.append(result)
        fold_df = pd.DataFrame(fold_rows)
        rows.append({
            "seed": seed,
            "mean_auc": fold_df["auc"].mean(),
            "mean_vs_base_rate": fold_df["vs_base_rate"].mean(),
            "win_folds": int((fold_df["vs_base_rate"] > 0).sum()),
            "n_folds": len(fold_df),
        })
        r = rows[-1]
        print(f"  [{model_name}] seed={seed}: AUC={r['mean_auc']:.4f}, "
              f"vs_base_rate={r['mean_vs_base_rate']:+.4f} "
              f"({r['win_folds']}/{r['n_folds']} fold 승, {time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if DEVICE.type == "cuda":
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU를 못 찾아서 CPU로 돌아감 (KAN은 spline 계산 때문에 CPU에서 특히 느릴 수 있음)\n")

    df = build_triple_barrier_dataset(TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
    # build_triple_barrier_dataset()은 label_tb(원본 -1/0/1)까지만 반환함 --
    # 이진 라벨은 기존 스크립트들(verify_base_only_triple_barrier.py 등)과 동일하게
    # 호출부에서 직접 생성.
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    X = df[FEATURE_COLS_BASE].values
    y = df["label_tb_binary"].values
    print(f"=== {TICKER_NAME} ({TICKER_KRX}) BASE 13개 피처, triple-barrier ===")
    print(f"데이터: {len(df)}행 ({df.index.min().date()} ~ {df.index.max().date()})")
    print(f"라벨 분포: {y.mean():.3f} (1의 비율)\n")

    all_results = {}
    for model_name in ["KAN", "XGBoost"]:
        print(f"=== {model_name} vs 동일가중 (5-seed walk-forward) ===")
        seed_df = run_multi_seed(X, y, model_name)
        print("\n" + seed_df.round(4).to_string(index=False))
        wins = int((seed_df["mean_vs_base_rate"] > 0).sum())
        print(f"\n{model_name}: vs_base_rate 5/5 양수 {wins}/5 {'(통과)' if wins == 5 else '(일관성 실패)'}\n")
        all_results[model_name] = seed_df

    print("=" * 60)
    kan_pass = (all_results["KAN"]["mean_vs_base_rate"] > 0).sum() == 5
    xgb_pass = (all_results["XGBoost"]["mean_vs_base_rate"] > 0).sum() == 5
    if not kan_pass and not xgb_pass:
        print("둘 다 5/5 실패 -- 기존 결론(신호 자체가 약함)을 재확인한 것으로 보고 접을 것.")
    elif kan_pass and not xgb_pass:
        print("KAN만 5/5 통과 -- triple-barrier 실제 백테스트(비용 반영, threshold sweep) 단계로 진행.")
    elif xgb_pass and not kan_pass:
        print("XGBoost만 5/5 통과 -- KAN이 이 조건에서는 XGBoost보다 못함. KAN 방향은 여기서 접을 것.")
    else:
        print("둘 다 5/5 통과 -- AUC/vs_base_rate 크기 비교 후 더 나은 쪽으로 백테스트 진행.")