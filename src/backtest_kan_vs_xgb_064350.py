"""
KAN vs XGBoost(BASE) -- 064350, 실제 triple-barrier 백테스트(비용 반영, threshold
sweep, 고정 B&H 구간 비교). run_seq_experiment_triple_barrier.py / backtest_triple_barrier.py
와 동일한 방법론.

배경: run_kan_base_baseline.py에서 KAN AUC(0.58~0.60)가 XGBoost AUC(0.56~0.57)보다
5/5 seed 일관되게 높게 나왔지만, vs_base_rate는 둘 다 크게 음수였음 -- 이건
learnings.md에 이미 기록된 "triple-barrier 라벨에서 vs_base_rate는 fold별 base rate
드리프트 때문에 구조적으로 불안정"한 아티팩트로 판단, accuracy 기반 지표는 아예 안 쓰고
바로 여기서 실제 백테스트로 판정.

⚠️ 사전등록: 064350 단독. proba >= threshold인 날 진입, holding_rows_tb만큼 보유 후
ret_tb로 청산(triple-barrier 배리어 도달까지 걸린 실제 거래일), 다음 진입은 그만큼
건너뜀. THRESHOLDS=[0.50,0.55,0.60,0.65,0.70] (production 검증 때와 동일 후보군),
5-seed(42/1/7/123/2024), ROUND_TRIP_COST=0.002. (model, seed) 조합당 학습은 1번만
하고 fold별 (test_idx, proba)를 캐싱해서 5개 threshold는 재학습 없이 거래 추출만
반복 (run_seq_experiment_triple_barrier.py와 동일 최적화).

사용법 (레포 루트에서):
    python src/backtest_kan_vs_xgb_064350.py
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

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
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
ROUND_TRIP_COST = 0.002

KAN_EPOCHS = 30
KAN_LR = 1e-3
KAN_BATCH_SIZE = 64
KAN_REG_WEIGHT = 1e-4

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# 1. walk-forward 분할
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
# 2. fold 하나 학습 -> (test_idx, proba) 캐시 (재학습 없이 threshold sweep용)
# ------------------------------------------------------------------
def normalize_fold_kan(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    z_train = np.clip((X_train - mean) / std, -3, 3) / 3
    z_test = np.clip((X_test - mean) / std, -3, 3) / 3
    return z_train, z_test


def fit_predict_fold_kan(X: np.ndarray, y: np.ndarray, train_idx, test_idx, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw = X[test_idx]
    if len(np.unique(y_train)) < 2:
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
    return proba


def fit_predict_fold_xgb(X: np.ndarray, y: np.ndarray, train_idx, test_idx, seed: int):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]
    if len(np.unique(y_train)) < 2:
        return None

    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def cache_fold_predictions(X: np.ndarray, y: np.ndarray, splits, model_name: str, seed: int):
    """(model, seed) 조합당 한 번만 학습 -- fold별 (test_idx, proba) 캐시 반환."""
    fit_fn = fit_predict_fold_kan if model_name == "KAN" else fit_predict_fold_xgb
    cached = []
    for train_idx, test_idx in splits:
        proba = fit_fn(X, y, train_idx, test_idx, seed)
        if proba is not None:
            cached.append((test_idx, proba))
    return cached


# ------------------------------------------------------------------
# 3. 캐시된 proba로 threshold별 거래 추출 (재학습 없음)
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, cached_predictions, threshold: float) -> pd.DataFrame:
    trades = []
    for test_idx, proba in cached_predictions:
        i = 0
        while i < len(test_idx):
            if proba[i] >= threshold:
                row_idx = test_idx[i]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx])
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": df.index[row_idx],
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1
    return pd.DataFrame(trades)


def get_fixed_bh_window(df: pd.DataFrame, splits):
    """어떤 seed가 어떤 날짜를 신호로 골랐는지와 무관하게 항상 동일한 B&H 비교 구간."""
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    last_test_idx = min(last_test_idx, len(df) - 1)
    return df.index[first_test_idx], df.index[last_test_idx]


def summarize_trades(trades: pd.DataFrame, bh_return: float) -> dict:
    if trades.empty:
        return {"n_trades": 0, "total_net_return": np.nan, "win_rate": np.nan, "excess_vs_bh": np.nan}
    total_net_return = (1 + trades["net_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()
    return {
        "n_trades": len(trades),
        "total_net_return": total_net_return,
        "win_rate": win_rate,
        "excess_vs_bh": total_net_return - bh_return,
    }


# ------------------------------------------------------------------
# 4. 메인: model x threshold x seed 전수비교
# ------------------------------------------------------------------
if __name__ == "__main__":
    if DEVICE.type == "cuda":
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}\n")
    else:
        print("GPU를 못 찾아서 CPU로 돌아감\n")

    df = build_triple_barrier_dataset(TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    X = df[FEATURE_COLS_BASE].values
    y = df["label_tb_binary"].values

    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO_DAYS)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    bh_start, bh_end = get_fixed_bh_window(df, splits)
    bh_return = df.loc[bh_end, "Close"] / df.loc[bh_start, "Close"] - 1
    print(f"=== {TICKER_NAME} ({TICKER_KRX}) triple-barrier 백테스트 ===")
    print(f"고정 B&H 구간: {bh_start.date()} ~ {bh_end.date()}, B&H 수익률={bh_return:+.2%}\n")

    rows = []
    for model_name in ["KAN", "XGBoost"]:
        for seed in SEEDS:
            t0 = time.time()
            cached = cache_fold_predictions(X, y, splits, model_name, seed)
            for threshold in THRESHOLDS:
                trades = generate_trades(df, cached, threshold)
                summary = summarize_trades(trades, bh_return)
                summary.update({"model": model_name, "threshold": threshold, "seed": seed})
                rows.append(summary)
            print(f"  [{model_name}] seed={seed} 완료 ({time.time() - t0:.1f}s)")

    result_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("=== model x threshold별 5-seed 요약 ===")
    print("=" * 70)
    grouped = result_df.groupby(["model", "threshold"]).agg(
        mean_n_trades=("n_trades", "mean"),
        mean_net_return=("total_net_return", "mean"),
        mean_excess_vs_bh=("excess_vs_bh", "mean"),
        seeds_beat_bh=("excess_vs_bh", lambda s: (s > 0).sum()),
    )
    print(grouped.round(4).to_string())

    print("\n5-seed 전부(5/5) excess_vs_bh > 0인 (model, threshold) 조합이 있는지 확인.")
    print("KAN이 XGBoost(production, threshold=0.60)보다 우위가 있는지가 핵심 -- 하나도")
    print("없으면 KAN 방향은 여기서 접는 게 맞음 (AUC 우위가 실전 수익으로 안 이어짐).")