"""
nvda_features.csv를 이용한 XGBoost 방향성 분류 - Walk-Forward 버전

핵심 포인트:
- fold마다 학습 구간이 슬라이드하며, 각 fold의 '검증 구간 베이스레이트'를 같이 찍음
  (단순 80/20 분할에서는 검증 구간이 우연히 추세장/횡보장에 걸리면 결과가 크게 왜곡됨 -> WFO로 여러 국면을 다 봄)
- embargo: train 마지막 구간과 test 사이에 horizon만큼 gap을 둬서 라벨 겹침으로 인한 누수 방지
- 모든 fold의 예측 확률을 모아서(pooled) threshold sweep -> 국면에 상관없이 일관되게 베이스레이트를 이기는지 확인

사용법:
    python train_xgboost_wfo.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


FEATURE_COLS = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

HORIZON = 10  # feature_engineering.py에서 쓴 horizon과 반드시 동일하게 맞출 것 (embargo에 사용)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(path: str = None) -> pd.DataFrame:
    path = path or DATA_DIR / "nvda_features.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (train -> embargo gap -> test, step만큼 슬라이드)
# ------------------------------------------------------------------
def walk_forward_splits(n_rows: int, train_size: int, test_size: int, step: int, embargo: int):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo  # embargo만큼 비우고 test 시작
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


# ------------------------------------------------------------------
# 3. fold별 학습 + 평가, 전체 pooled 결과 반환
# ------------------------------------------------------------------
def run_walk_forward(df: pd.DataFrame, train_size=300, test_size=60, step=60, embargo=HORIZON, threshold=0.5):
    X = df[FEATURE_COLS]
    y = df["label"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요. train_size/test_size를 줄이세요.")

    fold_rows = []
    pooled_proba, pooled_y = [], []
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
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        base_rate = y_test.mean()  # 이 fold 검증 구간의 실제 상승 비율
        acc = accuracy_score(y_test, pred)

        fold_rows.append({
            "fold": fold,
            "test_start": df.index[list(test_idx)[0]].date(),
            "test_end": df.index[list(test_idx)[-1]].date(),
            "base_rate": base_rate,
            "accuracy": acc,
            "vs_base_rate": acc - max(base_rate, 1 - base_rate),  # 베이스라인(다수클래스 찍기) 대비 우위
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        })

        pooled_proba.append(proba)
        pooled_y.append(y_test.values)
        importances.append(model.feature_importances_)

    fold_df = pd.DataFrame(fold_rows)
    pooled_proba = np.concatenate(pooled_proba)
    pooled_y = np.concatenate(pooled_y)
    importance = pd.Series(np.mean(importances, axis=0), index=FEATURE_COLS).sort_values(ascending=False)

    return fold_df, pooled_proba, pooled_y, importance


# ------------------------------------------------------------------
# 4. Pooled threshold sweep (모든 fold 예측을 합쳐서 확인)
# ------------------------------------------------------------------
def threshold_sweep(proba: np.ndarray, y_true: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    base_rate = y_true.mean()
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        n_signals = int(pred.sum())
        prec = precision_score(y_true, pred, zero_division=0)
        rows.append({
            "threshold": t,
            "n_signals": n_signals,
            "signal_ratio": n_signals / len(pred),
            "precision": prec,
            "vs_base_rate": prec - base_rate,  # 이게 핵심: 양수여야 엣지가 있다는 뜻
            "recall": recall_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 5. 고신뢰 신호(high threshold)가 어느 fold(시기)에 몰려있는지 확인
#    신호가 특정 국면에만 쏠려있으면 "모델의 판별력"이 아니라
#    "그 국면에서만 상승 예측을 많이 한 것"일 수 있음
# ------------------------------------------------------------------
def signal_concentration_by_fold(df: pd.DataFrame, train_size=300, test_size=60, step=60,
                                  embargo=HORIZON, threshold=0.7):
    X = df[FEATURE_COLS]
    y = df["label"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        n_signals = int(pred.sum())
        if n_signals > 0:
            fold_precision = precision_score(y_test, pred, zero_division=0)
        else:
            fold_precision = np.nan  # 이 fold엔 threshold 넘는 신호가 아예 없음

        rows.append({
            "fold": fold,
            "test_start": df.index[list(test_idx)[0]].date(),
            "test_end": df.index[list(test_idx)[-1]].date(),
            "base_rate": y_test.mean(),
            "n_signals": n_signals,
            "signal_ratio": n_signals / len(y_test),
            "precision_on_signals": fold_precision,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_dataset()
    print(f"전체 데이터: {df.shape[0]}행, 전체 라벨 분포:\n{df['label'].value_counts(normalize=True)}\n")

    print("=== Threshold=0.7 신호의 fold별(시기별) 분포 ===")
    conc_df = signal_concentration_by_fold(df, threshold=0.7)
    print(conc_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    n_folds_with_signal = (conc_df["n_signals"] > 0).sum()
    print(f"\n신호가 하나라도 있는 fold: {n_folds_with_signal} / {len(conc_df)}")
    print(f"신호 개수 표준편차: {conc_df['n_signals'].std():.2f} (fold당 평균 {conc_df['n_signals'].mean():.2f}개)")
    print(f"신호가 몰린 상위 3개 fold:\n{conc_df.nlargest(3, 'n_signals')[['fold', 'test_start', 'test_end', 'n_signals', 'base_rate']]}")