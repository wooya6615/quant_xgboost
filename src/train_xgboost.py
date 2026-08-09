"""
nvda_features.csv를 이용한 XGBoost 방향성 분류 모델 학습
단순 시간순 80/20 분할 (앞 80% 학습, 뒤 20% 검증 — 랜덤 셔플 금지, 룩어헤드 방지)

사용법:
    python train_xgboost.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


FEATURE_COLS = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(path: str = None) -> pd.DataFrame:
    path = path or DATA_DIR / "nvda_features.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()  # 시간순 정렬 확인 (필수)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()  # inf 방어
    return df


# ------------------------------------------------------------------
# 2. 단순 시간순 분할 (앞 80% 학습, 뒤 20% 검증)
#    embargo: horizon만큼 train 뒤쪽을 비워서 라벨 겹침으로 인한 정보 누수 방지
#    예) horizon=10이면 train 마지막 10일의 라벨이 test 초반 구간까지 걸쳐있어 누수 위험
# ------------------------------------------------------------------
def train_test_split_by_time(df: pd.DataFrame, train_ratio: float = 0.8, embargo: int = 0):
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:max(split_idx - embargo, 1)]  # train 뒤쪽 embargo만큼 잘라냄
    test = df.iloc[split_idx:]
    return train, test


# ------------------------------------------------------------------
# 3. 모델 학습 + 평가
# ------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, train_ratio: float = 0.8, threshold: float = 0.5, embargo: int = 0):
    train, test = train_test_split_by_time(df, train_ratio, embargo)

    X_train, y_train = train[FEATURE_COLS], train["label"]
    X_test, y_test = test[FEATURE_COLS], test["label"]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,   # L1 정규화
        reg_lambda=1.0,  # L2 정규화
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)  # <- 여기서 threshold 적용

    metrics = {
        "train_period": f"{train.index[0].date()} ~ {train.index[-1].date()} ({len(train)}행)",
        "test_period": f"{test.index[0].date()} ~ {test.index[-1].date()} ({len(test)}행)",
        "threshold": threshold,
        "n_signals": int(pred.sum()),  # threshold 넘겨서 "상승"으로 예측한 개수
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
    }

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    cm = confusion_matrix(y_test, pred)

    return model, metrics, importance, cm, proba, y_test


# ------------------------------------------------------------------
# 3-1. Threshold sweep: 여러 threshold에서 precision/recall이 어떻게 바뀌는지 비교
# ------------------------------------------------------------------
def threshold_sweep(proba: np.ndarray, y_test: pd.Series, thresholds=None):
    if thresholds is None:
        thresholds = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        n_signals = int(pred.sum())
        rows.append({
            "threshold": t,
            "n_signals": n_signals,                       # threshold 넘긴 "상승" 신호 개수
            "signal_ratio": n_signals / len(pred),         # 전체 중 신호 비율
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # embargo는 feature_engineering.py에서 쓴 horizon 값과 동일하게 맞춰주세요.
    # (라벨이 'N일 후 수익률'이라 train 마지막 N일이 test 초반과 겹치기 때문)
    HORIZON = 10

    df = load_dataset()
    print(f"전체 데이터: {df.shape[0]}행, 라벨 분포:\n{df['label'].value_counts(normalize=True)}\n")

    model, metrics, importance, cm, proba, y_test = train_and_evaluate(df, threshold=0.5, embargo=HORIZON)

    print(f"학습 구간: {metrics['train_period']}")
    print(f"검증 구간: {metrics['test_period']}\n")

    print("=== 검증 성능 (threshold=0.5) ===")
    print(f"accuracy:  {metrics['accuracy']:.3f}")
    print(f"precision: {metrics['precision']:.3f}")
    print(f"recall:    {metrics['recall']:.3f}")
    print(f"auc:       {metrics['auc']:.3f}")

    print("\n=== Confusion Matrix ===")
    print(pd.DataFrame(cm, index=["실제 하락/보합", "실제 상승"], columns=["예측 하락/보합", "예측 상승"]))

    print("\n=== Feature Importance (상위순) ===")
    print(importance)

    print("\n=== Threshold Sweep ===")
    sweep_df = threshold_sweep(proba, y_test)
    print(sweep_df.to_string(index=False))