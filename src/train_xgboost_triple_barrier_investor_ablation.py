"""
Triple-barrier 라벨(label_tb_binary)을 고정하고, feature만 BASE -> BASE+INVESTOR로
바꿔서 AUC 수준에서 비교. (train_xgboost_triple_barrier_valuation_ablation.py의
investor 버전 -- 라벨/모델/walk-forward 구조는 전부 동일, feature 축만 바꿈)

배경: backtest_triple_barrier_pooled_investor.py 5-seed 결과가 0/5로 Buy&Hold를
전혀 못 이겼음 (BASE+VALUATION 대비 뚜렷하게 나쁨). 근데 이게
    (a) 수급 feature가 모델의 판별력(AUC) 자체를 깎아먹은 건지
    (b) AUC는 비슷한데 threshold=0.65 트레이드 진입 시점/빈도가 나빴던 건지
구분이 안 된 상태라, 백테스트 레이어를 걷어내고 AUC만 순수하게 비교해서 원인을 좁힘.

가설: 수급 feature(foreign_net_3d/5d 등)는 h=5에서만 검증된 "단기 소진성" 신호라,
num_days=20짜리 triple-barrier(최대 20일 보유) 프레임에서는 신호가 이미 소진돼
노이즈로만 작용할 가능성. COMBINED가 BASE보다 AUC가 낮게 나오면 이 가설을 뒷받침함.

사용법 (레포 루트에서):
    python src/train_xgboost_triple_barrier_investor_ablation.py

전제:
    feature_engineering_triple_barrier_investor.py로 3종목 다
    {ticker}_features_triple_barrier_pt2sl1_nd20_hl_investor.csv가 생성돼 있어야 함.
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
FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR

CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKERS = ["064350", "052690", "118990"]
SEEDS = [42, 1, 7, 123, 2024]
NUM_DAYS = 20  # embargo용


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_investor.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
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
                      test_size: int = 60, step: int = 60, random_state: int = 42) -> pd.DataFrame:
    X = df[feature_cols]
    y = df["label_tb_binary"]

    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "fold": fold,
            "auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        })

    return pd.DataFrame(rows)


def compare_ticker(ticker_krx: str, seed: int) -> dict:
    df = load_dataset(ticker_krx)
    base_fold = run_walk_forward(df, FEATURE_COLS_BASE, embargo=NUM_DAYS, random_state=seed)
    combined_fold = run_walk_forward(df, FEATURE_COLS_COMBINED, embargo=NUM_DAYS, random_state=seed)

    return {
        "ticker": ticker_krx,
        "seed": seed,
        "n_folds": len(base_fold),
        "BASE_auc": base_fold["auc"].mean(),
        "COMBINED_auc": combined_fold["auc"].mean(),
        "COMBINED_auc_diff": combined_fold["auc"].mean() - base_fold["auc"].mean(),
    }


if __name__ == "__main__":
    all_rows = []
    for ticker in TICKERS:
        print(f"\n{'=' * 60}\n=== {ticker} ===\n{'=' * 60}")
        for seed in SEEDS:
            r = compare_ticker(ticker, seed)
            all_rows.append(r)
            print(f"  seed={seed}: BASE_auc={r['BASE_auc']:.4f}, "
                  f"COMBINED_auc={r['COMBINED_auc']:.4f}, diff={r['COMBINED_auc_diff']:+.4f}")

    result_df = pd.DataFrame(all_rows)

    print("\n\n" + "=" * 60)
    print("=== 종목별 5-seed 평균 요약 ===")
    print("=" * 60)
    summary = result_df.groupby("ticker")[["BASE_auc", "COMBINED_auc", "COMBINED_auc_diff"]].mean()
    print(summary.round(4).to_string())

    n_negative = (result_df["COMBINED_auc_diff"] < 0).sum()
    n_total = len(result_df)
    print(f"\nCOMBINED_auc_diff < 0 인 (종목 x 시드) 조합: {n_negative}/{n_total}")

    print("\n[해석 가이드]")
    print("- COMBINED_auc_diff가 대부분/전부 음수면: 수급 feature가 AUC 레벨에서부터")
    print("  이미 노이즈로 작용한다는 뜻 -- h=5에서만 검증된 신호가 h=20(20일 보유)")
    print("  프레임에서는 소진돼 있다는 가설과 일치. 이 조합 자체를 폐기하는 게 맞음.")
    print("- COMBINED_auc_diff가 대체로 양수인데 백테스트는 0/5로 졌다면: AUC 개선이")
    print("  실제 트레이드 타이밍/threshold와는 안 맞았다는 뜻 -- threshold sweep이나")
    print("  다른 num_days로 재검증해볼 여지가 있음 (전자보다 살릴 가능성 있는 경우).")