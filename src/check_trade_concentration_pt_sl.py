"""
[확인] pt_sl 후보(파일명 기반)의 거래/연도 집중도 -- check_trade_concentration.py를
pt_sl 축 후보에도 쓸 수 있게 일반화한 버전.

backtest_pt1sl1_064350.py는 모듈 상단 PT_SL 상수를 바꿔가며 재사용하는 구조라,
사용자가 그 상수를 (1,1)->(3,1)로 계속 바꾸면 이전 실행 결과를 재현하기 어려움.
이 스크립트는 파일명(pt_sl 라벨)을 직접 인자로 받아서 어떤 pt_sl 후보든
독립적으로 집중도를 확인할 수 있게 함.

사용법 (레포 루트에서):
    python src/check_trade_concentration_pt_sl.py <pt> <sl> <threshold> <seed>
    예: python src/check_trade_concentration_pt_sl.py 3 1 0.50 42
        python src/check_trade_concentration_pt_sl.py 3 1 0.50 1

전제:
    compute_pbo_pt_sl_064350.py 또는 backtest_pt1sl1_064350.py로 해당 pt_sl의
    data/064350_features_triple_barrier_pt{p}sl{s}_nd20_hl_base.csv가 이미
    생성돼 있어야 함.
"""

import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKER_KRX = "064350"
NUM_DAYS = 20
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002


def walk_forward_splits(n_rows, train_size, test_size, step, embargo):
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_rows:
        train_idx = range(start, start + train_size)
        test_start = start + train_size + embargo
        test_idx = range(test_start, test_start + test_size)
        splits.append((train_idx, test_idx))
        start += step
    return splits


def generate_trades(df: pd.DataFrame, threshold: float, seed: int) -> pd.DataFrame:
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1

    return pd.DataFrame(trades)


def analyze(pt: float, sl: float, threshold: float, seed: int):
    label = f"pt{pt}sl{sl}_nd{NUM_DAYS}_hl"
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{label}_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)

    trades = generate_trades(df, threshold, seed).sort_values("entry_date").reset_index(drop=True)
    total_return = (1 + trades["net_return"]).prod() - 1

    print(f"\n{'#' * 60}\n# pt_sl=({pt},{sl}), threshold={threshold}, seed={seed}\n{'#' * 60}")
    print(f"총 거래 수: {len(trades)}, 총 순수익률: {total_return:+.2%}")

    by_size = trades.sort_values("net_return", ascending=False)
    for k in [1, 3, 5]:
        remaining = trades.drop(by_size.head(k).index)
        remaining_return = (1 + remaining["net_return"]).prod() - 1
        top_k_returns = by_size.head(k)["net_return"].tolist()
        print(f"  상위 {k}개 거래 제외 시 총수익률: {remaining_return:+.2%} "
              f"(낙폭 {total_return - remaining_return:+.2%}p) -- "
              f"{[f'{r:+.1%}' for r in top_k_returns]}")

    trades["year"] = trades["entry_date"].dt.year
    yearly_df = trades.groupby("year")["net_return"].apply(lambda r: (1 + r).prod() - 1).reset_index()
    yearly_df.columns = ["year", "yearly_return"]
    print("\n  연도별 복리수익률:")
    print(yearly_df.to_string(index=False))
    max_year_row = yearly_df.loc[yearly_df["yearly_return"].abs().idxmax()]
    print(f"\n  최대 기여 연도: {int(max_year_row['year'])} ({max_year_row['yearly_return']:+.2%})")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("사용법: python src/check_trade_concentration_pt_sl.py <pt> <sl> <threshold> <seed>")
        print("예:     python src/check_trade_concentration_pt_sl.py 3 1 0.50 42")
        sys.exit(1)

    pt, sl, threshold, seed = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    if pt == int(pt):
        pt = int(pt)
    if sl == int(sl):
        sl = int(sl)
    analyze(pt, sl, threshold, seed)