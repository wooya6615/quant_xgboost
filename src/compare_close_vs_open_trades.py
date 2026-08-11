"""
종가체결(pt2sl1_nd20) vs 시가체결(pt2sl1_nd20_open) 실제 거래내역 직접 비교.

같은 threshold/seed로 두 버전의 거래를 각각 생성한 뒤:
1. 공통 진입일(같은 날 둘 다 진입한 거래)의 성패가 어떻게 갈렸는지
2. diagnose_open_close_adjustment.py에서 찾은 극단치 갭 날짜가 어느 거래의
   보유기간(entry_date~exit_date) 안에 들어가는지
를 확인해서, "소표본이 극단치 몇 건에 좌우된다"는 가설을 직접 검증.

사용법 (레포 루트에서):
    python src/compare_close_vs_open_trades.py <ticker_krx> [extreme_dates...]
    예: python src/compare_close_vs_open_trades.py 118990 2020-03-13 2022-06-02 2022-06-10 2022-06-14 2022-06-23
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

ROUND_TRIP_COST = 0.002
NUM_DAYS = 20
THRESHOLD = 0.65
SEED = 42


def load_dataset(ticker_krx: str, open_entry: bool) -> pd.DataFrame:
    config = "pt2sl1_nd20_open" if open_entry else "pt2sl1_nd20"
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{config}_valuation.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


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


def generate_trades(df, threshold=THRESHOLD, train_size=300, test_size=60, step=60,
                     embargo=NUM_DAYS, random_state=SEED):
    X = df[FEATURE_COLS_COMBINED]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state,
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
                holding = int(df["holding_rows_tb"].iloc[row_idx])

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": df.index[exit_row],
                        "gross_return": gross_return,
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


if __name__ == "__main__":
    ticker_krx = sys.argv[1] if len(sys.argv) > 1 else "118990"
    extreme_dates = [pd.Timestamp(d) for d in sys.argv[2:]] if len(sys.argv) > 2 else []

    print(f"=== {ticker_krx}: 종가체결 vs 시가체결 거래내역 비교 (threshold={THRESHOLD}, seed={SEED}) ===\n")

    df_close = load_dataset(ticker_krx, open_entry=False)
    df_open = load_dataset(ticker_krx, open_entry=True)

    trades_close = generate_trades(df_close)
    trades_open = generate_trades(df_open)

    print(f"종가체결 거래 수: {len(trades_close)}, net={((1+trades_close['net_return']).prod()-1):.1%}")
    print(f"시가체결 거래 수: {len(trades_open)}, net={((1+trades_open['net_return']).prod()-1):.1%}")

    # 공통 진입일 찾기
    common_dates = set(trades_close["entry_date"]) & set(trades_open["entry_date"])
    print(f"\n공통 진입일: {len(common_dates)}건 "
          f"(종가체결만: {len(set(trades_close['entry_date'])-common_dates)}건, "
          f"시가체결만: {len(set(trades_open['entry_date'])-common_dates)}건)")

    print("\n=== 공통 진입일 거래의 net_return 비교 ===")
    tc = trades_close.set_index("entry_date").loc[list(common_dates)].sort_index()
    to = trades_open.set_index("entry_date").loc[list(common_dates)].sort_index()
    comparison = pd.DataFrame({
        "종가체결_net": tc["net_return"],
        "시가체결_net": to["net_return"],
        "차이": to["net_return"] - tc["net_return"],
    }).sort_values("차이")
    print(comparison.round(4).to_string())

    print(f"\n차이 평균: {comparison['차이'].mean():+.4f}, 차이 표준편차: {comparison['차이'].std():.4f}")

    # 극단치 날짜가 어느 거래의 보유기간에 걸리는지 확인
    if extreme_dates:
        print(f"\n=== 극단치 갭 날짜가 걸리는 거래 확인 ===")
        for ed in extreme_dates:
            print(f"\n갭 날짜: {ed.date()}")
            for label, trades in [("종가체결", trades_close), ("시가체결", trades_open)]:
                hits = trades[(trades["entry_date"] <= ed) & (trades["exit_date"] >= ed)]
                if not hits.empty:
                    print(f"  [{label}] 이 날짜를 보유기간에 포함하는 거래 {len(hits)}건:")
                    print(hits[["entry_date", "exit_date", "net_return"]].to_string(index=False))
                else:
                    print(f"  [{label}] 이 날짜를 포함하는 거래 없음")