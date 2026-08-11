"""
하이브리드 백테스트: 종가 라벨로 학습(모델이 뭘 배울지 고정) + 선택된 신호의 실제
손익은 시가체결 데이터에서 조회.

배경: compare_close_vs_open_trades.py에서 종가체결/시가체결이 서로 다른 76건을
골랐다는 걸 확인함 -- 이는 진입가 정의가 X(feature)로는 안 들어가지만, y(라벨)를
통해 모델이 배우는 패턴 자체를 바꿔버리기 때문. "어떤 날을 신호로 볼지"와
"그 신호의 실제 손익이 얼마인지"를 분리해서, 학습 목표는 하나로 고정하고 체결
가정만 바꿔서 비교하면 더 깨끗한 테스트가 됨.

X(feature)는 두 데이터셋 간에 완전히 동일함 (진입가 정의는 라벨/수익률에만 영향).

사용법 (레포 루트에서):
    python src/backtest_train_close_execute_open.py <ticker_krx>
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
SEEDS = [42, 1, 7, 123, 2024]


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


def generate_hybrid_trades(df_train: pd.DataFrame, df_execute: pd.DataFrame,
                            threshold: float = THRESHOLD, train_size: int = 300,
                            test_size: int = 60, step: int = 60, embargo: int = NUM_DAYS,
                            random_state: int = 42) -> pd.DataFrame:
    """
    df_train: 학습 목표(y=label_tb_binary)를 제공 -- "모델이 뭘 배울지"를 결정
    df_execute: 선택된 신호의 실제 gross_return/holding_rows를 제공 -- "실제 체결 결과"
    X(feature)는 df_train 기준으로 뽑음 (두 데이터셋 간 동일하므로 어느 쪽이든 무관).
    """
    if not (df_train.index.equals(df_execute.index)):
        common_idx = df_train.index.intersection(df_execute.index)
        print(f"경고: 두 데이터셋 인덱스가 완전히 일치하지 않음 -- 공통 구간 {len(common_idx)}행으로 정렬")
        df_train = df_train.loc[common_idx]
        df_execute = df_execute.loc[common_idx]

    X = df_train[FEATURE_COLS_COMBINED]
    y = df_train["label_tb_binary"]
    splits = walk_forward_splits(len(df_train), train_size, test_size, step, embargo)

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
                entry_date = df_execute.index[row_idx]
                # 실제 손익/보유기간은 df_execute(시가체결)에서 조회 -- 핵심
                gross_return = df_execute["ret_tb"].iloc[row_idx]
                holding = int(df_execute["holding_rows_tb"].iloc[row_idx])

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df_execute) - 1)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": df_execute.index[exit_row],
                        "gross_return": gross_return,
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


def get_fixed_bh_window(df: pd.DataFrame, train_size: int = 300, test_size: int = 60,
                         step: int = 60, embargo: int = NUM_DAYS) -> tuple:
    """
    Buy&Hold 비교 기준을 "이번에 나온 거래들의 기간"이 아니라 walk-forward 전체
    테스트 구간(첫 fold 시작 ~ 마지막 fold 끝)으로 고정. 그래야 어떤 모델이 어떤
    날짜를 신호로 골랐는지와 무관하게, 세 버전(순수종가/순수시가/하이브리드)이
    전부 동일한 기간에 대해 Buy&Hold와 비교되어 공정함.
    """
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return df.index[first_test_idx], df.index[min(last_test_idx, len(df) - 1)]


def summarize(trades: pd.DataFrame, df: pd.DataFrame, label: str,
              bh_window: tuple | None = None) -> dict:
    if trades.empty:
        print(f"[{label}] 거래 없음")
        return {"label": label, "n_trades": 0}
    total_net = (1 + trades["net_return"]).prod() - 1

    if bh_window is not None:
        bh_start, bh_end = bh_window
    else:
        bh_start, bh_end = trades["entry_date"].min(), trades["exit_date"].max()
    bh = df.loc[bh_end, "Close"] / df.loc[bh_start, "Close"] - 1

    print(f"[{label}] 거래 수: {len(trades)}, net: {total_net:.1%}, "
          f"Buy&Hold(고정구간 {bh_start.date()}~{bh_end.date()}): {bh:.1%}, "
          f"{'이김' if total_net > bh else '못 이김'}")
    return {"label": label, "n_trades": len(trades), "net": total_net, "bh": bh}


if __name__ == "__main__":
    ticker_krx = sys.argv[1] if len(sys.argv) > 1 else "118990"

    df_close = load_dataset(ticker_krx, open_entry=False)
    df_open = load_dataset(ticker_krx, open_entry=True)

    # 세 버전 전부 동일한 Buy&Hold 비교 구간 사용 (df_close 기준으로 고정 -- df_open과
    # 길이가 거의 같으므로 어느 쪽 기준이든 결과는 사실상 동일)
    bh_window = get_fixed_bh_window(df_close)
    print(f"고정 Buy&Hold 비교 구간: {bh_window[0].date()} ~ {bh_window[1].date()}\n")

    print(f"=== {ticker_krx}: 종가라벨 학습 + 시가체결 실현손익 (하이브리드) 5-seed ===\n")
    results = []
    for seed in SEEDS:
        trades = generate_hybrid_trades(df_close, df_open, random_state=seed)
        r = summarize(trades, df_open, f"seed={seed}", bh_window=bh_window)
        results.append(r)

    win_count = sum(1 for r in results if r.get("net") is not None and r["net"] > r["bh"])
    print(f"\n5개 시드 중 Buy & Hold를 이긴 시드: {win_count}/5")

    print(f"\n(참고) 순수 종가체결(학습도 체결도 종가) 5-seed 재확인:")
    for seed in SEEDS:
        trades_pure_close = generate_hybrid_trades(df_close, df_close, random_state=seed)
        summarize(trades_pure_close, df_close, f"seed={seed} (순수 종가체결)", bh_window=bh_window)

    print(f"\n(참고) 순수 시가체결(학습도 체결도 시가) 5-seed 재확인:")
    for seed in SEEDS:
        trades_pure_open = generate_hybrid_trades(df_open, df_open, random_state=seed)
        summarize(trades_pure_open, df_open, f"seed={seed} (순수 시가체결)", bh_window=bh_window)