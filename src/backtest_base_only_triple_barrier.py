"""
[재판정] BASE 전용 triple-barrier 재검증 -- vs_base_rate 대신 실제 백테스트로.

diagnose_base_rate_drift.py에서 확인됨: test_size=60(약 3개월) walk-forward
fold의 test_base_rate가 0.03~0.98까지 흔들림 (그 분기 추세 방향에 라벨 비율이
좌우됨). 이 상태에서 "다수 클래스 비율(base_rate)을 이기는가"는 정상적으로
보정된 확률 모델에게 불공정한 기준 -- AUC(0.52~0.55, 시드 간 매우 안정적)는
살아있는데 vs_base_rate만 크게 마이너스로 나온 이유가 이것으로 설명됨.

그래서 vs_base_rate 게이트는 버리고, 이 프로젝트의 최종 기준인 "거래비용 반영
후 Buy & Hold를 이기는가"로 바로 판정함. AUC가 좋아도 여기서 항상 무너졌던
과거 패턴(README.md 방법론 요약 참고)을 감안해 기대치는 낮게 잡을 것.

전제:
    verify_base_only_triple_barrier.py로 data/{ticker}_features_triple_barrier_
    pt2sl1_nd20_hl_base.csv가 이미 생성돼 있어야 함.

사용법 (레포 루트에서):
    python src/backtest_base_only_triple_barrier.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = ["064350", "052690", "118990"]
SEEDS = [42, 1, 7, 123, 2024]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20
ROUND_TRIP_COST = 0.002
EXCLUDE_YEAR = 2025


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


def generate_trades(df: pd.DataFrame, threshold: float, random_state: int) -> pd.DataFrame:
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
                holding = max(holding, 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "proba": proba[i],
                        "gross_return": gross_return,
                        "net_return": gross_return - ROUND_TRIP_COST,
                        "holding_rows": holding,
                    })
                i += holding  # 보유 기간만큼 겹침 방지
            else:
                i += 1

    return pd.DataFrame(trades)


def buy_and_hold_return(df: pd.DataFrame) -> float:
    """같은 기간 Buy & Hold 수익률 (단순 첫날 대비 마지막날 종가)."""
    return df["Close"].iloc[-1] / df["Close"].iloc[0] - 1


def evaluate_ticker(ticker_krx: str, cutoff_year: int | None = None) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)

    if cutoff_year is not None:
        df = df[df.index.year < cutoff_year]

    bh_return = buy_and_hold_return(df)

    rows = []
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            trades = generate_trades(df, threshold, seed)
            if trades.empty:
                rows.append({
                    "threshold": threshold, "seed": seed, "n_trades": 0,
                    "net_total_return": np.nan, "beats_bh": False,
                })
                continue

            # 순차 복리 (겹치지 않는 거래들이므로 단순 곱)
            net_total = (1 + trades["net_return"]).prod() - 1
            rows.append({
                "threshold": threshold,
                "seed": seed,
                "n_trades": len(trades),
                "win_rate": (trades["net_return"] > 0).mean(),
                "net_total_return": net_total,
                "bh_return": bh_return,
                "beats_bh": net_total > bh_return,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_full, all_excl = [], []

    for ticker_krx in TICKERS:
        print(f"\n{'#' * 60}\n# {ticker_krx} -- 전체기간\n{'#' * 60}")
        full = evaluate_ticker(ticker_krx)
        full.insert(0, "ticker_krx", ticker_krx)
        print(full.round(4).to_string(index=False))
        all_full.append(full)

        print(f"\n# {ticker_krx} -- {EXCLUDE_YEAR}년 제외")
        excl = evaluate_ticker(ticker_krx, cutoff_year=EXCLUDE_YEAR)
        excl.insert(0, "ticker_krx", ticker_krx)
        print(excl.round(4).to_string(index=False))
        all_excl.append(excl)

    full_df = pd.concat(all_full, ignore_index=True)
    excl_df = pd.concat(all_excl, ignore_index=True)

    full_df.to_csv(DATA_DIR / "base_only_backtest_full.csv", index=False)
    excl_df.to_csv(DATA_DIR / "base_only_backtest_excl2025.csv", index=False)

    print(f"\n{'=' * 60}\n=== threshold별 5-seed 요약 (전체기간) ===\n{'=' * 60}")
    summary = full_df.groupby(["ticker_krx", "threshold"]).agg(
        win_seeds=("beats_bh", "sum"), n_seeds=("beats_bh", "count"),
        mean_net_return=("net_total_return", "mean"), mean_n_trades=("n_trades", "mean"),
    ).reset_index()
    print(summary.round(4).to_string(index=False))

    print("\n[판정 가이드]")
    print("- threshold 하나라도 5/5 seed로 B&H를 이기고, 국면제외(2025년 제외)에서도")
    print("  같은 threshold가 방향을 유지하면: 그 threshold로 다음 단계(PBO 검증)로 진행")
    print("- 전부 5/5 미달이면: BASE 전용으로는 여전히 [실패] -- 이번엔 metric 버그가")
    print("  아니라 진짜 실전 손익에서 확인된 것이므로 확정 지어도 됨")
    print("- threshold를 3개 스윕했으므로, 하나를 최종 채택하기 전에 반드시 pt_sl 때처럼")
    print("  threshold 축 PBO를 돌려서 '탐색해서 고른 것' 리스크를 확인할 것")