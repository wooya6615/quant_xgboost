"""
현대로템 + 한전기술 + 모트렉스 풀링 백테스트, BASE+VALUATION+MARKET_VAL COMBINED.
(backtest_triple_barrier_pooled.py를 MARKET_VAL 포함 버전으로 리네이밍 -- 나머지 로직 100% 동일)

⚠️ pt_sl=(2,1), num_days=20, threshold=0.65, TICKERS(3종목 전체) 전부 사전고정값 그대로
유지. 시장 PER 피처만 추가해서, 이게 pt_sl 재검증 실패(3/5)를 살리는 데 도움이 되는지
확인하는 목적 -- 종목 조합이나 threshold는 절대 건드리지 않음.

사용법 (레포 루트에서):
    python src/backtest_triple_barrier_market_pooled.py

전제:
    3종목 다 feature_engineering_triple_barrier_market.py로
    {ticker}_features_triple_barrier_pt2sl1_nd20_hl_valuation_market.csv가 생성돼 있어야 함.
"""

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
FEATURE_COLS_MARKET_VAL = [
    "market_per", "market_pbr",
    "market_per_zscore_252d", "market_pbr_zscore_252d",
]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION + FEATURE_COLS_MARKET_VAL

ROUND_TRIP_COST = 0.002
NUM_DAYS = 20               # 사전등록값 -- 절대 안 바꿈
CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKERS = ["064350", "052690", "118990"]  # 3종목 전체 -- 사전등록값, 절대 안 바꿈
THRESHOLD = 0.65            # 사전등록값 -- 절대 안 바꿈
SEEDS = [42, 1, 7, 123, 2024]
WEIGHT = 1 / len(TICKERS)


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation_market.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
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


def generate_trades(df: pd.DataFrame, threshold: float = THRESHOLD, train_size: int = 300,
                     test_size: int = 60, step: int = 60, embargo: int = NUM_DAYS,
                     random_state: int = 42) -> pd.DataFrame:
    """기존 backtest_triple_barrier.py와 완전히 동일한 로직."""
    X = df[FEATURE_COLS_COMBINED]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

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

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


def get_fixed_bh_window(df: pd.DataFrame, train_size: int = 300, test_size: int = 60,
                         step: int = 60, embargo: int = NUM_DAYS):
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return df.index[first_test_idx], df.index[min(last_test_idx, len(df) - 1)]


def run_single_ticker(ticker: str, random_state: int = 42) -> dict:
    df = load_dataset(ticker)
    trades = generate_trades(df, random_state=random_state)
    bh_window = get_fixed_bh_window(df)
    if trades.empty:
        return {"ticker": ticker, "n_trades": 0, "net": 0.0, "bh": None, "trades": trades}

    net = (1 + trades["net_return"]).prod() - 1
    bh = df.loc[bh_window[1], "Close"] / df.loc[bh_window[0], "Close"] - 1
    return {"ticker": ticker, "n_trades": len(trades), "net": net, "bh": bh, "trades": trades}


if __name__ == "__main__":
    print("=== 종목별 개별 결과 (seed=42, 참고용) ===")
    per_ticker = {}
    for ticker in TICKERS:
        r = run_single_ticker(ticker, random_state=42)
        per_ticker[ticker] = r
        print(f"{ticker}: 거래 수 {r['n_trades']}, net {r['net']:.1%}, Buy&Hold {r['bh']:.1%}")

    all_trades = pd.concat([per_ticker[t]["trades"].assign(ticker=t) for t in TICKERS], ignore_index=True)
    total_n_trades = len(all_trades)
    win_rate = (all_trades["net_return"] > 0).mean()

    blended_net = sum(WEIGHT * per_ticker[t]["net"] for t in TICKERS)
    blended_bh = sum(WEIGHT * per_ticker[t]["bh"] for t in TICKERS)

    print(f"\n=== 풀링 결과 (동일가중 블렌딩, seed=42) ===")
    ticker_counts = ", ".join(f"{t}={per_ticker[t]['n_trades']}" for t in TICKERS)
    print(f"총 거래 수: {total_n_trades} ({ticker_counts})")
    print(f"전체 거래 기준 승률: {win_rate:.1%}")
    print(f"블렌딩 포트폴리오 net: {blended_net:.1%}")
    print(f"블렌딩 Buy & Hold: {blended_bh:.1%}")
    print(f"블렌딩 기준: {'이김' if blended_net > blended_bh else '못 이김'}")

    print("\n\n" + "=" * 60)
    print("=== 5-seed 블렌딩 재검증 ===")
    print("=" * 60)
    seed_results = []
    for seed in SEEDS:
        seed_data = {t: run_single_ticker(t, random_state=seed) for t in TICKERS}
        n_trades_total = sum(seed_data[t]["n_trades"] for t in TICKERS)
        blended_net_seed = sum(WEIGHT * seed_data[t]["net"] for t in TICKERS)
        blended_bh_seed = sum(WEIGHT * seed_data[t]["bh"] for t in TICKERS)
        seed_results.append({
            "seed": seed, "n_trades": n_trades_total,
            "blended_net": blended_net_seed, "blended_bh": blended_bh_seed,
        })
        print(f"seed={seed}: 총 거래수={n_trades_total}, "
              f"블렌딩 net={blended_net_seed:.1%}, 블렌딩 Buy&Hold={blended_bh_seed:.1%}, "
              f"{'이김' if blended_net_seed > blended_bh_seed else '못 이김'}")

    seed_df = pd.DataFrame(seed_results)
    print("\n--- 5-seed 요약 ---")
    print(seed_df.round(4).to_string(index=False))
    win_count = (seed_df["blended_net"] > seed_df["blended_bh"]).sum()
    print(f"\n블렌딩 포트폴리오 기준 5개 시드 중 Buy & Hold를 이긴 시드: {win_count}/5")
    print("비교 기준: pt_sl 재검증(MARKET_VAL 없이)은 3/5로 실패했음 -- 여기서 4~5/5가")
    print("나오면 시장PER 추가가 실제로 도움이 된 것. 여전히 3/5 이하면 시장PER도")
    print("이 블렌딩을 못 살린 것으로 봐야 함.")