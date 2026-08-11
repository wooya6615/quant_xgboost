"""
2024~25 조선/방산/전력기기 슈퍼사이클을 주도했던 상위 종목군(quant_ranking_kr/
quant_sector_rotation에서 확인) 중 4종목을 triple-barrier COMBINED 전략으로 일괄 검증.

한화에어로스페이스(012450), HD현대일렉트릭(267260), LS ELECTRIC(010120), 효성중공업(298040)

주의: 이 네 종목이 "저유동성 대기업 계열"이라는 프로필이 확인된 건 아님 (LS ELECTRIC은
시총 30조대 초대형주). 완전히 독립적인 프로필 재현성 검증이라기보다는 "같은 2024~25
테마 안에서 이 전략이 얼마나 넓게 통하는지"를 보는 탐색적 성격에 가까움.

설정은 현대로템/한전기술 검증 때와 완전히 동일하게 고정 (전략을 바꾸지 않는 원칙 유지):
    pt_sl=(2,1), num_days=20, BASE+VALUATION COMBINED, threshold=0.65

사용법 (레포 루트에서):
    python src/run_universe_triple_barrier.py

전제:
    feature_engineering_triple_barrier.py, feature_engineering_valuation.py,
    labeling_triple_barrier.py가 같은 폴더(src/)에 있어야 함. 데이터셋이 없으면
    이 스크립트가 자동으로 생성함 (pykrx/yfinance 호출 있음 -- 시간 좀 걸릴 수 있음).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import build_triple_barrier_dataset_combined

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

PT_SL = (2, 1)
NUM_DAYS = 20
CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}"
THRESHOLD = 0.65
ROUND_TRIP_COST = 0.002
SEEDS = [42, 1, 7, 123, 2024]

CANDIDATES = {
    "012450": "한화에어로스페이스",
    "267260": "HD현대일렉트릭",
    "010120": "LS ELECTRIC",
    "298040": "효성중공업",
}


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
    """현대로템/한전기술 검증 때(backtest_triple_barrier.py)와 완전히 동일한 로직."""
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


def get_or_build_dataset(ticker_krx: str, name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
    if not path.exists():
        print(f"[{name}] 데이터셋 없음 -- 생성 중 (pykrx/yfinance 호출)...")
        dataset = build_triple_barrier_dataset_combined(
            ticker=f"{ticker_krx}.KS", ticker_krx=ticker_krx, pt_sl=PT_SL, num_days=NUM_DAYS,
        )
        dataset.to_csv(path)
        print(f"[{name}] 저장 완료: {path} ({dataset.shape[0]}행)")
    else:
        print(f"[{name}] 기존 데이터셋 재사용: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


def run_ticker(ticker_krx: str, name: str) -> dict:
    df = get_or_build_dataset(ticker_krx, name)

    seed_nets, seed_bhs, n_trades_list = [], [], []
    for seed in SEEDS:
        trades = generate_trades(df, random_state=seed)
        if trades.empty:
            seed_nets.append(np.nan)
            seed_bhs.append(np.nan)
            n_trades_list.append(0)
            continue
        net = (1 + trades["net_return"]).prod() - 1
        test_start, test_end = trades["entry_date"].min(), trades["exit_date"].max()
        bh = df.loc[test_end, "Close"] / df.loc[test_start, "Close"] - 1
        seed_nets.append(net)
        seed_bhs.append(bh)
        n_trades_list.append(len(trades))

    win_count = sum(
        1 for n, b in zip(seed_nets, seed_bhs)
        if pd.notna(n) and pd.notna(b) and n > b
    )
    return {
        "ticker": ticker_krx, "name": name,
        "n_trades": n_trades_list, "seed_nets": seed_nets, "seed_bhs": seed_bhs,
        "win_count": win_count,
    }


if __name__ == "__main__":
    results = []
    for ticker_krx, name in CANDIDATES.items():
        print(f"\n{'=' * 60}\n=== {name} ({ticker_krx}) ===\n{'=' * 60}")
        r = run_ticker(ticker_krx, name)
        results.append(r)
        for seed, n_trades, net, bh in zip(SEEDS, r["n_trades"], r["seed_nets"], r["seed_bhs"]):
            if pd.isna(bh):
                print(f"  seed={seed}: 거래 없음")
                continue
            outcome = "이김" if net > bh else "못 이김"
            print(f"  seed={seed}: 거래수={n_trades}, net={net:.1%}, Buy&Hold={bh:.1%}, {outcome}")
        print(f"  -> {name} 종합: {r['win_count']}/5")

    print("\n\n" + "=" * 60)
    print("=== 4종목 종합 요약 ===")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {"종목": r["name"], "코드": r["ticker"], "판정": f"{r['win_count']}/5"}
        for r in results
    ])
    print(summary_df.to_string(index=False))

    passed = [r["ticker"] for r in results if r["win_count"] >= 4]
    print(f"\n4/5 이상 통과 종목: {passed if passed else '없음'}")
    print("통과한 종목은 backtest_triple_barrier_pooled.py의 TICKERS 리스트에 추가해서")
    print("현대로템/한전기술과 함께 풀링 재검증할 것. (아직 '2025년 이전 절단' 재검증은")
    print("안 했으니, 여기서 통과한 종목만 그 단계까지 마저 거칠 것)")