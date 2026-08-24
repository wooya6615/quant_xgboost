"""
새로 스크리닝한 10종목(상위20일 수익률 집중도 중앙값 이하)이 기존 triple-barrier
BASE+VALUATION 전략(pt_sl=2:1, num_days=20, threshold=0.65)을 개별 5-seed로
통과하는지 검증. run_universe_triple_barrier.py(조선/방산 4종목 검증 때 썼던
스크립트)와 동일한 구조 재사용.

⚠️ 종목/전략 설정 둘 다 사전등록값 그대로: screen_universe.py에서 나온 순서를
그대로 쓰고(백테스트 성과로 재정렬 안 함), pt_sl/threshold/num_days도 기존
검증값(064350/052690/118990) 그대로 유지.

⚠️ 009155(삼성전기우, 우선주)는 자산 성격이 달라 스크리닝 결과에서 제외.
082740(한화엔진, 조선업 밸류체인)은 포함 여부 미정 -- 기본 포함, 필요시
CANDIDATES에서 지우고 다음 순번으로 교체할 것.

사용법 (레포 루트에서):
    python src/run_screened_universe_triple_barrier.py

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
CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"
THRESHOLD = 0.65
ROUND_TRIP_COST = 0.002
SEEDS = [42, 1, 7, 123, 2024]

# ticker_krx: (종목명, yfinance suffix)
CANDIDATES = {
    "064760": ("티씨케이", ".KQ"),
    "035900": ("JYP Ent.", ".KQ"),
    "000990": ("DB하이텍", ".KS"),
    "082740": ("한화엔진", ".KS"),   # ⚠️ 조선업 밸류체인 -- 포함 여부 미정
    "058470": ("리노공업", ".KQ"),
    "083450": ("GST", ".KQ"),
    "131290": ("티에스이", ".KQ"),
    "089030": ("테크윙", ".KQ"),
    "005290": ("동진쎄미켐", ".KQ"),
    "014680": ("한솔케미칼", ".KS"),
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
    """기존 3종목 검증 때(backtest_triple_barrier.py)와 완전히 동일한 로직."""
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


def get_or_build_dataset(ticker_krx: str, name: str, suffix: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
    if not path.exists():
        print(f"[{name}] 데이터셋 없음 -- 생성 중 (pykrx/yfinance 호출)...")
        dataset = build_triple_barrier_dataset_combined(
            ticker=f"{ticker_krx}{suffix}", ticker_krx=ticker_krx, pt_sl=PT_SL, num_days=NUM_DAYS,
        )
        dataset.to_csv(path)
        print(f"[{name}] 저장 완료: {path} ({dataset.shape[0]}행)")
    else:
        print(f"[{name}] 기존 데이터셋 재사용: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)
    return df


def run_ticker(ticker_krx: str, name: str, suffix: str) -> dict:
    df = get_or_build_dataset(ticker_krx, name, suffix)

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
    for ticker_krx, (name, suffix) in CANDIDATES.items():
        print(f"\n{'=' * 60}\n=== {name} ({ticker_krx}) ===\n{'=' * 60}")
        r = run_ticker(ticker_krx, name, suffix)
        results.append(r)
        for seed, n_trades, net, bh in zip(SEEDS, r["n_trades"], r["seed_nets"], r["seed_bhs"]):
            if pd.isna(bh):
                print(f"  seed={seed}: 거래 없음")
                continue
            outcome = "이김" if net > bh else "못 이김"
            print(f"  seed={seed}: 거래수={n_trades}, net={net:.1%}, Buy&Hold={bh:.1%}, {outcome}")
        print(f"  -> {name} 종합: {r['win_count']}/5")

    print("\n\n" + "=" * 60)
    print("=== 10종목 종합 요약 ===")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {"종목": r["name"], "코드": r["ticker"], "판정": f"{r['win_count']}/5"}
        for r in results
    ])
    print(summary_df.to_string(index=False))

    passed = [r["ticker"] for r in results if r["win_count"] >= 4]
    print(f"\n4/5 이상 통과 종목: {passed if passed else '없음'}")
    print("통과한 종목만 섹터PER ablation/풀링 재검증으로 넘어갈 것.")