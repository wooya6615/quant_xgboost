"""
064350 BASE 전용, num_days=30 직접 검증.

num_days 축 PBO(compute_pbo_num_days_064350.py)에서 num_days=30이 IS
최고빈도 204/252(81%)로 압도적이었고, 이 축의 PBO 자체가 17.5%(낮음),
logit 평균 +2.920(강한 양수)로 나와서 -- pt_sl=(3,1) 때(PBO 86.9%, 함정)와
달리 이번엔 "IS 선호가 실제로 일반화될 가능성이 높은" 패턴. backtest_pt1sl1_
064350.py와 동일한 구조로 직접 5-seed + 자체 최대기여연도 국면검증까지 확인.

[주의] num_days가 바뀌면 배리어 폭(target = daily_vol * sqrt(num_days))도
같이 바뀌므로 이건 "더 오래 기다리는 같은 전략"이 아니라 "폭이 다른 별개
전략"임 -- 결과 해석 시 유의.

사용법 (레포 루트에서):
    python src/backtest_num_days30_064350.py

전제:
    compute_pbo_num_days_064350.py로 data/064350_features_triple_barrier_
    pt2sl1_nd30_hl_base.csv가 이미 생성돼 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKER_KRX = "064350"
TICKER = "064350.KS"
PT_SL = (2, 1)
NUM_DAYS = 30
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SEEDS = [42, 1, 7, 123, 2024]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def build_and_save_dataset() -> pd.DataFrame:
    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label()}_base.csv"
    if out_path.exists():
        print(f"이미 존재함, 재사용: {out_path.name}")
        df = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"num_days={NUM_DAYS} 데이터 생성 중...")
        df = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
        df.to_csv(out_path)
        print(f"저장 완료: {out_path.name} ({df.shape[0]}행)")

    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
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
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "net_return": gross_return - ROUND_TRIP_COST,
                        "holding_rows": holding,
                    })
                i += holding
            else:
                i += 1

    return pd.DataFrame(trades)


def buy_and_hold_return(df: pd.DataFrame) -> float:
    return df["Close"].iloc[-1] / df["Close"].iloc[0] - 1


def find_dominant_year(df: pd.DataFrame, threshold: float, seeds: list) -> int:
    year_votes = {}
    for seed in seeds:
        trades = generate_trades(df, threshold, seed)
        if trades.empty:
            continue
        trades["year"] = trades["entry_date"].dt.year
        yearly = trades.groupby("year")["net_return"].apply(lambda r: (1 + r).prod() - 1)
        top_year = yearly.abs().idxmax()
        year_votes[top_year] = year_votes.get(top_year, 0) + 1
    return max(year_votes, key=year_votes.get) if year_votes else None


def bh_return_excluding_year(df: pd.DataFrame, exclude_year: int) -> float:
    before = df[df.index.year < exclude_year]
    after = df[df.index.year > exclude_year]
    ret = 1.0
    if len(before) > 1:
        ret *= before["Close"].iloc[-1] / before["Close"].iloc[0]
    if len(after) > 1:
        ret *= after["Close"].iloc[-1] / after["Close"].iloc[0]
    return ret - 1


if __name__ == "__main__":
    df = build_and_save_dataset()
    print(f"\n라벨 분포: {df['label_tb_binary'].value_counts(normalize=True).round(4).to_dict()}")
    bh_full = buy_and_hold_return(df)

    print(f"\n{'=' * 60}\n=== 전체기간 threshold 스윕 (num_days={NUM_DAYS}) ===\n{'=' * 60}")
    full_rows = []
    for threshold in THRESHOLDS:
        for seed in SEEDS:
            trades = generate_trades(df, threshold, seed)
            net_total = (1 + trades["net_return"]).prod() - 1 if not trades.empty else np.nan
            full_rows.append({
                "threshold": threshold, "seed": seed, "n_trades": len(trades),
                "net_total_return": net_total, "bh_return": bh_full,
                "beats_bh": (net_total > bh_full) if pd.notna(net_total) else False,
            })
    full_df = pd.DataFrame(full_rows)
    print(full_df.round(4).to_string(index=False))

    summary = full_df.groupby("threshold").agg(
        win_seeds=("beats_bh", "sum"), mean_net_return=("net_total_return", "mean"),
        mean_n_trades=("n_trades", "mean"),
    ).reset_index()
    print(f"\n=== threshold별 5-seed 요약 ===")
    print(summary.round(4).to_string(index=False))

    passing = summary[summary["win_seeds"] == 5]
    if passing.empty:
        print(f"\n5/5 통과 threshold 없음 -- num_days={NUM_DAYS}은 이 상태로는 num_days=20보다 못함")
    else:
        best_threshold = passing.sort_values("mean_net_return", ascending=False).iloc[0]["threshold"]
        print(f"\n5/5 통과 threshold 중 최고: {best_threshold} -- 국면검증 진행")

        dominant_year = find_dominant_year(df, best_threshold, SEEDS)
        print(f"최대 기여 연도: {dominant_year}")

        bh_excl = bh_return_excluding_year(df, dominant_year)
        excl_rows = []
        for seed in SEEDS:
            trades = generate_trades(df, best_threshold, seed)
            trades["year"] = trades["entry_date"].dt.year
            remaining = trades[trades["year"] != dominant_year]
            net_total = (1 + remaining["net_return"]).prod() - 1 if not remaining.empty else np.nan
            excl_rows.append({
                "seed": seed, "net_total_return_excl": net_total,
                "bh_return_excl": bh_excl, "beats_bh": (net_total > bh_excl) if pd.notna(net_total) else False,
            })
        excl_df = pd.DataFrame(excl_rows)
        print(f"\n=== {dominant_year}년 제외 재검증 (threshold={best_threshold}) ===")
        print(excl_df.round(4).to_string(index=False))
        print(f"\nBuy&Hold({dominant_year}년 제외): {bh_excl:+.2%} "
              f"{'(벤치마크 붕괴 -- 절대수익률 부호를 볼 것)' if bh_excl < -0.3 else ''}")

        # 거래 집중도까지 바로 확인 (118990/pt_sl=(3,1) 때와 동일 원칙)
        print(f"\n=== 거래 집중도 (threshold={best_threshold}, seed=42/1) ===")
        for seed in [42, 1]:
            trades = generate_trades(df, best_threshold, seed).sort_values("entry_date")
            total = (1 + trades["net_return"]).prod() - 1
            by_size = trades.sort_values("net_return", ascending=False)
            top5_returns = by_size.head(5)["net_return"].tolist()
            remaining5 = trades.drop(by_size.head(5).index)
            remaining5_return = (1 + remaining5["net_return"]).prod() - 1
            print(f"  seed={seed}: 총 {len(trades)}건, 총수익 {total:+.2%}, "
                  f"상위5개 제외시 {remaining5_return:+.2%} -- {[f'{r:+.1%}' for r in top5_returns]}")

    print(f"\n{'=' * 60}")
    print("[비교 가이드]")
    print("- num_days=30이 5/5 통과 + 국면제외 견고 + 집중도가 num_days=20과 비슷한 수준이면:")
    print("  새 채택 후보로 진지하게 고려 -- 단 threshold PBO는 num_days=30 기준으로 다시")
    print("  돌려야 함 (pt_sl=(3,1) 때 threshold 하나만 보고 성급히 판단했다가 뒤집힌 전례)")