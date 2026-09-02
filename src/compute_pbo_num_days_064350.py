"""
PBO(Probability of Backtest Overfitting) -- 064350 BASE 전용, num_days 축.

[주의] num_days를 바꾸면 수직 배리어 시점뿐 아니라 배리어 폭 자체도 같이
바뀜 (target = daily_vol * sqrt(num_days), pt_pct/sl_pct = pt_sl * target).
즉 "더 오래/짧게 기다리는 같은 전략"이 아니라 "배리어 폭이 다른 별개 전략"임
-- 예전 118990 실험에서 num_days 20->10을 "표본 확대"로 착각했다가 실제로는
전략 자체가 바뀐 거였다는 걸 뒤늦게 깨달은 전례가 있음 (PROJECT_SUMMARY.md
"[재검증]" 섹션 참고). 이번엔 처음부터 "별개 전략 후보"로 취급하고 검증함.

pt_sl=(2,1), threshold=0.60은 고정하고 num_days만 스윕 -- 이 축 하나만
순수하게 검증하기 위함.

사용법 (레포 루트에서):
    python src/compute_pbo_num_days_064350.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKER_KRX = "064350"
TICKER = "064350.KS"
PT_SL = (2, 1)
THRESHOLD = 0.60
NUM_DAYS_CANDIDATES = [10, 15, 20, 25, 30]
SEED = 42
N_SUBPERIODS = 10
TRAIN_SIZE, TEST_SIZE, STEP = 300, 60, 60
ROUND_TRIP_COST = 0.002


def config_label(num_days: int) -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{num_days}_hl"


def load_or_build_dataset(num_days: int) -> pd.DataFrame:
    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label(num_days)}_base.csv"
    if out_path.exists():
        df = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"num_days={num_days} 데이터 생성 중...")
        df = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=num_days)
        df.to_csv(out_path)
        print(f"  저장 완료: {out_path.name} ({df.shape[0]}행)")
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


def generate_trades(df: pd.DataFrame, num_days: int) -> pd.DataFrame:
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, embargo=num_days)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=SEED,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= THRESHOLD:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": entry_date + pd.Timedelta(days=holding),
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1

    return pd.DataFrame(trades)


def build_performance_matrix() -> pd.DataFrame:
    datasets = {nd: load_or_build_dataset(nd) for nd in NUM_DAYS_CANDIDATES}
    start = max(df.index.min() for df in datasets.values())
    end = min(df.index.max() for df in datasets.values())
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)
    print(f"전체 비교 구간: {start.date()} ~ {end.date()}, {N_SUBPERIODS}개 구간\n")

    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=NUM_DAYS_CANDIDATES, dtype=float)

    for num_days in NUM_DAYS_CANDIDATES:
        print(f"num_days={num_days} 거래 생성 중...")
        trades = generate_trades(datasets[num_days], num_days)

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            if trades.empty:
                perf.loc[s, num_days] = 0.0
                continue
            in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
            perf.loc[s, num_days] = (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0

    return perf


def cscv_pbo(perf: pd.DataFrame) -> dict:
    n_periods = perf.shape[0]
    half = n_periods // 2
    all_periods = list(range(n_periods))

    logits, best_is_candidates = [], []
    for is_periods in combinations(all_periods, half):
        oos_periods = [p for p in all_periods if p not in is_periods]
        is_perf = perf.loc[list(is_periods)].mean()
        oos_perf = perf.loc[oos_periods].mean()

        best_candidate_is = is_perf.idxmax()
        best_is_candidates.append(best_candidate_is)
        oos_rank = oos_perf.rank(pct=True)[best_candidate_is]

        oos_rank_clipped = np.clip(oos_rank, 0.01, 0.99)
        logit = np.log(oos_rank_clipped / (1 - oos_rank_clipped))
        logits.append(logit)

    logits = np.array(logits)
    return {
        "n_combinations": len(logits),
        "pbo": (logits <= 0).mean(),
        "logit_mean": logits.mean(),
        "logit_median": np.median(logits),
        "best_is_candidates": pd.Series(best_is_candidates).value_counts(),
    }


if __name__ == "__main__":
    perf = build_performance_matrix()
    print(f"\n=== 구간별 num_days 후보 성과 (pt_sl={PT_SL}, threshold={THRESHOLD} 고정) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (num_days 축, {len(NUM_DAYS_CANDIDATES)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 num_days ===")
    print(result["best_is_candidates"].to_string())
    print("\n(채택값 20이 여기서 자주 1등이면 선택이 근거 있음. 다른 값이 압도적이면")
    print(" 그 값을 backtest_pt1sl1_064350.py 패턴처럼 직접 5-seed로 재검증할 것 --")
    print(" IS 선호가 실전 강건성으로 이어지는지는 항상 별도 확인 필요, pt_sl 축 때 확인된 원칙.)")

    print("\n판정 기준: PBO<20% 낮음 / 20~50% 중간 / >=50% 높음")