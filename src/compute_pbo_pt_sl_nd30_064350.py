"""
PBO(Probability of Backtest Overfitting) -- 064350 BASE 전용, num_days=30 고정,
pt_sl 축.

[배경] compute_pbo_pt_sl_064350.py는 num_days=20 기준으로 pt_sl 축을 검증했음
(PBO 39.3%/중간, IS 1위는 (3,1) 63%였지만 자체 threshold PBO 86.9%로 함정임이
드러나 최종적으로 (2,1) 유지). 그 직후 별도 축(num_days)을 검증하면서
num_days=20 -> 30으로 교체됐는데, 이 교체 이후 pt_sl 축을 다시 확인한 적이
없음 -- build_production_model_064350_base.py FINAL_VERDICT에 "pt_sl 축은
num_days=20 기준으로만 검증, nd=30 기준으로는 재확인 안 됨"으로 남아있던
한계를 여기서 좁힘.

threshold는 0.60(현재 채택값), num_days도 30(현재 채택값)으로 둘 다 고정하고
pt_sl만 스윕 -- 이 축 하나만 순수하게 검증하기 위함.

pt_sl 후보: (1,1), (1.5,1), (2,1)[현재 채택], (2.5,1), (3,1) -- 손절(sl)=1
고정, 익절(pt)만 넓혀가는 기존 스윕 컨벤션 그대로.

사용법 (레포 루트에서):
    python src/compute_pbo_pt_sl_nd30_064350.py

전제:
    feature_engineering_triple_barrier.py와 같은 폴더(src/)에 있어야 함.
    pt_sl별 nd=30 데이터셋은 없으면 이 스크립트가 알아서 생성함 -- (2,1)은
    backtest_num_days30_064350.py에서 이미 만들어져 있을 것이고, 나머지 4개
    ((1,1)/(1.5,1)/(2.5,1)/(3,1))는 이번에 새로 생성됨 (yfinance/pykrx 네트워크
    호출 필요, 로컬에서 실행할 것).
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
NUM_DAYS = 30  # 현재 채택값 (기존 pt_sl 축 검증은 20 기준이었음)
THRESHOLD = 0.60  # 현재 채택값으로 고정 -- pt_sl 축만 순수하게 검증
SEED = 42
N_SUBPERIODS = 10
PT_SL_CANDIDATES = [(1, 1), (1.5, 1), (2, 1), (2.5, 1), (3, 1)]

TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002


def config_label(pt_sl: tuple) -> str:
    return f"pt{pt_sl[0]}sl{pt_sl[1]}_nd{NUM_DAYS}_hl"


def load_or_build_dataset(pt_sl: tuple) -> pd.DataFrame:
    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label(pt_sl)}_base.csv"
    if out_path.exists():
        df = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"pt_sl={pt_sl} (num_days={NUM_DAYS}) 데이터 생성 중...")
        df = build_triple_barrier_dataset(ticker=TICKER, pt_sl=pt_sl, num_days=NUM_DAYS)
        df.to_csv(out_path)
        print(f"  저장 완료: {out_path.name}")
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


def generate_trades(df: pd.DataFrame) -> pd.DataFrame:
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
    """구간(row) x pt_sl 후보(column) 행렬. 값 = 그 구간에서 그 pt_sl의 복리수익률."""
    datasets = {pt_sl: load_or_build_dataset(pt_sl) for pt_sl in PT_SL_CANDIDATES}
    start = max(df.index.min() for df in datasets.values())
    end = min(df.index.max() for df in datasets.values())
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)
    print(f"전체 비교 구간: {start.date()} ~ {end.date()}, {N_SUBPERIODS}개 구간으로 분할")

    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=[str(p) for p in PT_SL_CANDIDATES], dtype=float)

    for pt_sl in PT_SL_CANDIDATES:
        print(f"pt_sl={pt_sl} 거래 생성 중...")
        trades = generate_trades(datasets[pt_sl])

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            if trades.empty:
                perf.loc[s, str(pt_sl)] = 0.0
                continue
            in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
            perf.loc[s, str(pt_sl)] = (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0

    return perf


def cscv_pbo(perf: pd.DataFrame) -> dict:
    n_periods = perf.shape[0]
    half = n_periods // 2
    all_periods = list(range(n_periods))

    logits = []
    best_is_candidates = []
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
    pbo = (logits <= 0).mean()

    return {
        "n_combinations": len(logits),
        "pbo": pbo,
        "logit_mean": logits.mean(),
        "logit_median": np.median(logits),
        "best_is_candidates": pd.Series(best_is_candidates).value_counts(),
    }


if __name__ == "__main__":
    perf = build_performance_matrix()
    print(f"\n=== 구간별 pt_sl 후보 성과 (064350 BASE 전용, num_days={NUM_DAYS}, threshold={THRESHOLD} 고정) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (pt_sl 축, num_days={NUM_DAYS}, {len(PT_SL_CANDIDATES)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 pt_sl ===")
    print(result["best_is_candidates"].to_string())
    print("\n(채택한 '(2, 1)'이 여기서 자주 1등이면 선택이 근거 있었다는 뜻.")
    print(" num_days=20 기준 이전 결과: PBO 39.3%(중간), IS 1위는 (3,1) 63%였지만")
    print(" 그 (3,1)은 자체 threshold PBO 86.9%로 함정임이 드러나 (2,1)이 최종 유지됐음 --")
    print(" num_days=30인 지금 같은 패턴이 재현되는지, 다른 후보가 뜨는지 비교할 것.")
    print(" 만약 (2,1) 아닌 값이 IS 최고빈도로 나오면, (3,1) 때처럼 그 값을 직접")
    print(" 5-seed 백테스트로 재검증하기 전까지는 바로 채택하지 말 것.)")

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- IS 최고 후보가 OOS에서 신뢰 어려움")