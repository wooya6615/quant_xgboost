"""
PBO(Probability of Backtest Overfitting) -- 064350 BASE 전용, pt_sl x threshold
완전 결합 축.

지금까지(compute_pbo_pt_sl_064350.py, compute_pbo_base_only_064350.py,
compute_pbo_threshold_pt3sl1_064350.py)는 두 축을 서로 고정한 채 따로따로
검증했음 (pt_sl 축은 threshold=0.60 고정, threshold 축은 pt_sl 하나 고정).
이번엔 pt_sl(5개) x threshold(6개) = 30개 후보를 전부 하나의 CSCV 후보 풀로
묶어서, "이 프로젝트가 이 세션에서 실제로 탐색했던 전체 파라미터 공간"을
정확히 반영한 PBO를 계산함.

[최적화] threshold는 모델 학습에 영향을 주지 않고 거래 추출(진입 여부 판단)
단계에서만 쓰이므로, pt_sl 하나당 walk-forward 학습은 1번만 하고 각 fold의
(test_positions, proba)를 캐싱해서 6개 threshold 각각에 대해 거래 추출만
반복함 (30번 전체 재학습 대신 5번만 재학습).

사용법 (레포 루트에서):
    python src/compute_pbo_joint_pt_sl_threshold_064350.py

전제:
    compute_pbo_pt_sl_064350.py로 pt_sl 5개 후보(1,1)/(1.5,1)/(2,1)/(2.5,1)/(3,1)
    데이터셋이 이미 생성돼 있어야 함 (없으면 이 스크립트가 알아서 생성함).
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
NUM_DAYS = 20
PT_SL_CANDIDATES = [(1, 1), (1.5, 1), (2, 1), (2.5, 1), (3, 1)]
THRESHOLD_CANDIDATES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
SEED = 42
N_SUBPERIODS = 10
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002


def config_label(pt_sl: tuple) -> str:
    return f"pt{pt_sl[0]}sl{pt_sl[1]}_nd{NUM_DAYS}_hl"


def load_or_build_dataset(pt_sl: tuple) -> pd.DataFrame:
    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label(pt_sl)}_base.csv"
    if out_path.exists():
        df = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"pt_sl={pt_sl} 데이터 생성 중...")
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


def compute_fold_probas(df: pd.DataFrame) -> list:
    """pt_sl 하나당 1번만 학습 -- fold별 (test_positions, proba)를 리스트로 반환."""
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    fold_results = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        if y_train.nunique() < 2:
            fold_results.append(None)
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
        fold_results.append((test_positions, proba))

    return fold_results


def extract_trades(df: pd.DataFrame, fold_results: list, threshold: float) -> pd.DataFrame:
    """캐싱된 fold별 proba로 threshold만 바꿔서 거래 추출 (재학습 없음)."""
    trades = []
    for fold in fold_results:
        if fold is None:
            continue
        test_positions, proba = fold

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
                        "exit_date": entry_date + pd.Timedelta(days=holding),
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1

    return pd.DataFrame(trades)


def build_performance_matrix() -> tuple:
    datasets = {pt_sl: load_or_build_dataset(pt_sl) for pt_sl in PT_SL_CANDIDATES}
    start = max(df.index.min() for df in datasets.values())
    end = min(df.index.max() for df in datasets.values())
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)
    print(f"전체 비교 구간: {start.date()} ~ {end.date()}, {N_SUBPERIODS}개 구간\n")

    candidate_labels = [f"pt_sl={p}, th={t}" for p in PT_SL_CANDIDATES for t in THRESHOLD_CANDIDATES]
    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=candidate_labels, dtype=float)

    for pt_sl in PT_SL_CANDIDATES:
        print(f"pt_sl={pt_sl} 학습 중 (1회, 이후 threshold {len(THRESHOLD_CANDIDATES)}개는 재사용)...")
        fold_results = compute_fold_probas(datasets[pt_sl])

        for threshold in THRESHOLD_CANDIDATES:
            trades = extract_trades(datasets[pt_sl], fold_results, threshold)
            label = f"pt_sl={pt_sl}, th={threshold}"

            for s in range(N_SUBPERIODS):
                period_start, period_end = bounds[s], bounds[s + 1]
                if trades.empty:
                    perf.loc[s, label] = 0.0
                    continue
                in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
                perf.loc[s, label] = (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0

    return perf, candidate_labels


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
    perf, labels = build_performance_matrix()

    print(f"\n=== 구간별 전체 후보 평균 성과 (상위 5개, 전체 {len(labels)}개 후보 중) ===")
    print(perf.mean().sort_values(ascending=False).head(5).round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (pt_sl x threshold 완전 결합, {len(labels)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 후보 (상위 10개) ===")
    print(result["best_is_candidates"].head(10).to_string())

    adopted_label = "pt_sl=(2, 1), th=0.6"
    if adopted_label in result["best_is_candidates"].index:
        rank = list(result["best_is_candidates"].index).index(adopted_label) + 1
        count = result["best_is_candidates"][adopted_label]
        print(f"\n채택값 '{adopted_label}': IS 최고빈도 {count}/{result['n_combinations']}, 전체 {len(labels)}개 중 {rank}위")
    else:
        print(f"\n채택값 '{adopted_label}'은 IS 최고빈도 0회 (한 번도 1등 못 함)")

    print("\n판정 기준: PBO<20% 낮음 / 20~50% 중간 / >=50% 높음")
    print("(개별 축 PBO: pt_sl 39.3%, threshold(2,1기준) 19.4%, threshold(3,1기준) 86.9% --")
    print(" 완전 결합은 이 셋보다 후보 수가 많아(30개) 일반적으로 더 높게 나올 가능성이 있음,")
    print(" 이는 방법론적으로 정상 -- 더 넓은 공간을 탐색했다는 사실 자체를 반영하는 것)")