"""
PBO(Probability of Backtest Overfitting) -- pt_sl(손익비) 축.

compute_pbo.py(threshold 축), compute_pbo_v2.py(threshold x 종목조합 축)에 이은
세 번째. 이번엔 종목조합/threshold는 이미 정한 대로(3종목 블렌딩, threshold=0.65)
고정하고, 이 세션에서 실제로 탐색했던 pt_sl 축만 검증. 이게 이 세션의 실제
탐색 과정과 가장 정확히 일치하는 PBO 축임.

사용법 (레포 루트에서):
    python src/compute_pbo_v3.py

전제:
    build_pt_sl_variants.py로 5개 pt_sl 후보 데이터셋이 생성돼 있어야 함.
"""

from itertools import combinations
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
TICKERS = ["064350", "052690", "118990"]
PT_SL_CANDIDATES = [(1, 1), (1.5, 1), (2, 1), (2.5, 1), (3, 1)]
THRESHOLD = 0.65
SEED = 42
N_SUBPERIODS = 10


def config_label(pt_sl: tuple) -> str:
    return f"pt{pt_sl[0]}sl{pt_sl[1]}_nd{NUM_DAYS}_hl"


def load_dataset(ticker_krx: str, pt_sl: tuple) -> pd.DataFrame:
    label = config_label(pt_sl)
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{label}_valuation.csv"
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
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = int(df["holding_rows_tb"].iloc[row_idx])
                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "exit_date": df.index[exit_row],
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1
    return pd.DataFrame(trades)


def build_performance_matrix() -> pd.DataFrame:
    """행=구간, 열=pt_sl 후보. 값=3종목 블렌딩 포트폴리오의 그 구간 복리수익률."""
    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=[str(p) for p in PT_SL_CANDIDATES], dtype=float)

    for pt_sl in PT_SL_CANDIDATES:
        print(f"pt_sl={pt_sl} 거래 생성 중...")
        dfs = {t: load_dataset(t, pt_sl) for t in TICKERS}
        start = max(df.index.min() for df in dfs.values())
        end = min(df.index.max() for df in dfs.values())
        bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)

        ticker_trades = {t: generate_trades(dfs[t]) for t in TICKERS}

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            ticker_period_returns = []
            for t in TICKERS:
                trades = ticker_trades[t]
                if trades.empty:
                    ticker_period_returns.append(0.0)
                    continue
                in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
                ticker_period_returns.append(
                    (1 + in_period["net_return"]).prod() - 1 if not in_period.empty else 0.0
                )
            perf.loc[s, str(pt_sl)] = np.mean(ticker_period_returns)

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
    print(f"\n=== 구간별 pt_sl 후보 성과 (3종목 블렌딩, threshold={THRESHOLD}) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (pt_sl 축, {len(PT_SL_CANDIDATES)}개 후보) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f}")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 pt_sl ===")
    print(result["best_is_candidates"].to_string())
    print("\n(우리가 최종 채택한 '(2, 1)'이 여기서 가장 자주 1등이면 선택이 근거 있었다는 뜻)")

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- 신뢰 어려움")