"""
PBO(Probability of Backtest Overfitting) 계산 -- BASE+INVESTOR 버전, threshold 축.

compute_pbo.py(BASE+VALUATION, threshold 축)와 동일한 CSCV 방법론을 그대로 적용.
BASE+VALUATION 라인은 pt_sl 축(compute_pbo_v3.py)에서 PBO 95.6%로 무너졌지만, 이건
"pt_sl을 세션 중에 탐색하며 골랐다"는 그 세션 고유의 탐색 과정 자체가 문제였던 것 --
feature 소스(수급 vs 밸류에이션)와는 별개 질문. 이 스크립트는 가장 기본적인 축인
threshold 하나만 검증해서 "수급 feature로 바꿔도 이 최소한의 축에서부터 이미 위험한지"
1차로 확인하는 단계.

⚠️ threshold를 통과하면 그걸로 "배포 가능"은 아님 -- BASE+VALUATION도 threshold 축
   PBO는 6.7%로 낮았지만 최종적으로는 pt_sl 축에서 무너졌음. 여기서 통과하면 다음
   단계로 pt_sl 축까지 마저 검증할 것 (compute_pbo_v3_investor.py에 해당하는 스크립트,
   pt_sl별 investor 데이터셋을 추가로 생성해야 함).

사용법 (레포 루트에서):
    python src/compute_pbo_investor.py

전제:
    feature_engineering_triple_barrier_investor.py로 3종목 다
    {ticker}_features_triple_barrier_pt2sl1_nd20_hl_investor.csv가 생성돼 있어야 함.
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
FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR

ROUND_TRIP_COST = 0.002
NUM_DAYS = 20
CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKERS = ["064350", "052690", "118990"]
CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SEED = 42
N_SUBPERIODS = 10  # S -- 짝수, C(10,5)=252로 계산량 적당함


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_investor.csv"
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


def generate_trades(df, threshold, train_size=300, test_size=60, step=60,
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
    """
    행=구간, 열=threshold 후보. 값=3종목 풀링 포트폴리오의 그 구간 복리수익률.
    """
    dfs = {t: load_dataset(t) for t in TICKERS}
    start = max(df.index.min() for df in dfs.values())
    end = min(df.index.max() for df in dfs.values())
    bounds = pd.date_range(start, end, periods=N_SUBPERIODS + 1)
    print(f"전체 비교 구간: {start.date()} ~ {end.date()}, {N_SUBPERIODS}개 구간으로 분할")

    perf = pd.DataFrame(index=range(N_SUBPERIODS), columns=CANDIDATE_THRESHOLDS, dtype=float)

    for threshold in CANDIDATE_THRESHOLDS:
        print(f"  threshold={threshold} 거래 생성 중...")
        ticker_trades = {t: generate_trades(dfs[t], threshold=threshold) for t in TICKERS}

        for s in range(N_SUBPERIODS):
            period_start, period_end = bounds[s], bounds[s + 1]
            ticker_period_returns = []
            for t in TICKERS:
                trades = ticker_trades[t]
                if trades.empty:
                    ticker_period_returns.append(0.0)
                    continue
                in_period = trades[(trades["exit_date"] >= period_start) & (trades["exit_date"] < period_end)]
                if in_period.empty:
                    ticker_period_returns.append(0.0)
                else:
                    ticker_period_returns.append((1 + in_period["net_return"]).prod() - 1)
            perf.loc[s, threshold] = np.mean(ticker_period_returns)

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
    print("\n=== 구간별 threshold 후보 성과 (풀링 포트폴리오, 복리수익률, BASE+INVESTOR) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 (threshold 축) ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f} (양수면 IS 선택이 OOS에서도 유효했다는 뜻)")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print(f"\n=== IS에서 가장 자주 '최고'로 뽑힌 threshold ===")
    print(result["best_is_candidates"].to_string())

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- 신뢰 어려움")
    print("\n[비교 참고] BASE+VALUATION의 threshold 축 PBO는 6.7%였음 (compute_pbo.py).")
    print("여기서 그보다 훨씬 높게 나오면, 수급 feature 자체가 threshold 선택에")
    print("BASE+VALUATION보다 더 민감/불안정하다는 신호.")
    print("\n이 단계 통과 후 다음 순서: pt_sl 축 PBO까지 검증해야 '배포 부적합' 판정을")
    print("완전히 재확인한 게 됨 (BASE+VALUATION은 threshold 축은 통과했지만 pt_sl 축에서")
    print("무너졌던 걸 기억할 것 -- 여기서 통과했다고 안심하지 말 것).")