"""
PBO(Probability of Backtest Overfitting) 계산 -- CSCV(Combinatorially Symmetric
Cross-Validation) 방식 (Bailey, Borwein, de Prado, Zhu 2014).

배경: 이번 세션에서 pt_sl/num_days/threshold/체결방식/종목을 계속 스윕하면서 "제일
좋아 보이는 조합"을 골라왔음. 개별 조합마다 5-seed/국면검증은 통과했지만, "이만큼
많은 조합을 시도했다"는 사실 자체가 통계적으로 얼마나 위험한지는 따로 계산한 적이
없었음. CSCV는 이걸 직접 정량화함.

방법:
    1. 전체 기간을 S개의 연속 구간으로 나눔 (겹치지 않음)
    2. S개 구간을 절반씩(S/2, S/2)으로 나누는 모든 조합(C(S, S/2)개)에 대해:
       - 절반(IS, in-sample)에서 성과가 가장 좋은 후보(threshold 등)를 고름
       - 그 후보가 나머지 절반(OOS, out-of-sample)에서도 상위권인지 확인
       - OOS에서 하위 50% 밑으로 떨어지면 "이번 조합은 과적합 증거"로 카운트
    3. 전체 조합 중 과적합 증거 비율 = PBO

D+1 종가체결(최종 확정)로, threshold 5개 후보(0.50/0.55/0.60/0.65/0.70)를 대상으로
현대로템+한전기술+모트렉스 풀링 포트폴리오 기준 계산.

사용법 (레포 루트에서):
    python src/compute_pbo.py
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
CONFIG_LABEL = "pt2sl1_nd20_hl"  # D+1 종가체결 + High/Low 반영, 최종 확정
TICKERS = ["064350", "052690", "118990"]
CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SEED = 42
N_SUBPERIODS = 10  # S -- 짝수, C(10,5)=252로 계산량 적당함


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
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
    행=구간(0..S-1), 열=threshold 후보. 값=그 구간에서 그 후보(풀링 포트폴리오)의 복리수익률.
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
    for is_periods in combinations(all_periods, half):
        oos_periods = [p for p in all_periods if p not in is_periods]

        is_perf = perf.loc[list(is_periods)].mean()
        oos_perf = perf.loc[oos_periods].mean()

        best_candidate_is = is_perf.idxmax()
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
    }


if __name__ == "__main__":
    perf = build_performance_matrix()
    print("\n=== 구간별 threshold 후보 성과 (풀링 포트폴리오, 복리수익률) ===")
    print(perf.round(4).to_string())

    result = cscv_pbo(perf)
    print(f"\n=== CSCV 결과 ===")
    print(f"전체 조합 수: {result['n_combinations']}")
    print(f"PBO (과적합 확률): {result['pbo']:.1%}")
    print(f"logit 평균: {result['logit_mean']:+.3f} (양수면 전반적으로 IS 선택이 OOS에서도 유효했다는 뜻)")
    print(f"logit 중앙값: {result['logit_median']:+.3f}")

    print("\n판정 기준 (Bailey et al. 권고):")
    print("  PBO < 20%: 낮음 -- 과적합 우려 적음")
    print("  PBO 20~50%: 중간 -- 주의 필요")
    print("  PBO >= 50%: 높음 -- IS에서 최고였던 후보가 OOS에서 동전 던지기보다 못함,")
    print("               지금까지의 '최고 설정 선택' 자체를 신뢰하기 어려움")