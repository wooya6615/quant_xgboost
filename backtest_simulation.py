"""
거래비용 반영 백테스트: XGBoost 신호 기반 전략 vs Buy & Hold

핵심 설계:
- Walk-forward 방식 그대로 유지 (train_xgboost_wfo.py와 동일한 fold 구조 + embargo)
- 신호가 겹치는 문제 방지: 포지션 보유 중(horizon일)에는 새로운 진입 안 함
  (겹치는 진입을 허용하면 자본 배분이 애매해지고 복리 계산이 꼬임 -> 단순화를 위해 순차 진입만 허용)
- 매 거래마다 왕복 거래비용(진입+청산) 차감
- 최종적으로 전략 누적수익률 vs 같은 기간 buy&hold 비교

주의: nvda_features.csv에 'Close'와 'future_return' 컬럼이 있어야 합니다.
     (최신 feature_engineering.py로 다시 만드세요)

사용법:
    python backtest_simulation.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb


FEATURE_COLS = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

HORIZON = 10          # feature_engineering.py의 horizon과 동일하게
ROUND_TRIP_COST = 0.002  # 왕복 거래비용 0.2% (수수료+슬리피지 가정, 필요시 조정)


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_dataset(path: str = "064350_features_with_short_h5.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    required = set(FEATURE_COLS + ["label", "future_return", "Close"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 다음 컬럼이 없어요: {missing}. feature_engineering.py를 최신 버전으로 다시 실행하세요.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label", "future_return"])
    return df


# ------------------------------------------------------------------
# 2. Walk-Forward 분할 (train_xgboost_wfo.py와 동일)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 3. fold별로 신호 생성 (순차 진입, 겹침 방지)
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, train_size=300, test_size=60, step=60,
                     embargo=HORIZON, threshold=0.6, horizon=HORIZON):
    X = df[FEATURE_COLS]
    y = df["label"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

    trades = []  # 각 거래: 진입일, 청산일, 총수익률(future_return), 비용 차감 후 순수익률

    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        proba = model.predict_proba(X.iloc[test_positions])[:, 1]

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["future_return"].iloc[row_idx]  # horizon일 후 실제 수익률

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_date = df.index[min(row_idx + horizon, len(df) - 1)]
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "proba": proba[i],
                        "gross_return": gross_return,
                        "net_return": net_return,
                    })
                # 보유 기간(horizon)만큼은 새 진입 안 함 -> 겹침 방지
                i += horizon
            else:
                i += 1

    return pd.DataFrame(trades)


# ------------------------------------------------------------------
# 4-1. 최대 낙폭(MDD) 계산 유틸
# ------------------------------------------------------------------
def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


# ------------------------------------------------------------------
# 5. 성과 계산: 전략 vs Buy & Hold
# ------------------------------------------------------------------
def evaluate_strategy(trades: pd.DataFrame, df: pd.DataFrame, train_size=300, test_size=60,
                       step=60, embargo=HORIZON):
    if trades.empty:
        print("거래가 하나도 발생하지 않았어요. threshold를 낮춰보세요.")
        return

    n_trades = len(trades)
    win_rate = (trades["net_return"] > 0).mean()
    avg_net_return = trades["net_return"].mean()
    total_compound_return = (1 + trades["net_return"]).prod() - 1  # 모든 거래를 순차 복리로 가정

    sharpe_per_trade = avg_net_return / trades["net_return"].std() if trades["net_return"].std() > 0 else np.nan

    # 전략 MDD (거래 순서 기준 누적)
    strategy_equity = (1 + trades["net_return"]).cumprod()
    strategy_mdd = max_drawdown(strategy_equity)

    # 시장 노출 비율: 실제 보유했던 거래일 수 / 전체 검증 기간 거래일 수
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    total_test_days = last_test_idx - first_test_idx + 1
    exposure_ratio = (n_trades * HORIZON) / total_test_days

    # Buy & Hold 벤치마크: 같은 기간의 일별 종가로 누적수익 곡선 + MDD 계산
    bh_prices = df["Close"].iloc[first_test_idx:last_test_idx + 1]
    bh_equity = bh_prices / bh_prices.iloc[0]
    buy_hold_return = bh_equity.iloc[-1] - 1
    buy_hold_mdd = max_drawdown(bh_equity)

    # 리스크 조정 수익 (수익률 / |MDD|) -- 낙폭 대비 얼마나 벌었는지
    strategy_return_per_risk = total_compound_return / abs(strategy_mdd) if strategy_mdd != 0 else np.nan
    bh_return_per_risk = buy_hold_return / abs(buy_hold_mdd) if buy_hold_mdd != 0 else np.nan

    print("=== 전략 성과 (거래비용 반영) ===")
    print(f"총 거래 횟수:          {n_trades}")
    print(f"시장 노출 비율:        {exposure_ratio:.1%}  (나머지는 현금 보유)")
    print(f"승률:                  {win_rate:.1%}")
    print(f"거래당 평균 순수익률:   {avg_net_return:.2%}")
    print(f"전체 복리 누적수익률:   {total_compound_return:.1%}")
    print(f"거래 단위 Sharpe:       {sharpe_per_trade:.2f}")
    print(f"전략 MDD:              {strategy_mdd:.1%}")
    print(f"수익/|MDD| 비율:        {strategy_return_per_risk:.2f}")
    print()
    print("=== Buy & Hold 비교 ===")
    print(f"검증 전체 기간: {df.index[first_test_idx].date()} ~ {df.index[last_test_idx].date()}")
    print(f"Buy & Hold 누적수익률:  {buy_hold_return:.1%}")
    print(f"Buy & Hold MDD:         {buy_hold_mdd:.1%}")
    print(f"수익/|MDD| 비율:        {bh_return_per_risk:.2f}")
    print()
    print("=== MDD 비교 결론 ===")
    if abs(strategy_mdd) < abs(buy_hold_mdd):
        print(f"전략의 낙폭이 buy&hold보다 작음 ({strategy_mdd:.1%} vs {buy_hold_mdd:.1%}) "
              f"-> 리스크 관점에서는 전략이 더 방어적")
    else:
        print(f"전략의 낙폭이 buy&hold보다 큼 ({strategy_mdd:.1%} vs {buy_hold_mdd:.1%}) "
              f"-> 리스크 관점에서도 전략이 불리함, 절대수익도 못 이기고 낙폭도 못 줄였다는 뜻")

    if strategy_return_per_risk > bh_return_per_risk:
        print(f"수익/낙폭 비율은 전략이 더 우수 ({strategy_return_per_risk:.2f} vs {bh_return_per_risk:.2f})")
    else:
        print(f"수익/낙폭 비율도 buy&hold가 더 우수 ({bh_return_per_risk:.2f} vs {strategy_return_per_risk:.2f})")


if __name__ == "__main__":
    df = load_dataset("005930_features.csv")
    print(f"전체 데이터: {df.shape[0]}행\n")

    THRESHOLD = 0.6  # 이전 sweep에서 유의미한 우위 보인 구간으로 설정, 필요시 조정

    trades = generate_trades(df, threshold=THRESHOLD)
    print(f"threshold={THRESHOLD} 기준 발생한 거래:\n{trades.head(10)}\n")

    evaluate_strategy(trades, df)