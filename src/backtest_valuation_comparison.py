"""
BASE vs VALUATION_ONLY vs COMBINED 신호 기반 거래비용 반영 백테스트 비교

핵심 질문:
    ablation에서 VALUATION_ONLY 단독 AUC(0.58)가 COMBINED(0.54)보다 오히려 높게 나왔음.
    AUC가 실제 손익으로 이어지는지, 그리고 셋 중 뭐가 진짜 제일 나은지 여기서 확인.

설계:
    backtest_comparison.py와 동일 로직(순차 진입, 겹침 방지, 왕복 거래비용)을 재사용하되,
    horizon을 인자로 받아 어떤 horizon 데이터셋이든(특히 h10, h20) 바로 백테스트 가능하게 함.

사용법:
    python backtest_valuation_comparison.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from train_xgboost_valuation_ablation import (
    FEATURE_COLS_BASE, FEATURE_COLS_VALUATION_ONLY, FEATURE_COLS_COMBINED,
    load_dataset, walk_forward_splits,
)

ROUND_TRIP_COST = 0.002


# ------------------------------------------------------------------
# 1. fold별로 신호 생성 (순차 진입, 겹침 방지)
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, feature_cols: list, horizon: int, threshold: float = 0.55,
                     train_size=300, test_size=60, step=60, random_state=42):
    X = df[feature_cols]
    y = df["label"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo=horizon)

    trades = []
    n_signals_total = 0

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
        n_signals_total += int((proba >= threshold).sum())

        i = 0
        while i < len(test_positions):
            if proba[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["future_return"].iloc[row_idx]

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
                i += horizon  # 보유 기간만큼 새 진입 안 함 -> 겹침 방지
            else:
                i += 1

    return pd.DataFrame(trades), n_signals_total


# ------------------------------------------------------------------
# 2. 최대 낙폭(MDD)
# ------------------------------------------------------------------
def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


# ------------------------------------------------------------------
# 3. 전략 성과 계산
# ------------------------------------------------------------------
def evaluate_strategy(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {
            "label": label, "n_trades": 0, "win_rate": np.nan, "avg_net_return": np.nan,
            "total_compound_return": np.nan, "sharpe_per_trade": np.nan, "mdd": np.nan,
            "return_per_risk": np.nan,
        }

    n_trades = len(trades)
    win_rate = (trades["net_return"] > 0).mean()
    avg_net_return = trades["net_return"].mean()
    total_compound_return = (1 + trades["net_return"]).prod() - 1
    sharpe = avg_net_return / trades["net_return"].std() if trades["net_return"].std() > 0 else np.nan

    equity = (1 + trades["net_return"]).cumprod()
    mdd = max_drawdown(equity)
    return_per_risk = total_compound_return / abs(mdd) if mdd != 0 else np.nan

    return {
        "label": label, "n_trades": n_trades, "win_rate": win_rate,
        "avg_net_return": avg_net_return, "total_compound_return": total_compound_return,
        "sharpe_per_trade": sharpe, "mdd": mdd, "return_per_risk": return_per_risk,
    }


# ------------------------------------------------------------------
# 4. Buy & Hold 벤치마크
# ------------------------------------------------------------------
def buy_and_hold_benchmark(df: pd.DataFrame, horizon: int, train_size=300, test_size=60, step=60) -> dict:
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo=horizon)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]

    bh_prices = df["Close"].iloc[first_test_idx:last_test_idx + 1]
    bh_equity = bh_prices / bh_prices.iloc[0]
    bh_return = bh_equity.iloc[-1] - 1
    bh_mdd = max_drawdown(bh_equity)

    return {
        "label": "Buy & Hold", "n_trades": np.nan, "win_rate": np.nan, "avg_net_return": np.nan,
        "total_compound_return": bh_return, "sharpe_per_trade": np.nan, "mdd": bh_mdd,
        "return_per_risk": bh_return / abs(bh_mdd) if bh_mdd != 0 else np.nan,
        "period": f"{df.index[first_test_idx].date()} ~ {df.index[last_test_idx].date()}",
    }


# ------------------------------------------------------------------
# 5. 실행 -- BASE / VALUATION_ONLY / COMBINED 세 가지 다 비교
# ------------------------------------------------------------------
def run_backtest_comparison(ticker_krx: str, horizon: int, threshold: float = 0.55):
    df = load_dataset(ticker_krx=ticker_krx, horizon=horizon)
    print(f"전체 데이터: {df.shape[0]}행, horizon={horizon}, threshold={threshold}\n")

    feature_sets = {
        "BASE": FEATURE_COLS_BASE,
        "VALUATION_ONLY": FEATURE_COLS_VALUATION_ONLY,
        "COMBINED": FEATURE_COLS_COMBINED,
    }

    results = []
    for label, feature_cols in feature_sets.items():
        print(f"=== {label} 신호로 거래 생성 중... ===")
        trades, n_signals = generate_trades(df, feature_cols, horizon=horizon, threshold=threshold)
        print(f"  신호 개수(겹침 제거 전): {n_signals} / 실제 거래 성사: {len(trades)}")
        results.append(evaluate_strategy(trades, label))

    bh_result = buy_and_hold_benchmark(df, horizon=horizon)
    results.append(bh_result)

    print(f"\n검증 전체 기간: {bh_result['period']}\n")

    summary = pd.DataFrame(results).set_index("label")
    cols = ["n_trades", "win_rate", "avg_net_return", "total_compound_return", "sharpe_per_trade", "mdd", "return_per_risk"]

    print("=" * 70)
    print(f"=== 전략 성과 비교 (거래비용 반영, horizon={horizon}) ===")
    print("=" * 70)
    print(summary[cols].round(4).to_string())

    return summary


if __name__ == "__main__":
    TICKER_KRX = "064350"

    print("#" * 70)
    print("# horizon = 10")
    print("#" * 70)
    summary_h10 = run_backtest_comparison(TICKER_KRX, horizon=10)

    print("\n\n" + "#" * 70)
    print("# horizon = 20")
    print("#" * 70)
    summary_h20 = run_backtest_comparison(TICKER_KRX, horizon=20)

    print("\n" + "=" * 70)
    print("=== horizon=10 vs horizon=20 요약 비교 (COMBINED 기준) ===")
    print("=" * 70)
    for label, summary in [("horizon=10", summary_h10), ("horizon=20", summary_h20)]:
        combined_ret = summary.loc["COMBINED", "total_compound_return"]
        bh_ret = summary.loc["Buy & Hold", "total_compound_return"]
        print(f"{label}: COMBINED {combined_ret:.1%} vs Buy&Hold {bh_ret:.1%}")