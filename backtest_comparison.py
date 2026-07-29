"""
BASE vs COMBINED 신호 기반 거래비용 반영 백테스트 비교

핵심 질문:
    ablation에서 AUC/accuracy가 개선됐다고, 실제로 돈을 더 버는가?
    -- NVDA 프로젝트가 폐기된 결정적 이유가 정확히 이거였음.
       AUC는 0.55 나왔는데 거래비용 반영 백테스트에서 Buy & Hold를 크게 못 이김.
    이번엔 COMBINED가 정말 BASE보다 "실전에서" 나은지 여기서 확정지음.

설계:
    backtest_simulation.py와 동일한 로직(순차 진입, 겹침 방지, 왕복 거래비용)을 재사용하되,
    BASE/COMBINED 두 feature 세트로 각각 거래를 생성해서 나란히 비교.
    같은 fold 구성(train_xgboost_ablation.py)을 그대로 써서 공정하게 비교.

사용법:
    python backtest_comparison.py

전제:
    train_xgboost_ablation.py와 같은 폴더에 있어야 함 (거기서 상수/함수를 가져다 씀).
    064350_features_with_investor_h5.csv (또는 원하는 데이터셋)가 있어야 함.
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from train_xgboost_ablation import (
    FEATURE_COLS_BASE, FEATURE_COLS_COMBINED, HORIZON, load_dataset, walk_forward_splits,
)

ROUND_TRIP_COST = 0.002  # 왕복 거래비용 0.2% (backtest_simulation.py와 동일 가정)


# ------------------------------------------------------------------
# 1. fold별로 신호 생성 (순차 진입, 겹침 방지) -- backtest_simulation.py와 동일 로직
# ------------------------------------------------------------------
def generate_trades(df: pd.DataFrame, feature_cols: list, threshold: float = 0.55,
                     train_size=300, test_size=60, step=60, embargo=HORIZON,
                     horizon=HORIZON, random_state=42):
    X = df[feature_cols]
    y = df["label"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

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
                # 보유 기간(horizon)만큼은 새 진입 안 함 -> 겹침 방지
                i += horizon
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
# 4. Buy & Hold 벤치마크 (검증 전체 기간 동일 기준)
# ------------------------------------------------------------------
def buy_and_hold_benchmark(df: pd.DataFrame, train_size=300, test_size=60, step=60, embargo=HORIZON) -> dict:
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
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


if __name__ == "__main__":
    df = load_dataset()
    print(f"전체 데이터: {df.shape[0]}행\n")

    THRESHOLD = 0.55  # ablation에서 accuracy가 0.51~0.54대라 0.6은 신호가 너무 적을 수 있어 낮춰 잡음
    print(f"threshold = {THRESHOLD}\n")

    print("=== BASE 신호로 거래 생성 중... ===")
    base_trades, base_n_signals = generate_trades(df, FEATURE_COLS_BASE, threshold=THRESHOLD)
    print(f"  신호 개수(모든 fold 합산, 겹침 제거 전): {base_n_signals} / 실제 거래로 성사된 건수: {len(base_trades)}")

    print("=== COMBINED 신호로 거래 생성 중... ===")
    combined_trades, combined_n_signals = generate_trades(df, FEATURE_COLS_COMBINED, threshold=THRESHOLD)
    print(f"  신호 개수(모든 fold 합산, 겹침 제거 전): {combined_n_signals} / 실제 거래로 성사된 건수: {len(combined_trades)}")

    if base_trades.empty or combined_trades.empty:
        print("\n⚠️  거래가 하나도 발생하지 않은 세트가 있어요. THRESHOLD를 낮춰서 (예: 0.5) 다시 시도해보세요.")

    base_result = evaluate_strategy(base_trades, "BASE")
    combined_result = evaluate_strategy(combined_trades, "COMBINED")
    bh_result = buy_and_hold_benchmark(df)

    print(f"\n검증 전체 기간: {bh_result['period']}\n")

    summary = pd.DataFrame([base_result, combined_result, bh_result]).set_index("label")
    cols = ["n_trades", "win_rate", "avg_net_return", "total_compound_return", "sharpe_per_trade", "mdd", "return_per_risk"]

    print("=" * 70)
    print("=== 전략 성과 비교 (거래비용 반영) ===")
    print("=" * 70)
    print(summary[cols].round(4).to_string())

    print("\n" + "=" * 70)
    print("=== 판정 ===")
    print("=" * 70)

    base_ret = base_result["total_compound_return"]
    combined_ret = combined_result["total_compound_return"]
    bh_ret = bh_result["total_compound_return"]

    print(f"BASE 누적수익률:     {base_ret:.1%}" if pd.notna(base_ret) else "BASE: 거래 없음")
    print(f"COMBINED 누적수익률: {combined_ret:.1%}" if pd.notna(combined_ret) else "COMBINED: 거래 없음")
    print(f"Buy & Hold 누적수익률: {bh_ret:.1%}")

    if pd.notna(base_ret) and pd.notna(combined_ret):
        print()
        if combined_ret > base_ret:
            print(f"→ COMBINED가 BASE보다 누적수익률 {combined_ret - base_ret:+.1%}p 높음.")
        else:
            print(f"→ COMBINED가 BASE보다 누적수익률 {combined_ret - base_ret:+.1%}p 낮음 -- AUC 개선이 실전 손익으로 안 이어짐.")

        if combined_ret > bh_ret and base_ret <= bh_ret:
            print("→ 특히 COMBINED만 Buy & Hold를 이김. 수급 feature가 실전 손익 개선에 실질적으로 기여했다는 뜻.")
        elif combined_ret > bh_ret and base_ret > bh_ret:
            print("→ 둘 다 Buy & Hold를 이기지만, COMBINED가 그 중에서도 더 나음.")
        elif combined_ret <= bh_ret:
            print("→ COMBINED도 Buy & Hold를 못 이김 -- AUC/정확도 개선과 별개로 '전략'으로서는 아직 부족.")
            print("   (AUC 개선이 손익 개선으로 이어지려면 신호 품질뿐 아니라 신호 빈도/타이밍/포지션 사이징도 맞아야 함)")