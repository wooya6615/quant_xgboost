"""
triple-barrier 라벨(1:2 손익비, pt2sl1) 기반 실전 백테스트.
지금까지 매번 "AUC는 좋은데 백테스트에서 무너진다"는 패턴이 반복됐으니,
label_tb_binary가 5/5 시드 AUC 개선을 보였다고 여기서 끝내지 않고 반드시 확인.

기존 backtest_simulation.py / backtest_comparison_fx.py와의 차이:
    - 고정 horizon이 아니라 가변 보유기간(holding_rows_tb, 실제 배리어 도달까지 걸린
      거래일 수)만큼 다음 진입을 건너뜀 -- triple-barrier는 거래마다 청산 시점이 다르므로.
    - 손익은 future_return이 아니라 ret_tb(배리어 도달 시점 실현수익률)를 사용.

사용법 (레포 루트에서):
    python src/backtest_triple_barrier.py

전제:
    feature_engineering_triple_barrier.py로 084010_features_triple_barrier_pt2sl1.csv가
    먼저 생성돼 있어야 함.
"""

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

ROUND_TRIP_COST = 0.002  # 왕복 거래비용 0.2% -- 기존 실험들과 동일 가정
HORIZON_FOR_EMBARGO = 20  # 데이터 생성 시 num_days=20으로 만들었으므로 동일하게 embargo


def load_dataset(ticker_krx: str = "084010", pt_sl_label: str = "pt2sl1",
                  combined: bool = False) -> pd.DataFrame:
    suffix = "_valuation" if combined else ""
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{pt_sl_label}{suffix}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    df["holding_rows_tb"] = df["holding_rows_tb"].clip(lower=1).astype(int)  # 방어: 0 이하 방지
    return df


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


def generate_trades(df: pd.DataFrame, feature_cols: list = None, threshold: float = 0.55,
                     train_size: int = 300, test_size: int = 60, step: int = 60,
                     embargo: int = HORIZON_FOR_EMBARGO, random_state: int = 42) -> pd.DataFrame:
    feature_cols = feature_cols or FEATURE_COLS_BASE
    X = df[feature_cols]
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
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]  # triple-barrier 실현수익률
                holding = int(df["holding_rows_tb"].iloc[row_idx])

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": df.index[exit_row],
                        "proba": proba[i],
                        "holding_days": holding,
                        "gross_return": gross_return,
                        "net_return": net_return,
                    })
                # 실제 보유기간만큼 다음 진입 대기 -> 겹침 방지 (fixed horizon과 다른 점)
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, df: pd.DataFrame) -> dict:
    if trades.empty:
        print("거래가 하나도 생성되지 않았어 -- threshold를 낮추거나 데이터 기간을 늘려볼 것.")
        return {"n_trades": 0, "total_net": None}

    total_net = (1 + trades["net_return"]).prod() - 1
    total_gross = (1 + trades["gross_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()

    test_start = trades["entry_date"].min()
    test_end = trades["exit_date"].max()
    bh_return = df.loc[test_end, "Close"] / df.loc[test_start, "Close"] - 1

    print(f"거래 수: {len(trades)}, 평균 보유일수: {trades['holding_days'].mean():.1f}일")
    print(f"승률(net_return>0): {win_rate:.1%}")
    print(f"복리 총수익 (net, 거래비용 반영): {total_net:.1%}")
    print(f"복리 총수익 (gross, 거래비용 미반영): {total_gross:.1%}")
    print(f"같은 기간 Buy & Hold: {bh_return:.1%} ({test_start.date()} ~ {test_end.date()})")
    print(f"Buy & Hold 대비: {'이김' if total_net > bh_return else '못 이김'}")

    print("\n=== 연도별 분해 (국면 집중도 확인) ===")
    trades["year"] = trades["entry_date"].dt.year
    yearly = trades.groupby("year")["net_return"].agg(["count", "sum", "mean"])
    yearly.columns = ["거래수", "연간합계수익률", "거래당평균수익률"]
    print(yearly.round(4).to_string())

    total_sum_return = trades["net_return"].sum()
    top_year_share = None
    if total_sum_return != 0:
        top_year_share = yearly["연간합계수익률"].abs().max() / yearly["연간합계수익률"].abs().sum()
        print(f"\n최대 기여 연도의 비중: {top_year_share:.1%} "
              f"(50% 넘으면 기존 판정기준상 '국면 의존적'으로 봄)")

    return {
        "n_trades": len(trades), "win_rate": win_rate,
        "total_net": total_net, "total_gross": total_gross,
        "bh_return": bh_return, "top_year_share": top_year_share,
    }


def run_threshold_sweep(df: pd.DataFrame, feature_cols: list, thresholds: list) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        print(f"\n--- threshold={threshold} ---")
        trades = generate_trades(df, feature_cols=feature_cols, threshold=threshold)
        result = summarize(trades, df)
        result["threshold"] = threshold
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== BASE ===")
    df = load_dataset("084010", combined=False)
    print(f"데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})\n")
    trades = generate_trades(df, feature_cols=FEATURE_COLS_BASE)
    summarize(trades, df)

    print("\n\n=== COMBINED (BASE + VALUATION) -- threshold 스윕 ===")
    df_combined = load_dataset("084010", combined=True)
    print(f"데이터: {df_combined.shape[0]}행 "
          f"({df_combined.index.min().date()} ~ {df_combined.index.max().date()})")

    THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
    sweep = run_threshold_sweep(df_combined, FEATURE_COLS_BASE + FEATURE_COLS_VALUATION, THRESHOLDS)

    print("\n\n=== threshold 스윕 요약 (COMBINED) ===")
    print(sweep[["threshold", "n_trades", "win_rate", "total_net", "total_gross", "bh_return"]]
          .round(4).to_string(index=False))
    print("\n판정: net이 threshold를 높일수록 개선되면 '거래비용 과다'가 진짜 원인이었다는 뜻.")
    print("거래 수가 너무 적어지면(예: 10건 미만) 통계적으로 믿기 어려우니 그 지점도 같이 볼 것.")