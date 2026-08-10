"""
Meta-labeling 백테스트: 1차 신호(모멘텀 부호)를 필터 없이 그대로 다 거래했을 때 vs
2차 모델이 확신하는 날만 거래했을 때를 나란히 비교.

이전 실험(backtest_triple_barrier.py)에서 확인된 문제 -- "gross는 플러스인데 거래
빈도가 너무 높아서 거래비용이 이익을 갉아먹는다" -- 를 메타 모델의 확신도 필터가
해결해주는지가 핵심 질문. 필터링으로 거래 수가 줄면서 net이 개선되는지 확인.

사용법 (레포 루트에서):
    python src/backtest_meta_labeling.py

전제:
    feature_engineering_meta_labeling.py로 064350_features_meta_labeling.csv가
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
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

ROUND_TRIP_COST = 0.002
NUM_DAYS = 10  # feature_engineering_meta_labeling.py와 동일하게 맞출 것
HORIZON_FOR_EMBARGO = NUM_DAYS


def load_dataset(ticker_krx: str = "064350") -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_meta_labeling_nd{NUM_DAYS}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df["holding_rows_meta"] = df["holding_rows_meta"].clip(lower=1).astype(int)
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


def generate_trades(df: pd.DataFrame, meta_threshold: float | None, invert: bool = False,
                     train_size: int = 300, test_size: int = 60, step: int = 60,
                     embargo: int = HORIZON_FOR_EMBARGO, random_state: int = 42) -> pd.DataFrame:
    """
    meta_threshold=None이면 필터 없이 1차 신호를 전부 거래 (baseline).
    meta_threshold=0.55 등을 주면 confidence >= threshold일 때만 거래.
    invert=True면 confidence = 1 - proba를 씀 -- AUC가 0.5보다 유의미하게 낮게 나온
    경우(모델이 순위를 거꾸로 매기고 있다는 뜻), "확률 낮게 나온 베팅"이 오히려
    성공 확률이 높다는 뜻이므로 이걸 반영.
    """
    X = df[FEATURE_COLS_COMBINED]
    y = df["meta_label"]
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)

    trades = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        test_positions = list(test_idx)

        if meta_threshold is not None:
            if y_train.nunique() < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", random_state=random_state,
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X.iloc[test_positions])[:, 1]
            confidence = (1 - proba) if invert else proba
        else:
            confidence = np.ones(len(test_positions))  # 필터 없음 -- 전부 1차 신호대로 거래

        i = 0
        while i < len(test_positions):
            take_trade = (confidence[i] >= meta_threshold) if meta_threshold is not None else True
            if take_trade:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_meta"].iloc[row_idx]
                holding = int(df["holding_rows_meta"].iloc[row_idx])

                if pd.notna(gross_return):
                    net_return = gross_return - ROUND_TRIP_COST
                    exit_row = min(row_idx + holding, len(df) - 1)
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": df.index[exit_row],
                        "primary_side": df["primary_side"].iloc[row_idx],
                        "confidence": confidence[i],
                        "holding_days": holding,
                        "gross_return": gross_return,
                        "net_return": net_return,
                    })
                i += max(holding, 1)
            else:
                i += 1

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, df: pd.DataFrame, label: str, show_yearly: bool = False) -> dict:
    print(f"\n--- {label} ---")
    if trades.empty:
        print("거래가 하나도 생성되지 않았어.")
        return {"label": label, "n_trades": 0}

    total_net = (1 + trades["net_return"]).prod() - 1
    total_gross = (1 + trades["gross_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()

    test_start, test_end = trades["entry_date"].min(), trades["exit_date"].max()
    bh_return = df.loc[test_end, "Close"] / df.loc[test_start, "Close"] - 1

    print(f"거래 수: {len(trades)}, 평균 보유일수: {trades['holding_days'].mean():.1f}일")
    print(f"승률: {win_rate:.1%}")
    print(f"복리 총수익 (net): {total_net:.1%} / (gross): {total_gross:.1%}")
    print(f"같은 기간 Buy & Hold: {bh_return:.1%}")
    print(f"Buy & Hold 대비: {'이김' if total_net > bh_return else '못 이김'}")

    if show_yearly:
        print("\n연도별 분해 (국면 집중도 확인):")
        trades = trades.copy()
        trades["year"] = trades["entry_date"].dt.year
        yearly = trades.groupby("year")["net_return"].agg(["count", "sum"])
        yearly.columns = ["거래수", "연간합계수익률"]
        print(yearly.round(4).to_string())
        # 최대 기여 연도 제외했을 때도 이기는지 (top-contributor exclusion)
        top_year = yearly["연간합계수익률"].idxmax()
        excl_trades = trades[trades["year"] != top_year]
        if not excl_trades.empty:
            excl_net = (1 + excl_trades["net_return"]).prod() - 1
            print(f"\n최대 기여 연도({top_year}) 제외 시 net: {excl_net:.1%} "
                  f"(원래 {total_net:.1%} -> 이 정도 차이면 그 해가 결과를 얼마나 좌우했는지 판단)")

    return {
        "label": label, "n_trades": len(trades), "win_rate": win_rate,
        "total_net": total_net, "total_gross": total_gross, "bh_return": bh_return,
    }


if __name__ == "__main__":
    df = load_dataset("064350")
    print(f"데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})")

    print("\n=== 1차 신호 무필터 (baseline -- 항상 모멘텀 부호대로 거래) ===")
    trades_unfiltered = generate_trades(df, meta_threshold=None)
    r0 = summarize(trades_unfiltered, df, "무필터")

    results = [r0]

    print("\n\n### 원래 방향: 확률 높은 베팅만 거래 (AUC 0.45 기준으로는 기대와 반대일 가능성) ###")
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        trades_filtered = generate_trades(df, meta_threshold=threshold, invert=False)
        r = summarize(trades_filtered, df, f"정방향 threshold={threshold}")
        results.append(r)

    print("\n\n### 반전 방향: 확률 낮은 베팅만 거래 (5-seed AUC가 0.5 아래로 일관되게 나온 것을 반영) ###")
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        trades_inverted = generate_trades(df, meta_threshold=threshold, invert=True)
        r = summarize(trades_inverted, df, f"반전 threshold={threshold}")
        results.append(r)

    print("\n\n=== 전체 비교 ===")
    summary_df = pd.DataFrame(results)
    print(summary_df.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 여기서 끝내지 않고, 가장 성과 좋았던 반전 threshold=0.65를 5-seed + 연도별
    # 분해로 검증 -- 지금까지 이 프로젝트에서 "첫 결과가 좋았다가 검증에서 무너진"
    # 사례가 압도적으로 많았으므로 반드시 거쳐야 하는 단계.
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("=== 검증 1: 반전 threshold=0.65의 5-seed 재현성 ===")
    print("=" * 60)
    SEEDS = [42, 1, 7, 123, 2024]
    seed_results = []
    for seed in SEEDS:
        trades_seed = generate_trades(df, meta_threshold=0.65, invert=True, random_state=seed)
        r = summarize(trades_seed, df, f"seed={seed}")
        seed_results.append(r)

    seed_df = pd.DataFrame(seed_results)
    print("\n--- 5-seed 요약 ---")
    print(seed_df[["label", "n_trades", "total_net", "bh_return"]].round(4).to_string(index=False))
    win_count = (seed_df["total_net"] > seed_df["bh_return"]).sum()
    print(f"\n5개 시드 중 Buy & Hold를 이긴 시드: {win_count}/5")
    print("판정: 5/5 또는 4/5 이겨야 '진짜' -- 1~2/5면 seed=42가 우연히 좋았던 것일 가능성 높음.")

    print("\n\n" + "=" * 60)
    print("=== 검증 2: 반전 threshold=0.65의 연도별 국면 집중도 (seed=42 기준) ===")
    print("=" * 60)
    trades_best = generate_trades(df, meta_threshold=0.65, invert=True, random_state=42)
    summarize(trades_best, df, "반전 threshold=0.65 (연도별 분해 포함)", show_yearly=True)