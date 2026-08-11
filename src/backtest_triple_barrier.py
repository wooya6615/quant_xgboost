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
    feature_engineering_triple_barrier.py로 118990_features_triple_barrier_pt2sl1_nd20.csv가
    먼저 생성돼 있어야 함 (CONFIG_LABEL 상수 참고).
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

TICKER_KRX = "118990"     # 종목 바꿀 땐 이것만 수정 (한전기술 -- 시가체결 재검증 3번째)
CONFIG_LABEL = "pt2sl1_nd20_hl"  # feature_engineering_triple_barrier.py와 동일하게 맞출 것 (D+1종가체결+High/Low 최종확정)
NUM_DAYS = 20              # CONFIG_LABEL의 nd 숫자와 동일하게 맞출 것 (embargo에 사용)

ROUND_TRIP_COST = 0.002  # 왕복 거래비용 0.2% -- 기존 실험들과 동일 가정
HORIZON_FOR_EMBARGO = NUM_DAYS


def load_dataset(ticker_krx: str = TICKER_KRX, pt_sl_label: str = CONFIG_LABEL,
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


def get_fixed_bh_window(df: pd.DataFrame, train_size: int = 300, test_size: int = 60,
                         step: int = 60, embargo: int = HORIZON_FOR_EMBARGO) -> tuple:
    """
    Buy&Hold 비교 기준을 "이번에 나온 거래들의 기간"이 아니라 walk-forward 전체
    테스트 구간(첫 fold 시작 ~ 마지막 fold 끝)으로 고정. 어떤 시드/threshold가
    어떤 날짜를 신호로 골랐는지와 무관하게 항상 동일한 기간과 비교해야 공정함
    (다르면 시드마다 Buy&Hold 값 자체가 흔들려서 비교가 왜곡됨).
    """
    splits = walk_forward_splits(len(df), train_size, test_size, step, embargo)
    first_test_idx = splits[0][1][0]
    last_test_idx = splits[-1][1][-1]
    return df.index[first_test_idx], df.index[min(last_test_idx, len(df) - 1)]


def summarize(trades: pd.DataFrame, df: pd.DataFrame, show_top_year_exclusion: bool = False,
              bh_window: tuple | None = None) -> dict:
    if trades.empty:
        print("거래가 하나도 생성되지 않았어 -- threshold를 낮추거나 데이터 기간을 늘려볼 것.")
        return {"n_trades": 0, "total_net": None}

    total_net = (1 + trades["net_return"]).prod() - 1
    total_gross = (1 + trades["gross_return"]).prod() - 1
    win_rate = (trades["net_return"] > 0).mean()

    if bh_window is not None:
        test_start, test_end = bh_window
    else:
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
    trades = trades.copy()
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

    if show_top_year_exclusion:
        top_year = yearly["연간합계수익률"].idxmax()
        excl_trades = trades[trades["year"] != top_year]
        if not excl_trades.empty:
            excl_net = (1 + excl_trades["net_return"]).prod() - 1
            print(f"최대 기여 연도({top_year}) 제외 시 net: {excl_net:.1%} (원래 {total_net:.1%})")

    return {
        "n_trades": len(trades), "win_rate": win_rate,
        "total_net": total_net, "total_gross": total_gross,
        "bh_return": bh_return, "top_year_share": top_year_share,
    }


def run_threshold_sweep(df: pd.DataFrame, feature_cols: list, thresholds: list,
                         bh_window: tuple | None = None) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        print(f"\n--- threshold={threshold} ---")
        trades = generate_trades(df, feature_cols=feature_cols, threshold=threshold)
        result = summarize(trades, df, bh_window=bh_window)
        result["threshold"] = threshold
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"=== BASE ({TICKER_KRX}, {CONFIG_LABEL}) ===")
    df = load_dataset(TICKER_KRX, combined=False)
    print(f"데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})\n")
    bh_window_base = get_fixed_bh_window(df)
    trades = generate_trades(df, feature_cols=FEATURE_COLS_BASE)
    summarize(trades, df, bh_window=bh_window_base)

    print("\n\n=== COMBINED (BASE + VALUATION) -- threshold 스윕 ===")
    df_combined = load_dataset(TICKER_KRX, combined=True)
    print(f"데이터: {df_combined.shape[0]}행 "
          f"({df_combined.index.min().date()} ~ {df_combined.index.max().date()})")
    bh_window_combined = get_fixed_bh_window(df_combined)
    print(f"고정 Buy&Hold 비교 구간(COMBINED): {bh_window_combined[0].date()} ~ {bh_window_combined[1].date()}")

    THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
    sweep = run_threshold_sweep(df_combined, FEATURE_COLS_BASE + FEATURE_COLS_VALUATION, THRESHOLDS,
                                 bh_window=bh_window_combined)

    print("\n\n=== threshold 스윕 요약 (COMBINED) ===")
    print(sweep[["threshold", "n_trades", "win_rate", "total_net", "total_gross", "bh_return"]]
          .round(4).to_string(index=False))
    print("\n판정: net이 threshold를 높일수록 개선되면 '거래비용 과다'가 진짜 원인이었다는 뜻.")
    print("거래 수가 너무 적어지면(예: 10건 미만) 통계적으로 믿기 어려우니 그 지점도 같이 볼 것.")

    # ------------------------------------------------------------------
    # 여기서 끝내지 않고, 가장 성과 좋았던 threshold=0.65를 5-seed로 재검증.
    # 거래 수가 95~113건으로, 지난번 대한제강 meta-labeling에서 seed=42만
    # 우연히 좋았던 사례(120건)와 비슷한 규모라 반드시 거쳐야 하는 단계.
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("=== 검증: COMBINED threshold=0.65의 5-seed 재현성 ===")
    print("=" * 60)
    SEEDS = [42, 1, 7, 123, 2024]
    seed_results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        trades_seed = generate_trades(
            df_combined, feature_cols=FEATURE_COLS_BASE + FEATURE_COLS_VALUATION,
            threshold=0.65, random_state=seed,
        )
        r = summarize(trades_seed, df_combined, show_top_year_exclusion=(seed == 42),
                      bh_window=bh_window_combined)
        r["seed"] = seed
        seed_results.append(r)

    seed_df = pd.DataFrame(seed_results)
    print("\n--- 5-seed 요약 ---")
    print(seed_df[["seed", "n_trades", "total_net", "bh_return"]].round(4).to_string(index=False))
    win_count = (seed_df["total_net"] > seed_df["bh_return"]).sum()
    print(f"\n5개 시드 중 Buy & Hold를 이긴 시드: {win_count}/5")
    print("판정: 5/5 또는 4/5 이겨야 '진짜' -- 대한제강 meta-labeling 사례(2/5 -> 0/5로 무너짐)를")
    print("기억할 것. 여기서도 무너지면 같은 패턴(소표본 우연)이 종목만 바뀌어 반복된 셈.")

    # ------------------------------------------------------------------
    # [수정] 처음엔 2025년 "행만" 빼고 2018~2026을 이어붙였는데, 이러면 Buy & Hold가
    # 여전히 진입일(2018)~청산일(2026) 종가 차이로 계산되면서 2025년의 실제 가격
    # 상승분을 그대로 반영해버림 (행을 지운다고 과거 사실인 주가 이력 자체가 바뀌지
    # 않으므로). 전략은 2025년에 거래를 못 하게 막혔는데 Buy & Hold는 2025년 이득을
    # 그대로 챙기는 불공정한 비교가 됐던 것 -- 실제로 bh_return이 672~720%로 원래
    # (691.6%)와 거의 그대로 나온 게 그 증거.
    #
    # 올바른 방법: 중간에서 빼는 게 아니라 2024년 말에서 데이터 자체를 잘라내서,
    # 전략과 Buy & Hold 둘 다 "2025년이 오기 전"까지만 보게 만듦.
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("=== 최종 검증: 2025년 이전으로 데이터 자체를 잘라서 5-seed 재검증 ===")
    print("=" * 60)
    df_before_2025 = df_combined[df_combined.index < "2025-01-01"]
    print(f"2025년 이전 데이터: {df_before_2025.shape[0]}행 (원래 {df_combined.shape[0]}행)")
    bh_window_before_2025 = get_fixed_bh_window(df_before_2025)
    print(f"고정 Buy&Hold 비교 구간(2025년 이전): {bh_window_before_2025[0].date()} ~ "
          f"{bh_window_before_2025[1].date()}")

    excl_results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} (2025년 이전까지만) ---")
        trades_excl = generate_trades(
            df_before_2025, feature_cols=FEATURE_COLS_BASE + FEATURE_COLS_VALUATION,
            threshold=0.65, random_state=seed,
        )
        r = summarize(trades_excl, df_before_2025, bh_window=bh_window_before_2025)
        r["seed"] = seed
        excl_results.append(r)

    excl_df = pd.DataFrame(excl_results)
    print("\n--- 2025년 이전 5-seed 요약 ---")
    print(excl_df[["seed", "n_trades", "total_net", "bh_return"]].round(4).to_string(index=False))
    excl_win_count = (excl_df["total_net"] > excl_df["bh_return"]).sum()
    print(f"\n2025년 이전 기준 5개 시드 중 Buy & Hold를 이긴 시드: {excl_win_count}/5")
    print("최종 판정: 여기서도 4~5/5로 이기면 2025는 '보너스'였을 뿐 진짜 edge -- 반대로")
    print("여기서 무너지면(0~2/5) 원래 결과 전체가 2025년 국면 하나에 기댄 것으로 결론.")