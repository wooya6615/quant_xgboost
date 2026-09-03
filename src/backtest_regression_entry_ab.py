"""
회귀(ret_tb) 예측 기반 백테스트 -- 진입 방식 A(threshold) vs B(top-N%) 비교.

train_xgboost_regression_ablation.py에서 IC까지는 3종목 x 2 objective 전부
5/5 통과했지만, 예측력(IC)과 실제 거래비용 반영 손익은 별개 질문 -- 이 스크립트가
그 다음 단계. objective는 이전 ablation에서 reg:squarederror/reg:pseudohubererror
IC가 사실상 동일했으므로(064350: 0.109/0.107, 052690: 0.125/0.129, 118990:
0.127/0.120) 여기선 reg:squarederror로 고정하고 진입 방식(A vs B) 차이에 집중.

두 진입 방식:
    A. threshold: 예측 수익률(pred) 절대값이 후보값 이상이면 진입
       (분류의 proba>=threshold와 같은 논리, 후보: 0.00/0.01/0.02/0.03/0.05)
    B. top-N%: 그 fold(60일) 안에서 pred 상위 N%에 해당하는 날만 진입
       -- IC가 순위상관이었으므로 이 방식이 지표와 논리적으로 더 맞음
       (후보: 상위 10/20/30/40/50%, fold별로 percentile을 동적으로 계산)

두 방식 다 신호 발생 후 처리는 기존 분류 백테스트와 동일: holding_rows_tb만큼
겹치는 신호는 건너뛰고, 거래비용(0.2% 왕복)을 뺀 net_return을 복리로 이어붙여
Buy & Hold와 비교.

fold별 (test_positions, pred)는 seed당 1번만 계산해서 캐싱 -- 후보별로 모델을
재학습하지 않고 캐싱된 예측값에서 진입일만 다르게 골라냄. 이 캐싱 자체는
안전함(threshold/top-pct 둘 다 학습에 영향을 주지 않는 거래 추출 단계 로직이라
pred 값이 후보에 따라 달라지지 않음).

[중요, 수정 이력] 방식 B(top-N%) 최초 버전은 fold 테스트 구간(60일) 전체의
예측값으로 percentile 커트라인을 계산했는데, 이러면 fold 첫날 진입 여부를
판단할 때 아직 오지 않은 미래 날짜의 예측값까지 커트라인에 반영되는 룩어헤드가
생김. 그 시점까지 '이미 나온' 예측값만으로 누적(expanding) percentile을
계산하도록 수정 -- fold 초반 min_lookback(기본 5)일은 비교 대상이 부족해
신호를 꺼둠(정상 동작).

사용법 (레포 루트에서):
    python src/backtest_regression_entry_ab.py

전제:
    feature_engineering_triple_barrier.py와 같은 폴더(src/)에 있어야 함.
    데이터(pt2sl1_nd20_hl_base.csv, 3종목)는 train_xgboost_regression_
    ablation.py 실행 시 이미 생성됐을 것.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = {
    "064350": "064350.KS",
    "052690": "052690.KS",
    "118990": "118990.KQ",
}
PT_SL = (2, 1)
NUM_DAYS = 20
OBJECTIVE = "reg:squarederror"
SEEDS = [42, 1, 7, 123, 2024]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
ROUND_TRIP_COST = 0.002

THRESHOLD_CANDIDATES = [0.00, 0.01, 0.02, 0.03, 0.05]  # 방식 A
TOP_PCT_CANDIDATES = [10, 20, 30, 40, 50]               # 방식 B (상위 N%)


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def load_or_build_dataset(ticker_krx: str, ticker: str) -> pd.DataFrame:
    out_path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{config_label()}_base.csv"
    if out_path.exists():
        df = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
    else:
        print(f"[{ticker_krx}] 데이터 생성 중...")
        df = build_triple_barrier_dataset(ticker=ticker, pt_sl=PT_SL, num_days=NUM_DAYS)
        df.to_csv(out_path)
        print(f"  저장 완료: {out_path.name}")
    df = df.dropna(subset=FEATURE_COLS_BASE + ["ret_tb"])
    return df


def buy_and_hold_return(df: pd.DataFrame) -> float:
    return df["Close"].iloc[-1] / df["Close"].iloc[0] - 1


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


def compute_fold_predictions(df: pd.DataFrame, seed: int) -> list:
    """seed당 1번만 학습 -- fold별 (test_positions, pred) 리스트로 반환. 재학습 없이 A/B 둘 다 재사용."""
    X = df[FEATURE_COLS_BASE]
    y = df["ret_tb"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    fold_results = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        model = xgb.XGBRegressor(
            objective=OBJECTIVE,
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        test_positions = list(test_idx)
        pred = model.predict(X.iloc[test_positions])
        fold_results.append((test_positions, pred))

    return fold_results


def generate_trades_threshold(df: pd.DataFrame, fold_results: list, threshold: float) -> pd.DataFrame:
    """방식 A -- pred가 고정 threshold 이상이면 진입. threshold는 데이터와 무관한 상수라
    미래 정보를 전혀 안 쓰므로 룩어헤드 걱정 없음."""
    trades = []
    for test_positions, pred in fold_results:
        i = 0
        while i < len(test_positions):
            if pred[i] >= threshold:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1
    return pd.DataFrame(trades)


def generate_trades_top_pct(df: pd.DataFrame, fold_results: list, top_pct: float,
                             min_lookback: int = 5) -> pd.DataFrame:
    """
    방식 B -- fold 안에서 상위 N% 진입. [주의] percentile 커트라인은 그 시점까지
    '이미 나온' 예측값(pred[:i+1])만으로 누적 계산 -- fold 전체(pred 전체)로 계산하면
    아직 오지 않은 미래 날짜의 예측값까지 커트라인에 반영되는 룩어헤드가 생김.
    fold 첫 min_lookback일은 비교 대상이 너무 적어 순위가 불안정하므로 신호를 끔
    (매 fold 초반 며칠은 거래 없음 -- 정상 동작).
    """
    trades = []
    for test_positions, pred in fold_results:
        i = 0
        while i < len(test_positions):
            if i >= min_lookback - 1:
                seen_so_far = pred[:i + 1]
                cutoff = np.percentile(seen_so_far, 100 - top_pct)
                is_signal = pred[i] >= cutoff
            else:
                is_signal = False

            if is_signal:
                row_idx = test_positions[i]
                entry_date = df.index[row_idx]
                gross_return = df["ret_tb"].iloc[row_idx]
                holding = max(int(df["holding_rows_tb"].iloc[row_idx]), 1)

                if pd.notna(gross_return):
                    trades.append({
                        "entry_date": entry_date,
                        "net_return": gross_return - ROUND_TRIP_COST,
                    })
                i += holding
            else:
                i += 1
    return pd.DataFrame(trades)


if __name__ == "__main__":
    rows = []

    for ticker_krx, ticker in TICKERS.items():
        df = load_or_build_dataset(ticker_krx, ticker)
        bh_full = buy_and_hold_return(df)
        print(f"\n{'#' * 60}\n# {ticker_krx} -- {df.shape[0]}행, Buy&Hold={bh_full:+.2%}\n{'#' * 60}")

        # seed별 fold 예측값을 먼저 전부 캐싱 (재학습 없이 A/B 후보 전부 재사용)
        seed_fold_results = {seed: compute_fold_predictions(df, seed) for seed in SEEDS}

        # 방식 A: threshold
        for threshold in THRESHOLD_CANDIDATES:
            for seed in SEEDS:
                trades = generate_trades_threshold(df, seed_fold_results[seed], threshold)
                net_total = (1 + trades["net_return"]).prod() - 1 if not trades.empty else np.nan
                rows.append({
                    "ticker": ticker_krx, "method": "A_threshold", "candidate": threshold,
                    "seed": seed, "n_trades": len(trades), "net_total_return": net_total,
                    "bh_return": bh_full, "beats_bh": (net_total > bh_full) if pd.notna(net_total) else False,
                })

        # 방식 B: top-N% (fold 안에서 그 시점까지의 누적 percentile로 동적 계산 -- 룩어헤드 방지)
        for top_pct in TOP_PCT_CANDIDATES:
            for seed in SEEDS:
                trades = generate_trades_top_pct(df, seed_fold_results[seed], top_pct)
                net_total = (1 + trades["net_return"]).prod() - 1 if not trades.empty else np.nan
                rows.append({
                    "ticker": ticker_krx, "method": "B_top_pct", "candidate": top_pct,
                    "seed": seed, "n_trades": len(trades), "net_total_return": net_total,
                    "bh_return": bh_full, "beats_bh": (net_total > bh_full) if pd.notna(net_total) else False,
                })

    result_df = pd.DataFrame(rows)

    print(f"\n{'=' * 60}\n=== 종목 x 방식 x 후보별 5-seed 요약 ===\n{'=' * 60}")
    summary = result_df.groupby(["ticker", "method", "candidate"]).agg(
        win_seeds=("beats_bh", "sum"),
        mean_net_return=("net_total_return", "mean"),
        mean_n_trades=("n_trades", "mean"),
    ).reset_index()
    print(summary.round(4).to_string(index=False))

    print(f"\n{'=' * 60}")
    print("[비교 가이드]")
    print("- 같은 종목 안에서 A(threshold)와 B(top-N%) 중 어느 쪽이 5/5 통과 + 절대수익률")
    print("  더 좋은 후보를 내는지 비교")
    print("- IC는 순위상관이었으므로 이론상 B가 더 잘 맞아야 함 -- A가 더 낫게 나오면")
    print("  왜 그런지(예: 절대 크기 자체에도 정보가 있다는 뜻인지) 추가로 볼 것")
    print("- 052690/118990이 여기서도 5/5를 통과하면: 분류에서 실패했던 게 magnitude")
    print("  정보를 버렸기 때문이라는 가설이 힘을 얻음. 여전히 실패하면: IC 단계에서")
    print("  봤던 신호가 라벨 창 겹침 등으로 인한 통계적 착시였을 가능성이 높아짐")
    print("\n판정 기준: win_seeds==5 -> 통과 후보. bh_return이 매우 낮은 종목은")
    print("  118990처럼 벤치마크 붕괴 착시 가능성 있으니 절대수익률도 같이 볼 것")