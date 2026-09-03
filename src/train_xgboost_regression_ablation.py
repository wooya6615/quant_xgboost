"""
분류(label_tb_binary) -> 회귀(ret_tb) 전환 실험.

사전등록: docs/prereg_classification_to_regression.md 참고.

라벨 타입만 바꾼 효과를 순수하게 보기 위해, feature/종목/walk-forward 프레임/
pt_sl/num_days는 3종목 공통으로 이미 검증됐던 값(pt_sl=(2,1), num_days=20)
그대로 고정 -- 064350의 최종 채택값(num_days=30)은 052690/118990과 조건을
맞추기 위해 여기선 쓰지 않음.

평가지표: 기존 AUC/vs_base_rate 자리에 fold별 Spearman IC(예측값과 실현
ret_tb의 순위상관)를 씀. objective는 reg:squarederror/reg:pseudohubererror
둘 다 비교 (Huber가 이상치에 덜 흔들리는지 확인).

기존 분류 결과(참고용, 재계산 안 함):
    064350: BASE 재검증 통과 (5-seed 5/5, PBO 검증까지 완료)
    052690: BASE 재검증 [실패] (전체기간부터 0/5)
    118990: BASE 재검증 [실패] (2020년 제외 시 절대수익률 부호조차 불안정)

사용법 (레포 루트에서):
    python src/train_xgboost_regression_ablation.py

전제:
    feature_engineering_triple_barrier.py와 같은 폴더(src/)에 있어야 함.
    데이터(pt2sl1_nd20_hl_base.csv, 3종목)는 이미 있을 것이고, 없으면
    이 스크립트가 알아서 생성함.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from feature_engineering_triple_barrier import build_triple_barrier_dataset, FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = {
    "064350": "064350.KS",
    "052690": "052690.KS",
    "118990": "118990.KQ",
}
PT_SL = (2, 1)
NUM_DAYS = 20
OBJECTIVES = ["reg:squarederror", "reg:pseudohubererror"]
SEEDS = [42, 1, 7, 123, 2024]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS


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
    # ret_tb가 NaN인 행(수직 배리어 미결 등)은 회귀 타겟 자체가 없으므로 제외
    df = df.dropna(subset=FEATURE_COLS_BASE + ["ret_tb"])
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


def run_regression_walk_forward(df: pd.DataFrame, objective: str, seed: int) -> list:
    """fold별 Spearman IC 리스트 반환 (예측값 vs 실현 ret_tb)."""
    X = df[FEATURE_COLS_BASE]
    y = df["ret_tb"]
    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    fold_ics = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        model = xgb.XGBRegressor(
            objective=objective,
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        if len(set(pred)) < 2 or y_test.nunique() < 2:
            continue
        ic, _ = spearmanr(pred, y_test)
        if pd.notna(ic):
            fold_ics.append(ic)

    return fold_ics


if __name__ == "__main__":
    all_rows = []

    for ticker_krx, ticker in TICKERS.items():
        df = load_or_build_dataset(ticker_krx, ticker)
        print(f"\n{'#' * 60}\n# {ticker_krx} -- {df.shape[0]}행\n{'#' * 60}")

        for objective in OBJECTIVES:
            for seed in SEEDS:
                fold_ics = run_regression_walk_forward(df, objective, seed)
                if not fold_ics:
                    mean_ic = np.nan
                else:
                    mean_ic = float(np.mean(fold_ics))
                all_rows.append({
                    "ticker": ticker_krx, "objective": objective, "seed": seed,
                    "n_folds": len(fold_ics), "mean_ic": mean_ic,
                })
                print(f"  [{objective}] seed={seed}: fold {len(fold_ics)}개, mean IC={mean_ic:+.4f}")

    result_df = pd.DataFrame(all_rows)

    print(f"\n{'=' * 60}\n=== 종목 x objective별 5-seed 요약 ===\n{'=' * 60}")
    summary = result_df.groupby(["ticker", "objective"]).agg(
        mean_ic_avg=("mean_ic", "mean"),
        mean_ic_std=("mean_ic", "std"),
        n_positive=("mean_ic", lambda s: (s > 0).sum()),
    ).reset_index()
    summary["std_over_mean"] = (summary["mean_ic_std"] / summary["mean_ic_avg"].abs()).round(3)
    print(summary.round(4).to_string(index=False))

    print(f"\n{'=' * 60}")
    print("[판정 기준] mean_ic_avg가 5-seed 중 4개 이상 양수 & std/mean < 50% -> 통과 후보")
    print("(기존 분류 결과 참고: 064350 통과, 052690/118990 [실패])")
    print("이 결과와 비교해서, 회귀로 바꿨을 때 052690/118990이 그래도 못 살아나면")
    print("'라벨 타입 문제가 아니라 이 종목 자체에 신호가 없다'는 쪽으로 결론 강화됨.")