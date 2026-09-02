"""
[진단] triple-barrier BASE 재검증에서 AUC(0.52~0.55, 안정적)와 vs_base_rate
(-13~-15%p, 역시 안정적)가 서로 모순되는 패턴이 나온 원인 확인용.

가설: walk-forward의 각 fold에서 train 구간과 test 구간의 label_tb_binary
base_rate(다수 클래스 비율)가 서로 크게 다르면, 모델이 train 시점 확률분포로
학습한 걸 고정 0.5 임계값으로 test에 적용할 때 랭킹(AUC)은 안 죽어도 accuracy는
무너질 수 있음. 이게 맞다면 "신호 없음"이 아니라 "라벨의 시간적 비정상성
(non-stationarity)" 문제로 재분류해야 함.

전제:
    verify_base_only_triple_barrier.py를 먼저 돌려서
    data/{ticker}_features_triple_barrier_pt2sl1_nd20_hl_base.csv가 있어야 함.

사용법 (레포 루트에서):
    python src/diagnose_base_rate_drift.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = ["064350", "052690", "118990"]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, 20


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


def diagnose_ticker(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)

    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    rows = []
    for i, (train_idx, test_idx) in enumerate(splits):
        y_train = df["label_tb_binary"].iloc[list(train_idx)]
        y_test = df["label_tb_binary"].iloc[list(test_idx)]
        train_start, train_end = df.index[list(train_idx)[0]], df.index[list(train_idx)[-1]]
        test_start, test_end = df.index[list(test_idx)[0]], df.index[list(test_idx)[-1]]

        rows.append({
            "fold": i,
            "train_period": f"{train_start.date()}~{train_end.date()}",
            "test_period": f"{test_start.date()}~{test_end.date()}",
            "train_base_rate": y_train.mean(),
            "test_base_rate": y_test.mean(),
            "drift": y_test.mean() - y_train.mean(),
        })

    fold_df = pd.DataFrame(rows)
    print(f"\n{'#' * 60}\n# {ticker_krx}\n{'#' * 60}")
    print(fold_df.to_string(index=False))
    print(f"\ndrift 절대값 평균: {fold_df['drift'].abs().mean():.3f}")
    print(f"|drift| > 0.15인 fold 수: {(fold_df['drift'].abs() > 0.15).sum()} / {len(fold_df)}")
    print(f"test_base_rate 자체의 표준편차(fold 간): {fold_df['test_base_rate'].std():.3f} "
          f"-- 이게 크면(예: >0.1) 라벨 비율이 시기별로 크게 출렁인다는 뜻")

    return fold_df


if __name__ == "__main__":
    all_results = {}
    for ticker_krx in TICKERS:
        all_results[ticker_krx] = diagnose_ticker(ticker_krx)

    print(f"\n{'=' * 60}\n=== 종목별 요약 ===\n{'=' * 60}")
    for ticker_krx, fold_df in all_results.items():
        print(f"{ticker_krx}: |drift| 평균={fold_df['drift'].abs().mean():.3f}, "
              f"test_base_rate std={fold_df['test_base_rate'].std():.3f}, "
              f"test_base_rate 범위=[{fold_df['test_base_rate'].min():.3f}, "
              f"{fold_df['test_base_rate'].max():.3f}]")

    print("\n[해석 가이드]")
    print("- |drift| 평균이나 test_base_rate std가 크면(대략 0.1 이상): AUC>0.5인데")
    print("  vs_base_rate가 크게 마이너스인 이유가 '신호 없음'이 아니라 fold 간 라벨")
    print("  비율 자체가 안 흔들리는(non-stationary) 문제일 가능성이 높음.")
    print("  -> 다음 단계: 고정 0.5 임계값 대신 fold별 test 중앙값 임계값으로 다시")
    print("     accuracy를 계산해서, 그래도 base_rate를 못 이기는지 재확인할 것.")
    print("- drift가 작은데도 vs_base_rate가 계속 마이너스면: 진짜 신호 없음으로 봐도 됨")
    print("  (지금 낸 [실패] 판정 유지)")