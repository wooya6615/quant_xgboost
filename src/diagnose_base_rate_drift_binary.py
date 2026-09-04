"""
[진단] 우리가 lag/GRU/TCN 실험에서 계속 쓴 단순 이진 라벨(horizon일 후 수익률 >
cost_threshold)에도, triple-barrier 재검증에서 발견된 것과 같은 fold별
test_base_rate 드리프트가 있는지 확인.

배경: diagnose_base_rate_drift.py(triple-barrier 라벨 기준)에서 AUC(0.52~0.55,
안정적)와 vs_base_rate(마이너스, 안정적)가 모순되는 걸 발견함 -- walk-forward의
각 fold(test_size=60, 약 3개월)마다 그 시기가 상승장/하락장이었냐에 따라
test_base_rate 자체가 크게 흔들려서, "다수 클래스 비율을 이기는가"라는 기준이
불공정해졌기 때문. 우리가 lag/GRU/TCN(3종목, 50종목)에서 낸 [실패] 판정도 전부
같은 패턴(AUC는 랜덤보다 살짝 위로 안정적, vs_base_rate만 계속 마이너스)이라
같은 문제일 가능성이 큼 -- 이 스크립트로 확인함.

⚠️ 사전 등록: horizon=10, cost_threshold=0.005 (이번 대화의 lag/GRU/TCN 실험과
동일 기준). embargo도 triple-barrier 쪽의 고정값 20이 아니라 우리 관행대로
horizon(=10)으로 둠 -- 우리가 실제로 썼던 walk-forward 구성 그대로 재현해야
드리프트 여부가 우리 결과에 그대로 적용되는지 확인 가능.

사용법 (레포 루트에서):
    python src/diagnose_base_rate_drift_binary.py
"""

import pandas as pd

from feature_engineering import build_feature_dataset

TICKER = "064350.KS"
TICKER_KRX = "064350"
HORIZON = 10
COST_THRESHOLD = 0.005
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, HORIZON


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


def diagnose(ticker: str = TICKER, ticker_krx: str = TICKER_KRX) -> pd.DataFrame:
    df = build_feature_dataset(ticker=ticker, horizon=HORIZON, cost_threshold=COST_THRESHOLD)
    df = df.sort_index()

    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)

    rows = []
    for i, (train_idx, test_idx) in enumerate(splits):
        y_train = df["label"].iloc[list(train_idx)]
        y_test = df["label"].iloc[list(test_idx)]
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
    print(f"{'#' * 60}\n# {ticker_krx} (horizon={HORIZON}, cost_threshold={COST_THRESHOLD})\n{'#' * 60}")
    print(fold_df.to_string(index=False))
    print(f"\ndrift 절대값 평균: {fold_df['drift'].abs().mean():.3f}")
    print(f"|drift| > 0.15인 fold 수: {(fold_df['drift'].abs() > 0.15).sum()} / {len(fold_df)}")
    print(f"test_base_rate 자체의 표준편차(fold 간): {fold_df['test_base_rate'].std():.3f} "
          f"-- 이게 크면(예: >0.1) 라벨 비율이 시기별로 크게 출렁인다는 뜻")
    print(f"test_base_rate 범위: [{fold_df['test_base_rate'].min():.3f}, {fold_df['test_base_rate'].max():.3f}]")

    return fold_df


if __name__ == "__main__":
    fold_df = diagnose()

    print("\n[해석 가이드]")
    print("- |drift| 평균이나 test_base_rate std가 크면(대략 0.1 이상): AUC가 랜덤보다")
    print("  높은데 vs_base_rate가 마이너스인 이유가 '신호 없음'이 아니라 fold 간 라벨")
    print("  비율 자체의 non-stationarity 문제일 가능성이 높음.")
    print("  -> 다음 단계: lag/GRU/TCN 결과를 vs_base_rate 대신 실제 백테스트")
    print("     (거래비용 반영, Buy & Hold 대비)로 재판정할 것.")
    print("- drift가 작은데도 vs_base_rate가 계속 마이너스면: 진짜 신호 없음으로 봐도 됨")
    print("  (지금까지 낸 [실패] 판정들 유지)")