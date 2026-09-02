"""
triple-barrier BASE 전용(밸류에이션 제외) 재검증 -- 3개 앵커 종목.

[배경] PER=0 버그 수정으로 triple-barrier + VALUATION-COMBINED 라인 전체가
무효 처리됨. build_triple_barrier_dataset()은 밸류에이션 코드를 아예 안 타서
그 버그엔 안 걸렸지만, BASE 단독으로 최신 라벨링(D+1 체결지연 + High/Low
장중터치 반영)으로 5-seed/국면검증을 독립적으로 통과한 기록이 없어서 이번에
처음부터 정식으로 재검증함.

전제:
    feature_engineering_triple_barrier.py, labeling_triple_barrier.py,
    feature_engineering.py와 같은 폴더(src/)에 있어야 함.

사용법 (레포 루트에서):
    python src/verify_base_only_triple_barrier.py

산출물:
    data/base_only_verification_summary.csv  -- 종목별 5-seed 요약
    콘솔에 종목별 통과/실패 판정 출력 (docs/prereg_base_only_reverification.md
    기준과 동일)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from feature_engineering_triple_barrier import (
    build_triple_barrier_dataset, FEATURE_COLS_BASE,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ------------------------------------------------------------------
# 사전등록 고정 파라미터 (docs/prereg_base_only_reverification.md 참고 --
# 결과 보고 여기 값을 바꾸지 않을 것)
# ------------------------------------------------------------------
VALIDATED_TICKERS = {
    "064350": "064350.KS",  # 현대로템
    "052690": "052690.KS",  # 한전기술
    "118990": "118990.KQ",  # 모트렉스 (코스닥)
}
PT_SL = (2, 1)
NUM_DAYS = 20
SEEDS = [42, 1, 7, 123, 2024]
TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO = 300, 60, 60, NUM_DAYS
EXCLUDE_YEAR = 2025  # 국면 제외 재검증용 -- 가장 최근 완결 연도 절단


# ------------------------------------------------------------------
# 1. walk-forward 분할 (기존 스크립트들과 동일 로직)
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


def run_walk_forward(df: pd.DataFrame, random_state: int) -> dict:
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]

    splits = walk_forward_splits(len(df), TRAIN_SIZE, TEST_SIZE, STEP, EMBARGO)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요.")

    fold_rows = []
    for train_idx, test_idx in splits:
        X_train, y_train = X.iloc[list(train_idx)], y.iloc[list(train_idx)]
        X_test, y_test = X.iloc[list(test_idx)], y.iloc[list(test_idx)]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=random_state,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        base_rate = max(y_test.mean(), 1 - y_test.mean())
        accuracy = (pred == y_test).mean()

        fold_rows.append({
            "auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy,
            "vs_base_rate": accuracy - base_rate,
        })

    fold_df = pd.DataFrame(fold_rows)
    return {
        "mean_auc": fold_df["auc"].mean(),
        "mean_vs_base_rate": fold_df["vs_base_rate"].mean(),
        "win_folds": int((fold_df["vs_base_rate"] > 0).sum()),
        "n_folds": len(fold_df),
    }


# ------------------------------------------------------------------
# 2. 종목 하나에 대해 5-seed + 국면제외 재검증까지 실행
# ------------------------------------------------------------------
def verify_ticker(ticker_krx: str, ticker: str) -> dict:
    print(f"\n{'#' * 60}\n# {ticker_krx} ({ticker}) -- BASE 전용 재검증\n{'#' * 60}")

    dataset = build_triple_barrier_dataset(ticker=ticker, pt_sl=PT_SL, num_days=NUM_DAYS)
    dataset["label_tb_binary"] = (dataset["label_tb"] > 0).astype(int)
    print(f"전체 데이터: {dataset.shape[0]}행 ({dataset.index.min().date()} ~ {dataset.index.max().date()})")

    # 원본 CSV로도 저장 -- 다음 단계(백테스트)에서 재사용
    out_path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl_base.csv"
    dataset.to_csv(out_path)
    print(f"저장 완료: {out_path.name}")

    # --- 5-seed 전체기간 검증 ---
    full_seed_rows = []
    for seed in SEEDS:
        r = run_walk_forward(dataset, random_state=seed)
        full_seed_rows.append({"seed": seed, **r})
        print(f"  [전체기간] seed={seed}: AUC={r['mean_auc']:.4f}, "
              f"vs_base_rate={r['mean_vs_base_rate']:+.4f} ({r['win_folds']}/{r['n_folds']} fold 승)")

    full_df = pd.DataFrame(full_seed_rows)
    aucs = full_df["mean_auc"].values
    vs_base = full_df["mean_vs_base_rate"].values
    win_seeds_full = int((vs_base > 0).sum())
    std_mean_ratio = aucs.std() / aucs.mean() if aucs.mean() != 0 else float("inf")

    print(f"  전체기간 요약: AUC 평균={aucs.mean():.4f}, std/mean={std_mean_ratio:.1%}, "
          f"vs_base_rate 5/5 양수={win_seeds_full}/5")

    # --- 국면 제외 재검증 (2025년 절단) ---
    cutoff_dataset = dataset[dataset.index.year < EXCLUDE_YEAR]
    print(f"\n  [{EXCLUDE_YEAR}년 제외] {cutoff_dataset.shape[0]}행 "
          f"({cutoff_dataset.index.min().date()} ~ {cutoff_dataset.index.max().date()})")

    excl_seed_rows = []
    for seed in SEEDS:
        try:
            r = run_walk_forward(cutoff_dataset, random_state=seed)
        except ValueError as e:
            print(f"  [{EXCLUDE_YEAR}년 제외] seed={seed}: 건너뜀 ({e})")
            continue
        excl_seed_rows.append({"seed": seed, **r})
        print(f"  [{EXCLUDE_YEAR}년 제외] seed={seed}: AUC={r['mean_auc']:.4f}, "
              f"vs_base_rate={r['mean_vs_base_rate']:+.4f} ({r['win_folds']}/{r['n_folds']} fold 승)")

    excl_df = pd.DataFrame(excl_seed_rows) if excl_seed_rows else pd.DataFrame()
    win_seeds_excl = int((excl_df["mean_vs_base_rate"] > 0).sum()) if not excl_df.empty else 0

    # --- 판정 (사전등록 기준 그대로) ---
    passed_5seed = (win_seeds_full == 5) and (std_mean_ratio < 0.5)
    passed_regime = win_seeds_excl >= 4  # 기존 컨벤션(4/5 이상)과 동일 기준 적용
    verdict = "통과" if (passed_5seed and passed_regime) else "[실패]"

    print(f"\n  === {ticker_krx} 최종 판정: {verdict} === "
          f"(5-seed: {'통과' if passed_5seed else '실패'}, "
          f"국면제외: {'통과' if passed_regime else '실패'})")

    return {
        "ticker_krx": ticker_krx,
        "n_rows_full": dataset.shape[0],
        "auc_mean_full": aucs.mean(),
        "auc_std_mean_ratio": std_mean_ratio,
        "win_seeds_full": win_seeds_full,
        "n_rows_excl_2025": cutoff_dataset.shape[0],
        "win_seeds_excl_2025": win_seeds_excl,
        "passed_5seed": passed_5seed,
        "passed_regime": passed_regime,
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = [verify_ticker(krx, tk) for krx, tk in VALIDATED_TICKERS.items()]
    summary = pd.DataFrame(results)

    out_path = DATA_DIR / "base_only_verification_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"\n{'=' * 60}\n=== 전체 요약 ===\n{'=' * 60}")
    print(summary.to_string(index=False))
    print(f"\n저장 완료: {out_path}")

    n_passed = (summary["verdict"] == "통과").sum()
    print(f"\n{n_passed}/{len(summary)} 종목 통과.")
    if n_passed == len(summary):
        print("-> 3종목 다 통과: triple-barrier 프레임워크 유지, 다음 단계(백테스트)로 진행.")
    elif n_passed == 0:
        print("-> 전부 실패: triple-barrier 프레임워크 자체를 재검토할 시점 "
              "(PROJECT_SUMMARY.md 다음 단계 후보 2번 참고).")
    else:
        print("-> 일부만 통과: 통과 종목만 남기고 공통 조건 분석으로 진행.")