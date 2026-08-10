"""
Triple-barrier 라벨 버전 BASE feature 데이터셋 생성 (현대로템 064350, BASE, h=20 기준).

기존 feature_engineering.py의 add_label()(고정 horizon 이진분류)은 비교 기준선으로
그대로 남겨두고, labeling_triple_barrier.py의 triple-barrier 방식으로 새 라벨을
추가로 만듦. feature 계산 로직(모멘텀/변동성/거래량/상대강도)은 feature_engineering.py를
그대로 재사용 -- "라벨 정의만 바꿨을 때 성능이 달라지는가"를 순수하게 보기 위해
feature/모델은 최대한 기존과 동일하게 유지 (quant_ranking_kr의 base_features.py가
"feature는 기존과 동일 유지" 원칙을 따른 것과 같은 이유).

산출 컬럼:
    label_fixed      기존 고정 horizon=20 이진 라벨 (비교 기준선)
    label_tb         triple-barrier 라벨 (-1/0/1)
    label_tb_binary  label_tb > 0 -> 1, 아니면 0 (기존 이진분류 파이프라인과 바로 비교 가능)
    ret_tb           triple-barrier 실현수익률 (가변 보유기간 -- future_return과 다름)

사용법 (레포 루트에서):
    python src/feature_engineering_triple_barrier.py

전제:
    feature_engineering.py, labeling_triple_barrier.py와 같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features, add_label,
)
from feature_engineering_valuation import load_valuation, add_valuation_features
from labeling_triple_barrier import (
    get_daily_volatility, get_vertical_barrier, apply_triple_barrier, get_bins,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]


def build_triple_barrier_dataset(
    ticker: str = "064350.KS",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon_fixed: int = 20,        # 기존 검증된 현대로템 BASE 설정과 동일
    cost_threshold: float = 0.012,  # 기존 h=20 실험 임계값과 동일 (DEFAULT_HORIZON_COST_MAP[20])
    pt_sl: tuple = (1, 1),          # 익절/손절 배수 -- 대칭 배리어로 시작 (추후 스윕 가능)
    vol_span: int = 20,
    num_days: int = 20,             # 수직 배리어(최대 보유기간) -- horizon_fixed와 동일하게 맞춰 비교
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    # 기존 고정 horizon 라벨 (비교 기준선)
    df = add_label(df, horizon=horizon_fixed, cost_threshold=cost_threshold)
    df = df.rename(columns={"label": "label_fixed"})

    # triple-barrier 라벨
    close = df["Close"]
    daily_vol = get_daily_volatility(close, span=vol_span)
    # [수정] 일간 변동성을 그대로 배리어 폭으로 쓰면 num_days(20일) 보유기간 대비
    # 배리어가 지나치게 타이트해짐 -- 20일 누적 변동성은 대략 sqrt(20)배로 커지므로,
    # 배리어를 "20일치 예상 변동폭" 기준으로 스케일링해야 실제로 num_days만큼 보유하는
    # 라벨이 나옴. 이걸 빠뜨렸더니 배리어가 하루이틀 안에 거의 다 터져서(label_tb=0 비율
    # 0.3%), BASE feature(중기 모멘텀)로는 예측 불가능한 "내일 방향" 라벨이 돼버렸었음
    # (AUC 0.50, 완전히 랜덤).
    vol = daily_vol * np.sqrt(num_days)
    t_events = close.index[vol_span:]  # 변동성 워밍업 구간 제외
    t1 = get_vertical_barrier(close, t_events, num_days=num_days)
    events = apply_triple_barrier(close, t_events, pt_sl=pt_sl, target=vol, t1=t1)
    bins = get_bins(events, close)

    df["label_tb"] = bins["label"]
    df["ret_tb"] = bins["ret"]
    # 백테스트에서 "이 거래를 며칠 보유했는지"로 다음 진입 시점을 정하는 데 씀
    # (fixed horizon과 달리 배리어 도달 시점이 거래마다 다름 -- 가변 보유기간)
    touch_pos = close.index.get_indexer(bins["touch_time"])
    entry_pos = close.index.get_indexer(bins.index)
    df["holding_rows_tb"] = pd.Series(touch_pos - entry_pos, index=bins.index)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_BASE,
        "future_return", "label_fixed",
        "label_tb", "ret_tb", "holding_rows_tb",
    ]

    result = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # 두 라벨 다 유효한 행만 남김 (공정 비교를 위해 같은 행 집합 사용)
    result = result.dropna(subset=FEATURE_COLS_BASE + ["label_fixed", "label_tb"])
    return result


def build_triple_barrier_dataset_combined(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon_fixed: int = 20,
    cost_threshold: float = 0.012,
    pt_sl: tuple = (2, 1),
    vol_span: int = 20,
    num_days: int = 20,
) -> pd.DataFrame:
    """
    BASE + VALUATION(PER/PBR/DIV, z-score 포함) feature까지 합친 triple-barrier 데이터셋.
    현대로템은 기존 밸류에이션 ablation에서 h=20 기준 COMBINED-BASE AUC가 단조증가하며
    5/5 시드로 개선됐고, 2025년(이례적 강세장) 제외해도 백테스트 우위가 재현됐던 조합
    (PROJECT_SUMMARY.md 2-3절 참고, 정확한 AUC diff 수치는 대한제강과 달리 기록 안 됨).
    feature_engineering_valuation.py의 load_valuation()/add_valuation_features()를 그대로 재사용.
    """
    df, bench = load_data(ticker, benchmark, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    df = add_label(df, horizon=horizon_fixed, cost_threshold=cost_threshold)
    df = df.rename(columns={"label": "label_fixed"})

    valuation_df = load_valuation(ticker_krx, start, end)
    df = add_valuation_features(df, valuation_df)

    close = df["Close"]
    daily_vol = get_daily_volatility(close, span=vol_span)
    vol = daily_vol * np.sqrt(num_days)  # build_triple_barrier_dataset()과 동일한 스케일링
    t_events = close.index[vol_span:]
    t1 = get_vertical_barrier(close, t_events, num_days=num_days)
    events = apply_triple_barrier(close, t_events, pt_sl=pt_sl, target=vol, t1=t1)
    bins = get_bins(events, close)

    df["label_tb"] = bins["label"]
    df["ret_tb"] = bins["ret"]
    touch_pos = close.index.get_indexer(bins["touch_time"])
    entry_pos = close.index.get_indexer(bins.index)
    df["holding_rows_tb"] = pd.Series(touch_pos - entry_pos, index=bins.index)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_BASE, *FEATURE_COLS_VALUATION,
        "future_return", "label_fixed",
        "label_tb", "ret_tb", "holding_rows_tb",
    ]

    result = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=FEATURE_COLS_BASE + FEATURE_COLS_VALUATION + ["label_fixed", "label_tb"]
    )
    return result


if __name__ == "__main__":
    TICKER = "052690.KS"
    TICKER_KRX = "052690"
    PT_SL = (2, 1)   # 1:2 손익비 -- 손절 대비 익절을 2배 넓게 (현대로템 검증 때와 동일)
    NUM_DAYS = 20    # [원복] 10 -> 20 -- 전략 자체는 안 바꾸고 종목만 바꿔서 깨끗하게 재현성 테스트
    CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}"

    print(f"=== BASE ({TICKER_KRX}, {CONFIG_LABEL}) ===")
    dataset = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
    print(f"생성된 데이터셋 shape: {dataset.shape}")
    print(f"설정: pt_sl={PT_SL}, num_days={NUM_DAYS} (변동성은 daily_vol * sqrt(num_days) 기준)")

    print(f"\n기존 고정 horizon 라벨(label_fixed) 분포:\n{dataset['label_fixed'].value_counts(normalize=True)}")
    tb_dist = dataset['label_tb'].value_counts(normalize=True)
    print(f"\ntriple-barrier 라벨(label_tb) 분포:\n{tb_dist}")
    if tb_dist.get(0.0, 0) < 0.02:
        print("경고: label_tb=0(수직 배리어까지 버틴 경우) 비율이 여전히 2% 미만 -- "
              "배리어가 아직도 너무 타이트할 수 있음. pt_sl을 키워서 재시도 고려.")

    label_tb_binary = (dataset["label_tb"] > 0).astype(int)
    agreement = (dataset["label_fixed"] == label_tb_binary).mean()
    print(f"\n두 라벨의 방향 일치율: {agreement:.1%} (참고용 -- 낮다고 나쁜 건 아님, 정의 자체가 다름)")

    out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}.csv"
    dataset.to_csv(out_path)
    print(f"저장 완료: {out_path}")

    print(f"\n=== COMBINED (BASE + VALUATION), {CONFIG_LABEL} ===")
    dataset_combined = build_triple_barrier_dataset_combined(
        ticker=TICKER, ticker_krx=TICKER_KRX, pt_sl=PT_SL, num_days=NUM_DAYS,
    )
    print(f"생성된 데이터셋 shape: {dataset_combined.shape} "
          f"(BASE 대비 {dataset.shape[0] - dataset_combined.shape[0]}행 감소 -- "
          f"밸류에이션 z-score의 252일 워밍업 구간 때문, 정상)")

    tb_dist_combined = dataset_combined['label_tb'].value_counts(normalize=True)
    print(f"\ntriple-barrier 라벨(label_tb) 분포 (COMBINED):\n{tb_dist_combined}")

    out_path_combined = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
    dataset_combined.to_csv(out_path_combined)
    print(f"저장 완료: {out_path_combined}")