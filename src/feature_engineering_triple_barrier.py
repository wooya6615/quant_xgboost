"""
Triple-barrier 라벨 버전 BASE feature 데이터셋 생성.

[체결 방식 최종 확정] "D일 종가까지 정보로 신호 -> D+1일 종가 체결"로 확정.
시가체결/하이브리드(종가라벨+시가체결) 등 여러 대안을 비교해본 결과, 시가체결은
배리어 라벨 자체가 더 시끄러워져서(진입일 하루치 장중 노이즈가 라벨에 섞임)
모델 신호 선별력이 떨어짐이 확인됨. 반면 종가체결은 그 노이즈가 원천 배제돼 더
깨끗한 학습 목표가 되고, 종가 동시호가(MOC) 주문으로 실제 체결도 가능한 정당한
가정이라 최종 채택.

[추가 수정] 청산(배리어 터치) 판정에 High/Low 반영. 기존엔 종가만으로 판정해서
장중에 배리어를 건드렸다가 종가에 회복된 경우를 놓쳤음 -- 실제 손절/익절 주문은
가격이 그 선에 닿는 즉시 체결되므로, High/Low로 장중 터치까지 감지하고 청산가도
그날 종가가 아니라 실제 배리어 트리거 가격을 쓰도록 labeling_triple_barrier.py의
apply_triple_barrier()/get_bins()를 확장함. 이 파일로 만든 기존 CSV는 이 개선
전이므로 재생성 필요.

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
from labeling_triple_barrier import build_shifted_barrier_labels

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d", "is_loss"]


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

    # triple-barrier 라벨 (체결 지연 1일 + D+1 종가체결로 확정 + High/Low로 장중 터치 반영)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else None
    low = df["Low"] if "Low" in df.columns else None
    if high is None or low is None:
        print("경고: 데이터에 High/Low 컬럼이 없어서 종가만으로 배리어 터치 판정함 "
              "(장중 터치 놓칠 수 있음)")
    bins = build_shifted_barrier_labels(
        close, vol_span=vol_span, num_days=num_days, pt_sl=pt_sl, high=high, low=low,
    )

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
    high = df["High"] if "High" in df.columns else None
    low = df["Low"] if "Low" in df.columns else None
    if high is None or low is None:
        print("경고: 데이터에 High/Low 컬럼이 없어서 종가만으로 배리어 터치 판정함")
    bins = build_shifted_barrier_labels(
        close, vol_span=vol_span, num_days=num_days, pt_sl=pt_sl, high=high, low=low,
    )

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
    # ⚠️ 밸류에이션 컬럼(FEATURE_COLS_VALUATION)은 dropna 대상에서 뺌 --
    # NaN을 XGBoost 네이티브 결측치 처리로 넘기기 위함. BASE feature와 라벨만
    # 필수로 유지.
    result = result.dropna(subset=FEATURE_COLS_BASE + ["label_fixed", "label_tb"])
    return result


if __name__ == "__main__":
    # [재생성] 체결 지연 버그 수정 반영 -- 이미 검증된 3종목 전부 다시 생성
    VALIDATED_TICKERS = {
        "064350": "064350.KS",  # 현대로템
        "052690": "052690.KS",  # 한전기술
        "118990": "118990.KQ",  # 모트렉스 (코스닥, .KQ 접미사)
    }
    PT_SL = (2, 1)
    NUM_DAYS = 20
    CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"  # D+1 종가체결(확정) + High/Low 장중터치 반영

    for TICKER_KRX, TICKER in VALIDATED_TICKERS.items():
        print(f"\n{'#' * 60}\n# {TICKER_KRX} ({TICKER})\n{'#' * 60}")

        print(f"=== BASE ({TICKER_KRX}, {CONFIG_LABEL}) ===")
        dataset = build_triple_barrier_dataset(ticker=TICKER, pt_sl=PT_SL, num_days=NUM_DAYS)
        print(f"생성된 데이터셋 shape: {dataset.shape}")

        tb_dist = dataset['label_tb'].value_counts(normalize=True)
        print(f"triple-barrier 라벨(label_tb) 분포:\n{tb_dist}")
        if tb_dist.get(0.0, 0) < 0.02:
            print("경고: label_tb=0 비율이 여전히 2% 미만 -- 배리어가 너무 타이트할 수 있음.")

        out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}.csv"
        dataset.to_csv(out_path)
        print(f"저장 완료: {out_path}")

        print(f"\n=== COMBINED (BASE + VALUATION), {CONFIG_LABEL} ===")
        dataset_combined = build_triple_barrier_dataset_combined(
            ticker=TICKER, ticker_krx=TICKER_KRX, pt_sl=PT_SL, num_days=NUM_DAYS,
        )
        print(f"생성된 데이터셋 shape: {dataset_combined.shape} "
              f"(BASE 대비 {dataset.shape[0] - dataset_combined.shape[0]}행 감소 -- "
              f"밸류에이션 z-score 워밍업 구간 때문, 정상)")

        out_path_combined = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
        dataset_combined.to_csv(out_path_combined)
        print(f"저장 완료: {out_path_combined}")

    print("\n\n3종목 전부 재생성 완료 -- backtest_triple_barrier.py로 종목별 재검증,")
    print("이어서 backtest_triple_barrier_pooled.py로 풀링 재검증할 것.")