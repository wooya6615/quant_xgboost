"""
Meta-labeling 데이터셋 생성 (현대로템 064350, BASE+VALUATION, 1:2 손익비).

1차 신호(primary_side): return_20d의 부호 (단순 모멘텀 룰, 항상 매수/매도 둘 중 하나 --
    "방향을 맞히는 것"은 이 규칙에 맡기고, 얼마나 잘 맞을지 확신도는 2차 모델이 판단)
2차 라벨(meta_label): labeling_triple_barrier.get_meta_labels()로 "그 방향 베팅이
    익절을 손절보다 먼저 쳤는가"를 0/1로 라벨링

이전 실험(backtest_triple_barrier.py)에서 확인된 핵심 문제 -- "gross는 플러스인데
거래비용이 다 갉아먹는다" -- 를 메타 모델의 확신도 threshold로 거래 빈도를 줄여서
해결할 수 있는지가 이 실험의 핵심 질문.

feature: BASE 13개 + VALUATION 5개 (직전 실험에서 밸류에이션이 5/5 시드로 AUC를
개선시켰으므로 처음부터 COMBINED로 시작).

사용법 (레포 루트에서):
    python src/feature_engineering_meta_labeling.py

전제:
    feature_engineering.py, feature_engineering_valuation.py, labeling_triple_barrier.py와
    같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features,
)
from feature_engineering_valuation import load_valuation, add_valuation_features
from labeling_triple_barrier import get_daily_volatility, get_meta_labels

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION


def build_meta_labeling_dataset(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    pt_sl: tuple = (2, 1),          # backtest_triple_barrier.py와 동일 (1:2 손익비)
    vol_span: int = 20,
    num_days: int = 20,
) -> pd.DataFrame:
    df, bench = load_data(ticker, benchmark, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    valuation_df = load_valuation(ticker_krx, start, end)
    df = add_valuation_features(df, valuation_df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED)

    close = df["Close"]

    # 1차 신호: 단순 모멘텀 부호 -- "방향"은 이 규칙이 정하고, "확신도"만 2차 모델이 판단
    primary_side = np.sign(df["return_20d"]).reindex(close.index)
    primary_side = primary_side[primary_side != 0]  # return_20d==0인 극히 드문 날은 베팅 자체가 없음

    daily_vol = get_daily_volatility(close, span=vol_span)
    vol = daily_vol * np.sqrt(num_days)  # triple-barrier 실험과 동일한 스케일링

    meta_bins = get_meta_labels(
        primary_side, close, pt_sl=pt_sl, target=vol, num_days=num_days,
    )

    df["primary_side"] = meta_bins["primary_side"]
    df["meta_label"] = meta_bins["meta_label"]
    df["ret_meta"] = meta_bins["ret"]  # side 보정된 실현수익률 (이미 방향 반영됨)

    touch_pos = close.index.get_indexer(meta_bins["touch_time"])
    entry_pos = close.index.get_indexer(meta_bins.index)
    df["holding_rows_meta"] = pd.Series(touch_pos - entry_pos, index=meta_bins.index)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_COMBINED,
        "primary_side", "meta_label", "ret_meta", "holding_rows_meta",
    ]
    result = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=FEATURE_COLS_COMBINED + ["primary_side", "meta_label"])
    return result


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"
    PT_SL = (2, 1)
    NUM_DAYS = 10  # [조정] 20 -> 10 -- 보유기간을 줄여서 거래 회전율을 높임
                   # (기존 20일 기준으로는 전체 2400행/평균 13일 보유 ~= 180건이 사실상 상한이라
                   #  threshold를 낮춰도 거래 수가 잘 안 늘었음 -- 5-seed 안정성 확보가 우선 목표)

    dataset = build_meta_labeling_dataset(
        ticker=TICKER, ticker_krx=TICKER_KRX, pt_sl=PT_SL, num_days=NUM_DAYS,
    )
    print(f"num_days={NUM_DAYS} 설정 (기존 20일 대비 회전율 상승 목적)")
    print(f"생성된 데이터셋 shape: {dataset.shape}")

    print(f"\n1차 신호(primary_side) 분포:\n{dataset['primary_side'].value_counts(normalize=True)}")
    print(f"\nmeta_label(베팅 성공 여부) 분포:\n{dataset['meta_label'].value_counts(normalize=True)}")

    long_win_rate = dataset.loc[dataset["primary_side"] == 1, "meta_label"].mean()
    short_win_rate = dataset.loc[dataset["primary_side"] == -1, "meta_label"].mean()
    print(f"\n1차 신호=매수(1)일 때 성공률: {long_win_rate:.1%}")
    print(f"1차 신호=매도(-1)일 때 성공률: {short_win_rate:.1%}")
    print("(둘 다 50%대에서 크게 안 벗어나면 1차 룰 자체가 원래 별 정보가 없다는 뜻 --")
    print(" 그래도 괜찮음, 애초에 '방향은 약해도 2차 모델이 골라내는지'가 이 실험의 핵심.)")

    out_path = DATA_DIR / f"{TICKER_KRX}_features_meta_labeling_nd{NUM_DAYS}.csv"
    dataset.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")