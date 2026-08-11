"""
저장된 production 모델(models/*.joblib)로 오늘자 매매 신호 생성.

[중요] docs/candidate_model_triple_barrier_pooled.md의 최종 판정을 반드시 먼저 읽을 것 --
pt_sl 축 PBO 95.6%로 배포 부적합 판정 상태. 이 스크립트가 뽑아주는 신호는 실전
매매용이 아니라 메커니즘 이해/paper trading용으로만 쓸 것.

라벨(익절/손절/수직배리어 도달 여부)은 미래 가격이 있어야 계산 가능하므로, "오늘"
시점에는 만들 수 없음 -- 그래서 feature_engineering_triple_barrier.py 전체를 다시
쓰지 않고, feature 계산 부분만 재사용해서 오늘자 X만 뽑음.

사용법 (레포 루트에서):
    python src/generate_daily_signal.py

전제:
    build_production_models.py로 models/*.joblib, models/production_metadata.json이
    이미 생성돼 있어야 함.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features,
)
from feature_engineering_valuation import load_valuation, add_valuation_features
from labeling_triple_barrier import get_daily_volatility

PT_SL = (2, 1)   # build_production_models.py의 설정과 동일하게 맞출 것
NUM_DAYS = 20
VOL_SPAN = 20

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

TICKER_SUFFIX = {
    "064350": "064350.KS",
    "052690": "052690.KS",
    "118990": "118990.KQ",
}
BENCHMARK = "^KS11"


def compute_latest_features(ticker_krx: str, lookback_days: int = 400) -> tuple:
    """
    오늘(가장 최근 거래일) 시점까지의 feature와 배리어 % 폭(target)을 계산.
    라벨은 계산하지 않음 (미래 데이터가 없어서 애초에 불가능).
    lookback_days: rolling feature(예: 252일 z-score) 계산에 필요한 만큼 과거로 넉넉히 잡음.

    반환: (feature Series, pt_pct, sl_pct)
        pt_pct: 익절 목표 (진입가 대비 +X%, 예: 0.12 = +12%)
        sl_pct: 손절 목표 (진입가 대비 -X%, 예: 0.06 = -6%)
    """
    ticker = TICKER_SUFFIX[ticker_krx]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    df, bench = load_data(ticker, BENCHMARK, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    valuation_df = load_valuation(ticker_krx, start, end)
    df = add_valuation_features(df, valuation_df)

    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=FEATURE_COLS_COMBINED)
    if df.empty:
        raise ValueError(f"{ticker_krx}: feature 계산 후 남은 행이 없음 -- lookback_days를 늘려볼 것")

    latest_row = df.iloc[-1]
    print(f"  [{ticker_krx}] 최신 데이터 날짜: {df.index[-1].date()}")

    # 오늘까지의 정보로 배리어 % 폭 계산 (labeling_triple_barrier.py와 동일한 공식)
    daily_vol = get_daily_volatility(df["Close"], span=VOL_SPAN)
    target = daily_vol.iloc[-1] * (NUM_DAYS ** 0.5)  # daily_vol * sqrt(num_days)
    pt_pct = PT_SL[0] * target
    sl_pct = PT_SL[1] * target

    return latest_row[FEATURE_COLS_COMBINED], pt_pct, sl_pct


if __name__ == "__main__":
    with open(MODELS_DIR / "production_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    if "FINAL_VERDICT" in metadata:
        print("=" * 70)
        print("경고:", metadata["FINAL_VERDICT"])
        print("=" * 70)
        print()

    threshold = metadata["strategy"]["entry_threshold"]
    print(f"threshold={threshold} 기준으로 오늘자 신호 확인\n")

    results = []
    for ticker_krx in TICKER_SUFFIX:
        print(f"[{ticker_krx}] 처리 중...")
        model_path = MODELS_DIR / f"{ticker_krx}_triple_barrier_model.joblib"
        model = joblib.load(model_path)

        X_latest, pt_pct, sl_pct = compute_latest_features(ticker_krx)
        proba = model.predict_proba(X_latest.values.reshape(1, -1))[0, 1]
        signal = "매수 신호" if proba >= threshold else "대기"

        results.append({
            "종목": ticker_krx, "확률": round(proba, 4), "판정": signal,
            "익절폭(%)": round(pt_pct * 100, 2), "손절폭(%)": round(-sl_pct * 100, 2),
        })
        print(f"  확률: {proba:.1%} -> {signal} (익절폭 +{pt_pct:.1%} / 손절폭 -{sl_pct:.1%})\n")

    print("=== 오늘자 신호 요약 ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n매수 신호가 뜬 종목은 '다음 거래일(D+1) 종가'에 진입하는 걸 가정한 전략임")
    print("(오늘 종가가 아님 -- labeling_triple_barrier.py의 체결 지연 로직과 일치시킬 것)")
    print("\n익절폭/손절폭은 % 기준이야 -- 내일 실제 체결가가 나오면")
    print("set_barrier_prices.py로 원화 가격으로 확정할 것.")