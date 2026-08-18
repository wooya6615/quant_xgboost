"""
저장된 production 모델(models/*.joblib)로 특정 날짜 기준 매매 신호 생성.

[중요] docs/candidate_model_triple_barrier_pooled.md의 최종 판정을 반드시 먼저 읽을 것 --
pt_sl 축 PBO 95.6%로 배포 부적합 판정 상태. 이 스크립트가 뽑아주는 신호는 실전
매매용이 아니라 메커니즘 이해/paper trading용으로만 쓸 것.

라벨(익절/손절/수직배리어 도달 여부)은 미래 가격이 있어야 계산 가능하므로, 기준일
시점에는 만들 수 없음 -- 그래서 feature_engineering_triple_barrier.py 전체를 다시
쓰지 않고, feature 계산 부분만 재사용해서 기준일 시점의 X만 뽑음.

사용법 (레포 루트에서):
    python src/generate_daily_signal.py                # 오늘 날짜 기준
    python src/generate_daily_signal.py 2026-08-10      # 특정 날짜 기준 (YYYY-MM-DD)

전제:
    build_production_models.py로 models/*.joblib, models/production_metadata.json이
    이미 생성돼 있어야 함.

주의:
    as_of_date를 과거로 줄 경우, 그 시점 이후에 실제로 벌어진 일(익절/손절 도달 여부 등)은
    당연히 반영되지 않음. 이 스크립트는 어디까지나 "그 날짜에 이 모델을 돌렸다면 어떤
    신호가 나왔을지"를 재현하는 용도임 -- 실제 백테스트(run_universe_triple_barrier.py 등)와
    혼동하지 말 것.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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

KRX_TZ = ZoneInfo("Asia/Seoul")
KRX_CLOSE_HOUR, KRX_CLOSE_MINUTE = 15, 30  # 코스피/코스닥 정규장 마감 15:30 KST


def _market_closed_today() -> bool:
    """오늘 KRX 정규장이 이미 마감했는지 여부 (단순 시각 기준, 휴장일은 고려 안 함)."""
    now_kst = datetime.now(KRX_TZ)
    return (now_kst.hour, now_kst.minute) >= (KRX_CLOSE_HOUR, KRX_CLOSE_MINUTE)


def compute_latest_features(ticker_krx: str, as_of_date: str = None, lookback_days: int = 400) -> tuple:
    """
    as_of_date(가장 최근 거래일 <= as_of_date) 시점까지의 feature와 배리어 % 폭(target)을 계산.
    라벨은 계산하지 않음 (미래 데이터가 없어서 애초에 불가능).

    as_of_date: "YYYY-MM-DD" 문자열. None이면 오늘 날짜.
    lookback_days: rolling feature(예: 252일 z-score) 계산에 필요한 만큼 과거로 넉넉히 잡음.

    반환: (feature Series, pt_pct, sl_pct)
        pt_pct: 익절 목표 (진입가 대비 +X%, 예: 0.12 = +12%)
        sl_pct: 손절 목표 (진입가 대비 -X%, 예: 0.06 = -6%)
    """
    ticker = TICKER_SUFFIX[ticker_krx]
    as_of_ts = pd.Timestamp.today() if as_of_date is None else pd.Timestamp(as_of_date)

    # yfinance의 end는 exclusive라서 as_of_ts 당일 데이터가 빠질 수 있음 --
    # 하루 여유를 두고 받은 다음, 아래에서 as_of_ts 이후 데이터를 다시 한번 잘라냄.
    end = (as_of_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (as_of_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    df, bench = load_data(ticker, BENCHMARK, start, end)

    # as_of_ts 이후 데이터가 섞여 들어오지 않도록 명시적으로 한 번 더 제한
    # (미래 데이터를 참조하면 실제로는 그 시점에 몰랐을 정보를 쓰게 되는 룩어헤드 버그가 됨)
    df = df[df.index <= as_of_ts]

    # [중요] as_of_date가 "오늘"이고 아직 정규장(15:30 KST)이 안 끝났다면,
    # yfinance가 오늘 날짜로 채워주는 행은 확정 종가가 아니라 그 시점까지의 미확정 체결가임.
    # 이걸 그대로 쓰면 "아직 안 나온 오늘 종가"를 이미 안 것처럼 feature를 계산하는 셈이라
    # 반드시 제외하고, 마지막으로 확정된 거래일(전 거래일)까지만 사용해야 함.
    today_ts = pd.Timestamp.today().normalize()
    if as_of_ts.normalize() == today_ts and not _market_closed_today():
        before_drop = len(df)
        df = df[df.index.normalize() < today_ts]
        if len(df) < before_drop:
            print(f"  [{ticker_krx}] 오늘 정규장 미마감 -- 미확정 오늘자 행 제외, 직전 거래일 사용")

    if df.empty:
        raise ValueError(f"{ticker_krx}: as_of_date={as_of_ts.date()} 이전 데이터가 없음")

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
    print(f"  [{ticker_krx}] 기준일: {as_of_ts.date()} / 실제 사용된 최신 거래일: {df.index[-1].date()}")

    # 기준일까지의 정보로 배리어 % 폭 계산 (labeling_triple_barrier.py와 동일한 공식)
    daily_vol = get_daily_volatility(df["Close"], span=VOL_SPAN)
    target = daily_vol.iloc[-1] * (NUM_DAYS ** 0.5)  # daily_vol * sqrt(num_days)
    pt_pct = PT_SL[0] * target
    sl_pct = PT_SL[1] * target

    return latest_row[FEATURE_COLS_COMBINED], pt_pct, sl_pct


if __name__ == "__main__":
    # 커맨드라인 인자로 날짜를 받음. 안 주면 오늘 날짜로 동작 (기존과 동일).
    # 예: python src/generate_daily_signal.py 2026-08-10
    AS_OF_DATE = sys.argv[1] if len(sys.argv) > 1 else None

    with open(MODELS_DIR / "production_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    if "FINAL_VERDICT" in metadata:
        print("=" * 70)
        print("경고:", metadata["FINAL_VERDICT"])
        print("=" * 70)
        print()

    threshold = metadata["strategy"]["entry_threshold"]
    date_label = AS_OF_DATE if AS_OF_DATE else "오늘"
    print(f"threshold={threshold} 기준으로 {date_label} 날짜 신호 확인\n")

    results = []
    for ticker_krx in TICKER_SUFFIX:
        print(f"[{ticker_krx}] 처리 중...")
        model_path = MODELS_DIR / f"{ticker_krx}_triple_barrier_model.joblib"
        model = joblib.load(model_path)

        X_latest, pt_pct, sl_pct = compute_latest_features(ticker_krx, as_of_date=AS_OF_DATE)
        proba = model.predict_proba(X_latest.values.reshape(1, -1))[0, 1]
        signal = "매수 신호" if proba >= threshold else "대기"

        results.append({
            "종목": ticker_krx, "확률": round(proba, 4), "판정": signal,
            "익절폭(%)": round(pt_pct * 100, 2), "손절폭(%)": round(-sl_pct * 100, 2),
        })
        print(f"  확률: {proba:.1%} -> {signal} (익절폭 +{pt_pct:.1%} / 손절폭 -{sl_pct:.1%})\n")

    print(f"=== {date_label} 기준 신호 요약 ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n매수 신호가 뜬 종목은 '다음 거래일(D+1) 종가'에 진입하는 걸 가정한 전략임")
    print("(기준일 종가가 아님 -- labeling_triple_barrier.py의 체결 지연 로직과 일치시킬 것)")
    print("\n익절폭/손절폭은 % 기준이야 -- 실제 체결가가 나오면")
    print("set_barrier_prices.py로 원화 가격으로 확정할 것.")