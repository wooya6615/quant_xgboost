"""
저장된 064350 BASE 전용 production 모델로 특정 날짜 기준 매매 신호 생성.
+ 포지션 상태 추적 (이미 보유중이면 새 신호 생성 안 함 -- 백테스트의
  "보유기간만큼 건너뛰기" 로직을 실전 신호 생성에도 동일하게 반영).

generate_daily_signal.py(3종목 COMBINED, [배포 부적합])의 KST 미확정봉 처리
로직은 그대로 가져오고, 밸류에이션 관련 부분(load_valuation/add_valuation_
features)은 전부 뺌 -- 이 모델은 BASE 13개 feature만 씀.

[중요] build_production_model_064350_base.py의 FINAL_VERDICT를 반드시 먼저
읽을 것 -- "잠정 통과 -- 완전 배포 적합 아님" 상태. paper trading(모의투자)
용도로만 쓸 것.

[포지션 추적 관련 중요 사항]
    - 상태 파일(models/064350_position_state.json)이 없으면 "포지션 없음"으로
      취급함 -- 과거 이력을 소급해서 채울 필요 없음. 실제로 모의투자를 시작하는
      날부터 자연스럽게 이 파일이 생성되기 시작함.
    - 포지션 보유중에는 새 매수 신호를 계산하지 않음 (백테스트 generate_trades()의
      "i += holding"과 동일한 원칙 -- 한 번에 포지션 하나만).
    - 강제청산 예정일(수직 배리어)이 지나면 자동으로 포지션을 정리하고 다시 신호를 봄.
    - 익절/손절이 그 전에 실제로 체결되면 스크립트가 자동으로 알 방법이 없음
      (실시간 시세 연동 안 함) -- 반드시 --close 옵션으로 직접 알려줄 것.
    - 특정 과거 날짜(AS_OF_DATE)를 인자로 주는 "재현" 모드에서는 실제 포지션
      상태 파일을 건드리지 않음 (오늘 날짜로 실행할 때만 상태 추적이 동작함).

사용법 (레포 루트에서):
    python src/generate_daily_signal_064350_base.py                # 오늘 날짜 기준, 포지션 추적됨
    python src/generate_daily_signal_064350_base.py 2026-08-10      # 특정 날짜 재현, 포지션 추적 안 함
    python src/generate_daily_signal_064350_base.py --close                    # 포지션 청산 처리만
    python src/generate_daily_signal_064350_base.py --close --close-price 52000 # 청산가 기록까지

전제:
    build_production_model_064350_base.py로 models/064350_triple_barrier_base_
    model.joblib, models/064350_triple_barrier_base_metadata.json이 이미
    생성돼 있어야 함.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features,
)
from labeling_triple_barrier import get_daily_volatility

PT_SL = (2, 1)   # build_production_model_064350_base.py와 동일하게 맞출 것
NUM_DAYS = 30    # [교체됨] 기존 20 -> num_days 축 검증으로 30 확정 (docs/PROJECT_SUMMARY.md 참고)
VOL_SPAN = 20

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "064350_triple_barrier_base_model.joblib"
METADATA_PATH = MODELS_DIR / "064350_triple_barrier_base_metadata.json"
STATE_PATH = MODELS_DIR / "064350_position_state.json"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

TICKER = "064350.KS"
TICKER_KRX = "064350"
BENCHMARK = "^KS11"

KRX_TZ = ZoneInfo("Asia/Seoul")
KRX_CLOSE_HOUR, KRX_CLOSE_MINUTE = 15, 30  # 코스피/코스닥 정규장 마감 15:30 KST


def _market_closed_today() -> bool:
    now_kst = datetime.now(KRX_TZ)
    return (now_kst.hour, now_kst.minute) >= (KRX_CLOSE_HOUR, KRX_CLOSE_MINUTE)


# ------------------------------------------------------------------
# 포지션 상태 관리 -- 상태 파일 없으면 "포지션 없음"이 기본값
# ------------------------------------------------------------------
def load_position_state() -> dict:
    if not STATE_PATH.exists():
        return {"position_open": False}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_position_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def close_position(close_price: float = None):
    state = load_position_state()
    if not state.get("position_open"):
        print("현재 열려있는 포지션이 없음 -- 청산 처리할 게 없습니다.")
        return
    print(f"포지션 청산 처리: 진입일={state.get('entry_date')}, "
          f"청산가={close_price if close_price is not None else '(미기록)'}")
    save_position_state({
        "position_open": False,
        "last_closed_entry_date": state.get("entry_date"),
        "last_closed_price": close_price,
        "last_closed_at": datetime.now(KRX_TZ).isoformat(),
    })


# ------------------------------------------------------------------
# feature 계산
# ------------------------------------------------------------------
def compute_latest_features(as_of_date: str = None, lookback_days: int = 400) -> tuple:
    as_of_ts = pd.Timestamp.today() if as_of_date is None else pd.Timestamp(as_of_date)

    end = (as_of_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (as_of_ts - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    df, bench = load_data(TICKER, BENCHMARK, start, end)
    df = df[df.index <= as_of_ts]

    today_ts = pd.Timestamp.today().normalize()
    if as_of_ts.normalize() == today_ts and not _market_closed_today():
        before_drop = len(df)
        df = df[df.index.normalize() < today_ts]
        if len(df) < before_drop:
            print("  오늘 정규장 미마감 -- 미확정 오늘자 행 제외, 직전 거래일 사용")

    if df.empty:
        raise ValueError(f"as_of_date={as_of_ts.date()} 이전 데이터가 없음")

    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=FEATURE_COLS_BASE)
    if df.empty:
        raise ValueError("feature 계산 후 남은 행이 없음 -- lookback_days를 늘려볼 것")

    latest_row = df.iloc[-1]
    print(f"기준일: {as_of_ts.date()} / 실제 사용된 최신 거래일: {df.index[-1].date()}")

    daily_vol = get_daily_volatility(df["Close"], span=VOL_SPAN)
    target = daily_vol.iloc[-1] * (NUM_DAYS ** 0.5)
    pt_pct = PT_SL[0] * target
    sl_pct = PT_SL[1] * target

    return latest_row[FEATURE_COLS_BASE], pt_pct, sl_pct, df.index[-1]


def estimate_entry_and_vertical_barrier_dates(last_confirmed_date: pd.Timestamp) -> tuple:
    """
    [주의] 미래 날짜라 실제 KRX 개장일(공휴일 등)을 정확히 반영 못 함 --
    주말만 제외한 영업일 근사치. 실제 강제청산일은 이보다 며칠 뒤로 밀릴 수 있음.
    """
    entry_date = last_confirmed_date + pd.tseries.offsets.BDay(1)
    vertical_barrier_date = entry_date + pd.tseries.offsets.BDay(NUM_DAYS)
    return entry_date, vertical_barrier_date


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("as_of_date", nargs="?", default=None,
                         help="YYYY-MM-DD. 생략하면 오늘 날짜 (포지션 추적 활성화됨)")
    parser.add_argument("--close", action="store_true",
                         help="현재 보유 포지션을 청산 처리 (익절/손절이 실제로 체결된 후 수동 기록)")
    parser.add_argument("--close-price", type=float, default=None,
                         help="청산 체결가 (기록용, 선택)")
    args = parser.parse_args()

    if args.close:
        close_position(args.close_price)
        raise SystemExit(0)

    AS_OF_DATE = args.as_of_date
    is_live_run = AS_OF_DATE is None  # 오늘 날짜로 돌릴 때만 포지션 상태를 실제로 추적/갱신함

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    if "FINAL_VERDICT" in metadata:
        print("=" * 70)
        print("경고:", metadata["FINAL_VERDICT"])
        print("=" * 70)
        print()

    # --- 포지션 보유중이면 새 신호 생성 자체를 건너뜀 (오늘 날짜 실행일 때만) ---
    if is_live_run:
        state = load_position_state()
        if state.get("position_open"):
            vb_date = pd.Timestamp(state["vertical_barrier_date"])
            today_ts = pd.Timestamp.today().normalize()

            if today_ts < vb_date:
                print(f"=== 포지션 보유중 ({TICKER_KRX}) -- 새 신호 생성 안 함 ===")
                print(f"진입일: {state.get('entry_date')}")
                print(f"강제청산 예정일: {state.get('vertical_barrier_date')}")
                print(f"익절폭: +{state.get('pt_pct', 0) * 100:.2f}%, "
                      f"손절폭: -{state.get('sl_pct', 0) * 100:.2f}%")
                print("\n실제로 익절/손절이 먼저 체결됐다면, 아래 명령으로 포지션을")
                print("정리한 뒤 다시 실행할 것:")
                print(f"  python src/generate_daily_signal_064350_base.py --close --close-price <체결가>")
                raise SystemExit(0)
            else:
                print(f"강제청산 예정일({state.get('vertical_barrier_date')}) 도달 -- "
                      f"포지션 자동 정리 후 새 신호 확인 진행\n")
                save_position_state({"position_open": False})

    threshold = metadata["threshold"]
    date_label = AS_OF_DATE if AS_OF_DATE else "오늘"
    print(f"threshold={threshold} 기준으로 {date_label} 날짜 신호 확인 (064350, BASE 전용)")
    if not is_live_run:
        print("(과거 날짜 재현 모드 -- 포지션 상태 파일은 갱신하지 않음)")
    print()

    model = joblib.load(MODEL_PATH)
    X_latest, pt_pct, sl_pct, last_confirmed_date = compute_latest_features(as_of_date=AS_OF_DATE)
    proba = model.predict_proba(X_latest.values.reshape(1, -1))[0, 1]
    signal = "매수 신호" if proba >= threshold else "대기"

    print(f"\n=== {date_label} 기준 신호 ({TICKER_KRX}) ===")
    print(f"확률: {proba:.1%} -> {signal}")
    print(f"익절폭: +{pt_pct:.1%}, 손절폭: -{sl_pct:.1%}")

    entry_date, vertical_barrier_date = estimate_entry_and_vertical_barrier_dates(last_confirmed_date)

    if signal == "매수 신호":
        print(f"\n예상 진입일(D+1): {entry_date.date()}")
        print(f"예상 강제청산일(수직 배리어, {NUM_DAYS}거래일 후): {vertical_barrier_date.date()}")
        print("  -> 이날까지 익절가/손절가 둘 다 안 건드리면 이날 종가로 강제 청산")
        print("  -> 공휴일 미반영 근사치라 실제로는 며칠 뒤로 밀릴 수 있음 -- "
              "체결 확정 후 실제 개장일 기준으로 다시 확인할 것")

        if is_live_run:
            save_position_state({
                "position_open": True,
                "signal_date": str(pd.Timestamp.today().date()),
                "entry_date": str(entry_date.date()),
                "vertical_barrier_date": str(vertical_barrier_date.date()),
                "pt_pct": pt_pct,
                "sl_pct": sl_pct,
                "proba_at_signal": float(proba),
            })
            print(f"\n포지션 상태 저장 완료: {STATE_PATH}")
            print("(다음 실행부터 청산 전까지 새 신호를 생성하지 않음)")

    print("\n매수 신호가 뜨면 '다음 거래일(D+1) 종가'에 진입하는 걸 가정한 전략임")
    print("(기준일 종가가 아님 -- labeling_triple_barrier.py 체결 지연 로직과 일치)")
    print("\n체결가가 나오면 다음 명령으로 실제 원화 익절가/손절가를 확정할 것:")
    print(f"  python src/set_barrier_prices.py {TICKER_KRX} <체결가> {pt_pct * 100:.2f} {sl_pct * 100:.2f}")