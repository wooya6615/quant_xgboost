"""
새 종목 스크리닝 -- 052690 조사에서 나온 "이벤트 집중도" 지표를 사전등록 기준으로
적용해서, 소수 이벤트일에 의존하지 않는 종목 풀을 새로 구성.

⚠️ 사전등록 원칙: 아래 기준은 전부 가격/거래량 데이터만으로 계산되고, 어떤 백테스트
성과도 참조하지 않음. 결과를 보고 기준을 바꾸지 않을 것.

사용법 (레포 루트에서):
    python src/screen_universe.py

⚠️ 코스피+코스닥 1차 필터 통과 종목 순회라 시간이 좀 걸릴 수 있음.
⚠️ get_market_cap_by_ticker()의 정확한 컬럼명(시가총액/거래대금/종목명)은
   pykrx 버전에 따라 다를 수 있음 -- 에러 나면 print(kospi.columns) 찍어서
   실제 컬럼명 알려줘.
"""

from dotenv import load_dotenv

load_dotenv()

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pykrx import stock

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RECENT_DATE = "20260718"
HIST_START = "20150101"
HIST_END = "20260718"
MIN_CAP = 100_000_000_000        # 1,000억원
MAX_CAP = 5_000_000_000_000      # 5조원
MIN_TRADING_VALUE_SNAPSHOT = 5_000_000_000  # 50억원 (스냅샷 기준 1차 필터)

RECENT_DATE = stock.get_nearest_business_day_in_a_week()
print(f"기준일(최근 영업일): {RECENT_DATE}")

HIST_START = "20150101"
HIST_END = "20260718"

def stage1_cross_sectional_filter() -> pd.DataFrame:
    kospi = stock.get_market_cap_by_ticker(RECENT_DATE, market="KOSPI")
    kosdaq = stock.get_market_cap_by_ticker(RECENT_DATE, market="KOSDAQ")

    # ⚠️ 디버그: 실제 컬럼명/타입/샘플값 확인
    print("컬럼:", kospi.columns.tolist())
    print("타입:\n", kospi.dtypes)
    print("샘플:\n", kospi.head(3))
    print(f"\n시가총액 분포: min={kospi['시가총액'].min():,.0f}, "
          f"median={kospi['시가총액'].median():,.0f}, max={kospi['시가총액'].max():,.0f}")
    print(f"거래대금 분포: min={kospi['거래대금'].min():,.0f}, "
          f"median={kospi['거래대금'].median():,.0f}, max={kospi['거래대금'].max():,.0f}")

    kospi["시장"] = "KOSPI"
    kosdaq["시장"] = "KOSDAQ"
    combined = pd.concat([kospi, kosdaq])

    filtered = combined[
        (combined["시가총액"] >= MIN_CAP) &
        (combined["시가총액"] <= MAX_CAP) &
        (combined["거래대금"] >= MIN_TRADING_VALUE_SNAPSHOT)
    ]
    print(f"1차 필터(시가총액+최근거래대금) 통과: {len(filtered)}종목 (전체 {len(combined)}종목 중)")
    return filtered


def stage2_history_and_concentration(candidates: pd.DataFrame, sleep_sec: float = 0.2) -> pd.DataFrame:
    rows = []
    for i, (ticker, row) in enumerate(candidates.iterrows()):
        if i % 20 == 0:
            print(f"  진행: {i}/{len(candidates)}")
        try:
            ohlcv = stock.get_market_ohlcv_by_date(HIST_START, HIST_END, ticker)
        except Exception:
            continue
        if ohlcv.empty:
            continue

        first_date = ohlcv.index.min()
        if first_date > pd.Timestamp("2015-06-01"):
            continue  # 2015년 상반기 이후 상장이면 히스토리 부족으로 제외

        daily_ret = ohlcv["종가"].pct_change().dropna()
        if len(daily_ret) < 1000:
            continue

        total_log_return = np.log1p(daily_ret).sum()
        if total_log_return <= 0:
            continue  # 전체 기간 하락 종목은 집중도(분모) 계산이 무의미 -- 제외

        top20_share = np.log1p(daily_ret).nlargest(20).sum() / total_log_return
        avg_trading_value = ohlcv["거래대금"].mean() if "거래대금" in ohlcv.columns else np.nan

        name = stock.get_market_ticker_name(ticker)
        
        rows.append({
            "ticker": ticker,
            "name": name,
            "market": row["시장"],
            "market_cap": row["시가총액"],
            "avg_trading_value_full_hist": avg_trading_value,
            "top20_concentration": top20_share,
            "first_date": first_date,
        })
        time.sleep(sleep_sec)

    result = pd.DataFrame(rows)
    print(f"\n2차 필터(2015년 이전 상장 + 데이터 충분) 통과: {len(result)}종목")
    return result


if __name__ == "__main__":
    stage1 = stage1_cross_sectional_filter()
    stage2 = stage2_history_and_concentration(stage1)

    median_concentration = stage2["top20_concentration"].median()
    print(f"\n상위20일 집중도 중앙값: {median_concentration:.1%}")

    final = stage2[stage2["top20_concentration"] <= median_concentration].copy()
    final = final.sort_values("top20_concentration")

    print(f"\n최종 통과: {len(final)}종목 (집중도 중앙값 이하)")
    print(final[["ticker", "name", "market", "market_cap", "top20_concentration"]].to_string(index=False))

    out_path = DATA_DIR / "screened_universe.csv"
    final.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")