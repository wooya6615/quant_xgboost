"""
3종목 기초 프로필 비교 -- 052690(한전기술)이 왜 유독 약한지 가설 세우기 위한
탐색적 조사. 이번 실험 판정에는 영향 없음 (참고용).

사용법:
    python src/analyze_ticker_profile.py
"""

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
import pandas as pd
import numpy as np
from pykrx import stock

TICKERS = {
    "064350": "현대로템",
    "052690": "한전기술",
    "118990": "모트렉스",
    "010170": "대한광통신",
}
START, END = "20150101", "20260718"


def profile_ticker(ticker_krx: str, name: str):
    print(f"\n{'=' * 50}\n{name} ({ticker_krx})\n{'=' * 50}")

    # 시가총액/거래대금 (유동성)
    cap_df = stock.get_market_cap_by_date(START, END, ticker_krx)
    avg_trading_value = cap_df["거래대금"].mean()
    avg_market_cap = cap_df["시가총액"].mean()
    print(f"평균 시가총액: {avg_market_cap/1e8:,.0f}억원")
    print(f"평균 일일 거래대금: {avg_trading_value/1e8:,.0f}억원")

    # OHLCV 기반 변동성/급등락 빈도
    ohlcv = stock.get_market_ohlcv_by_date(START, END, ticker_krx)
    daily_ret = ohlcv["종가"].pct_change()
    print(f"일간 수익률 변동성(연환산): {daily_ret.std() * np.sqrt(252):.1%}")
    print(f"±10% 이상 급등락일 비율: {(daily_ret.abs() > 0.10).mean():.2%}")
    print(f"±20% 이상 급등락일 비율: {(daily_ret.abs() > 0.20).mean():.2%}")

    # 최대 낙폭
    cum = (1 + daily_ret.fillna(0)).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    print(f"최대낙폭(MDD): {mdd:.1%}")

    return {
        "종목": name, "코드": ticker_krx,
        "평균거래대금(억)": avg_trading_value / 1e8,
        "연환산변동성": daily_ret.std() * np.sqrt(252),
        "급등락(10%+)비율": (daily_ret.abs() > 0.10).mean(),
        "MDD": mdd,
    }


if __name__ == "__main__":
    results = [profile_ticker(t, n) for t, n in TICKERS.items()]
    print("\n\n" + "=" * 50)
    print("=== 3종목 비교 요약 ===")
    print("=" * 50)
    print(pd.DataFrame(results).round(4).to_string(index=False))