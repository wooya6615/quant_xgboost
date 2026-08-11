"""
Open/Close 데이터의 액면분할·배당 조정 불일치 여부 진단.

배경: 시가체결로 바꿨더니 종가체결 대비 결과가 극단적으로 나빠졌는데, 합성 데이터로
통제 실험을 해봐도 그 정도 격차가 재현이 안 됨 -- 즉 "정보 우위"나 "노이즈 회피" 같은
경제적 설명으로는 부족하고, 데이터 자체의 문제일 가능성이 큼.

가장 흔한 원인: yfinance에서 Close는 조정(분할/배당 반영)됐는데 Open은 조정이 안 됐거나
그 반대인 경우 -- 실제 분할이 있었던 날짜 근처에서 Open/Close가 인위적으로 크게
어긋나면서 "가짜 갭"이 생김. 예: 2:1 분할이 있었다면 그 날 전후로 가격이 반토막 나는데,
한쪽만 조정되면 갭이 -50%에 가깝게 찍힘.

이 스크립트는 실제 종목의 Open/Close를 다시 받아서 "전일 종가 대비 당일 시가 갭"
분포를 보고, 비정상적으로 큰 값(예: |갭| > 15%)이 있는지, 그리고 그 날짜가 실제
액면분할/무상증자 등과 겹치는지 확인함.

사용법 (레포 루트에서):
    python src/diagnose_open_close_adjustment.py <ticker_krx>
    예: python src/diagnose_open_close_adjustment.py 064350
"""

import sys

import pandas as pd
import yfinance as yf

TICKER_SUFFIX_MAP = {
    "064350": "064350.KS",
    "052690": "052690.KS",
    "118990": "118990.KQ",  # 코스닥
}


def diagnose(ticker_krx: str, start: str = "2015-01-01", end: str = "2026-07-18"):
    ticker = TICKER_SUFFIX_MAP.get(ticker_krx, f"{ticker_krx}.KS")
    print(f"=== {ticker_krx} ({ticker}) ===")

    # auto_adjust=True (기본값, Close/Open 등 전부 조정됨)와
    # auto_adjust=False (Close만 별도 Adj Close 컬럼, 나머지는 미조정) 둘 다 받아서 비교
    df_adj = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df_raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if isinstance(df_adj.columns, pd.MultiIndex):
        df_adj.columns = df_adj.columns.get_level_values(0)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    print(f"\n[auto_adjust=True] 컬럼: {list(df_adj.columns)}")
    print(f"[auto_adjust=False] 컬럼: {list(df_raw.columns)}")

    # 두 버전의 Close가 완전히 같은지 확인 -- 다르면 분할/배당 이벤트가 있었다는 뜻
    close_diff = (df_adj["Close"] - df_raw["Close"]).abs()
    n_diff_days = (close_diff > 0.01).sum()
    print(f"\nauto_adjust=True vs False Close가 다른 날: {n_diff_days}일 "
          f"(0이면 조정 이벤트 없음, 0보다 크면 분할/배당 있었다는 뜻)")

    if n_diff_days > 0:
        first_diff_date = close_diff[close_diff > 0.01].index[0]
        print(f"조정이 시작되는 첫 날짜: {first_diff_date.date()} "
              f"(이 날짜 근처에 분할/배당이 있었을 가능성)")

    # 실제 사용 중인 auto_adjust=True 버전 기준으로, 전일종가->당일시가 갭 분포 확인
    close = df_adj["Close"]
    open_ = df_adj["Open"]
    gap = (open_ / close.shift(1) - 1).dropna()

    print(f"\n[auto_adjust=True 기준] 갭(전일종가->당일시가) 분포:")
    print(gap.describe().round(4))

    extreme = gap[gap.abs() > 0.15]
    print(f"\n|갭| > 15%인 날: {len(extreme)}건")
    if len(extreme) > 0:
        print(extreme.round(4))
        print("\n-> 이 날짜들이 실제 뉴스(상한가/하한가, 급락 등)로 설명 가능한지,")
        print("   아니면 데이터 조정 문제로 보이는지 직접 확인해볼 것")


if __name__ == "__main__":
    ticker_krx = sys.argv[1] if len(sys.argv) > 1 else "064350"
    diagnose(ticker_krx)