"""
진입 당일(시가 체결 그날)에 바로 배리어가 터지는 비율 진단.

holding_rows_tb == 0이면 "진입한 그날 종가까지의 움직임만으로 이미 배리어를 쳤다"는
뜻 -- 시가체결 버전에서만 발생 가능한 케이스 (종가체결은 구조상 진입일에 못 침).
이게 손절 쪽으로 쏠려있는지, 아니면 전체 손절/익절 비율이랑 비슷한지 확인.

사용법 (레포 루트에서):
    python src/diagnose_day1_touches.py <ticker_krx>
    예: python src/diagnose_day1_touches.py 064350
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_LABEL = "pt2sl1_nd20_open"


def diagnose(ticker_krx: str, combined: bool = True):
    suffix = "_valuation" if combined else ""
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}{suffix}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    total = len(df)
    overall_dist = df["label_tb"].value_counts(normalize=True)

    print(f"=== {ticker_krx} ({'COMBINED' if combined else 'BASE'}) ===")
    print(f"전체 이벤트: {total}건")
    print(f"\n전체 라벨 분포:\n{overall_dist.round(4)}")

    # holding_rows_tb 전체 분포 -- 짧은 보유기간에 쏠려있는지 감 잡기
    print(f"\nholding_rows_tb 분포 (몇 거래일 만에 배리어를 쳤는지):")
    print(df["holding_rows_tb"].describe().round(2))

    short_holds = df[df["holding_rows_tb"] <= 2]
    print(f"\n보유 2일 이하로 끝난 이벤트: {len(short_holds)}건 ({len(short_holds) / total:.1%})")
    if len(short_holds) > 0:
        print(f"그중 라벨 분포:\n{short_holds['label_tb'].value_counts(normalize=True).round(4)}")

    day1 = df[df["holding_rows_tb"] == 0]
    day1_dist = day1["label_tb"].value_counts(normalize=True)

    print(f"\n진입 당일(holding_rows_tb==0)에 바로 터진 이벤트: {len(day1)}건 "
          f"(전체의 {len(day1) / total:.1%})")
    if len(day1) > 0:
        print(f"그중 라벨 분포:\n{day1_dist.round(4)}")
        day1_sl_share = day1_dist.get(-1.0, 0)
        overall_sl_share = overall_dist.get(-1.0, 0)
        print(f"\n손절(-1) 비율: 전체 {overall_sl_share:.1%} vs 진입당일만 {day1_sl_share:.1%}")
        if day1_sl_share > overall_sl_share + 0.05:
            print("-> 진입 당일 손절 비율이 전체 평균보다 눈에 띄게 높음 -- "
                  "손절선이 진입일 하루의 정상적인 변동폭 대비 타이트할 가능성")
        else:
            print("-> 진입 당일 손절 비율이 전체 평균과 비슷함 -- "
                  "'진입일에 유난히 손절 몰림' 현상은 아닌 것으로 보임 (다른 원인 찾아야 함)")
    else:
        print("진입 당일 터진 이벤트가 없음 (종가체결 버전이거나 데이터 문제일 수 있음)")


if __name__ == "__main__":
    ticker_krx = sys.argv[1] if len(sys.argv) > 1 else "064350"
    diagnose(ticker_krx, combined=True)
    print()
    diagnose(ticker_krx, combined=False)