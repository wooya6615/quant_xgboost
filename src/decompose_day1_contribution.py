"""
진입 당일(D+1) 하루의 시가->종가 움직임이 전체 거래 수익에서 차지하는 비중 확인.

원리: 청산일이 동일한 거래라면
    (1+ret_close) = exit_price / close_{D+1}
    (1+ret_open)  = exit_price / open_{D+1}
두 식을 나누면 exit_price가 소거됨:
    close_{D+1} / open_{D+1} = (1+ret_open) / (1+ret_close)
    => day1_return = close_{D+1}/open_{D+1} - 1 = (1+ret_open)/(1+ret_close) - 1

이 day1_return이 "그날 하루 장중(시가->종가) 움직임"이고, 이게 순수종가체결과
순수시가체결의 최종 수익률 차이를 얼마나 설명하는지 확인.

청산일이 서로 다른 거래는 이 공식이 안 맞으므로 제외 (청산일이 같은 거래만 비교).

사용법 (레포 루트에서):
    python src/decompose_day1_contribution.py <ticker_krx>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_train_close_execute_open import (
    load_dataset, generate_hybrid_trades, NUM_DAYS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


if __name__ == "__main__":
    ticker_krx = sys.argv[1] if len(sys.argv) > 1 else "064350"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    df_close = load_dataset(ticker_krx, open_entry=False)
    df_open = load_dataset(ticker_krx, open_entry=True)

    # 순수종가 모델(seed 고정)이 고른 거래 목록 -- 이 날짜들에 대해서만 분해
    trades_close = generate_hybrid_trades(df_close, df_close, random_state=seed)
    print(f"순수종가 모델(seed={seed})이 고른 거래: {len(trades_close)}건")

    # 같은 날짜의 순수시가 결과와 대조 (청산일도 같은 거래만 남김)
    trades_open = generate_hybrid_trades(df_open, df_open, random_state=seed)

    tc = trades_close.set_index("entry_date")
    to = trades_open.set_index("entry_date")
    common = tc.index.intersection(to.index)
    print(f"공통 진입일: {len(common)}건")

    tc_c = tc.loc[common]
    to_c = to.loc[common]

    same_exit = tc_c["exit_date"] == to_c["exit_date"]
    print(f"그중 청산일까지 동일한 거래: {same_exit.sum()}건 (day1_return 계산 가능 대상)")

    tc_c = tc_c[same_exit]
    to_c = to_c[same_exit]

    ret_close = tc_c["gross_return"]  # 거래비용 반영 전 순수 가격수익률
    ret_open = to_c["gross_return"]

    day1_return = (1 + ret_open) / (1 + ret_close) - 1

    print(f"\nday1_return(진입 당일 시가->종가) 평균: {day1_return.mean():+.4%}")
    print(f"day1_return 분포:\n{day1_return.describe().round(4)}")

    total_gross_close = (1 + ret_close).prod() - 1
    total_gross_open = (1 + ret_open).prod() - 1
    print(f"\n(공통 거래만 기준) 종가체결 gross 복리: {total_gross_close:.1%}")
    print(f"(공통 거래만 기준) 시가체결 gross 복리: {total_gross_open:.1%}")

    print(f"\nday1_return > 0인 거래: {(day1_return > 0).sum()}건 / {len(day1_return)}건 "
          f"({(day1_return > 0).mean():.1%})")
    print(f"day1_return 절대값이 큰(>3%) 거래: {(day1_return.abs() > 0.03).sum()}건")

    print(f"\n=== day1_return 영향이 컸던 거래 상위 5건 ===")
    top5 = day1_return.abs().sort_values(ascending=False).head(5)
    for d in top5.index:
        print(f"{d.date()}: day1_return={day1_return[d]:+.2%}, "
              f"종가체결ret={ret_close[d]:+.2%}, 시가체결ret={ret_open[d]:+.2%}")