"""
체결 확정 후 실제 익절가/손절가(원화) 계산.

generate_daily_signal.py가 보여준 "익절폭 +X% / 손절폭 -Y%"는 어제(신호일) 기준
정보로 계산한 % 값이었음. 오늘 실제로 D+1 종가에 체결됐다면, 그 체결가를 넣어서
실제 원화 가격으로 확정하는 스크립트.

사용법 (레포 루트에서):
    python src/set_barrier_prices.py <종목코드> <체결가> <익절폭%> <손절폭%>
    예: python src/set_barrier_prices.py 064350 45000 12.3 6.15
        (generate_daily_signal.py 출력의 "익절폭(%)", "손절폭(%)" 값을 그대로 넣으면 됨)
"""

import sys


def compute_barrier_prices(entry_price: float, pt_pct: float, sl_pct: float) -> dict:
    """
    entry_price: 실제 체결가 (원)
    pt_pct, sl_pct: 퍼센트 단위 폭 (예: 12.3 = 12.3%). 둘 다 양수로 입력.
    """
    pt_price = entry_price * (1 + pt_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)
    return {"entry_price": entry_price, "pt_price": pt_price, "sl_price": sl_price}


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("사용법: python src/set_barrier_prices.py <종목코드> <체결가> <익절폭%> <손절폭%>")
        print("예:     python src/set_barrier_prices.py 064350 45000 12.3 6.15")
        sys.exit(1)

    ticker_krx = sys.argv[1]
    entry_price = float(sys.argv[2])
    pt_pct = float(sys.argv[3])
    sl_pct = float(sys.argv[4])

    result = compute_barrier_prices(entry_price, pt_pct, sl_pct)

    print(f"=== {ticker_krx} 배리어 확정 ===")
    print(f"체결가: {result['entry_price']:,.0f}원")
    print(f"익절가: {result['pt_price']:,.0f}원 (+{pt_pct:.2f}%)")
    print(f"손절가: {result['sl_price']:,.0f}원 (-{sl_pct:.2f}%)")
    print(f"\n이 가격을 실제 지정가 주문(익절)/손절 주문으로 걸어두거나,")
    print(f"장중 계속 지켜보다가 이 선을 건드리면 청산할 것.")
    print(f"만약 num_days(20거래일) 동안 둘 다 안 건드리면 그때 종가로 강제 청산 "
          f"(수직 배리어).")