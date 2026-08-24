"""
052690(한전기술) 총 상승분이 특정 며칠에 몰려있는지 확인.
'꾸준한 우상향'이면 모델이 못 잡을 이유가 없는데, '급등 소수일 집중형'이면
triple-barrier(모멘텀 기반 진입, num_days=20)가 구조적으로 놓치기 쉬움.
"""
from pykrx import stock
from dotenv import load_dotenv
import numpy as np

load_dotenv()

ohlcv = stock.get_market_ohlcv_by_date("20150101", "20260718", "064350")
daily_ret = ohlcv["종가"].pct_change().dropna()

total_log_return = np.log1p(daily_ret).sum()
top_n = [5, 10, 20, 50]
for n in top_n:
    top_days_log_return = np.log1p(daily_ret).nlargest(n).sum()
    share = top_days_log_return / total_log_return
    print(f"상위 {n}일이 전체 로그수익률의 {share:.1%} 차지")

# 참고: 상위 급등일 날짜 자체도 확인
print("\n상위 10일 급등일:")
print(daily_ret.nlargest(10))