"""
무작위 기저율(base rate) 확인 -- 이 전략(BASE+VALUATION triple-barrier,
pt_sl=2:1, num_days=20, threshold=0.65)이 "아무 종목이나 골라도" 이 정도
통과율이 나오는 건 아닌지 확인. 집중도 등 어떤 성과연관 기준도 쓰지 않고
순수 무작위 샘플링.

⚠️ random_state=42로 고정, 결과 보고 다시 뽑지 않음 (재추첨 자체가 사후선택).

사용법 (레포 루트에서):
    python src/sample_random_baseline.py
"""

import pandas as pd

SEED = 42
N_SAMPLE = 15

if __name__ == "__main__":
    # screen_universe.py의 stage2(2015년 이전 상장 + 데이터 충분, 104종목)에서
    # 집중도 기준 없이 순수 무작위 샘플링. 이미 검증한 13종목은 제외.
    already_tested = {"064350", "052690", "118990",
                       "064760", "035900", "000990", "082740", "058470",
                       "083450", "131290", "089030", "005290", "014680"}

    pool = pd.read_csv("data/screened_universe.csv", dtype={"ticker": str})
    pool = pool[~pool["ticker"].isin(already_tested)]

    sample = pool.sample(n=N_SAMPLE, random_state=SEED)
    print(sample[["ticker", "name", "market", "market_cap", "top20_concentration"]].to_string(index=False))
    sample.to_csv("data/random_baseline_sample.csv", index=False)