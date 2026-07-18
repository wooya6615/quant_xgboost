"""
여러 종목을 풀링(pool)해서 하나의 데이터셋으로 만드는 스크립트
기존 feature_engineering.py의 build_feature_dataset()을 종목별로 반복 호출 후 합침

핵심: 종목을 섞어서 "특정 종목/섹터의 그 시기 국면"이 아니라
      "여러 섹터에 공통적으로 나타나는 패턴"을 모델이 배우게 하려는 목적

사용법:
    python feature_engineering_pooled.py
"""

import pandas as pd
from feature_engineering import build_feature_dataset


# 같은 섹터(반도체/AI)로 묶음 -- 가격 패턴이 비슷할 테니
# 모델이 'ticker'에 의존하지 않고 공통 기술적 패턴을 학습하는지 확인하려는 목적
DEFAULT_TICKERS = {
    "NVDA": "반도체/AI (GPU)",
    "AMD": "반도체/AI (GPU, 경쟁사)",
    "AVGO": "반도체 (통신칩)",
    "MU": "반도체 (메모리)",
    "TSM": "반도체 (파운드리)",
}

BENCHMARK = "SOXX"  # 반도체 섹터 ETF -- 개별 종목이 아니라 섹터 전체 대비 상대강도를 보기 위함


def build_pooled_dataset(tickers: dict = DEFAULT_TICKERS, benchmark: str = BENCHMARK,
                          start: str = "2016-01-01", end: str = "2026-07-18",
                          horizon: int = 10) -> pd.DataFrame:
    frames = []
    for ticker, sector in tickers.items():
        try:
            df = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        except Exception as e:
            print(f"{ticker} 데이터 생성 실패, 건너뜀: {e}")
            continue
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)
        print(f"{ticker} ({sector}): {len(df)}행")

    pooled = pd.concat(frames)
    pooled = pooled.sort_index()  # 날짜순 정렬 (종목은 섞여있음 -- 의도된 것)
    return pooled


if __name__ == "__main__":
    pooled = build_pooled_dataset()
    print(f"\n전체 풀링 데이터: {pooled.shape[0]}행")
    print(f"\n종목별 행 개수:\n{pooled['ticker'].value_counts()}")
    print(f"\n전체 라벨 분포:\n{pooled['label'].value_counts(normalize=True)}")

    pooled.to_csv("pooled_features.csv")
    print("\n저장 완료: pooled_features.csv")