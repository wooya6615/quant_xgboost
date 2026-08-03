"""
밸류에이션 feature(PER/PBR/배당수익률) 추가 모듈

핵심 아이디어:
    가격/거래량 기술지표, 수급(순매수), 공매도 모두 결국 "시장 참여자들의 최근 행동"에서
    파생된 정보라 서로 겹칠 여지가 있음. 밸류에이션(PER/PBR)은 완전히 다른 축 --
    "재무제표 기준으로 지금 이 가격이 싼지 비싼지"라서 정보 소스 자체가 독립적.

    다만 이건 전통적으로 "단기 방향성"보다 "중장기 평균회귀"에 가까운 팩터라서,
    5일/10일 같은 짧은 horizon에서 얼마나 유효할지는 미지수 -- 검증 대상.

룩어헤드 관련 참고:
    PER/PBR은 그날 종가 기준으로 계산된 값(EPS/BPS는 이미 공시된 분기 실적)이라
    수급/공매도 데이터처럼 "장 마감 후 발표"가 아님. 가격 feature와 동일하게
    "당일 종가 시점에 이미 확정된 정보"로 취급 가능 -- 1일 shift 불필요.

사용법:
    python feature_engineering_valuation.py
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # ⚠️ pykrx import보다 반드시 먼저 실행
from pykrx import stock

from feature_engineering import build_feature_dataset


# ------------------------------------------------------------------
# 1. 밸류에이션 데이터 수집
# ------------------------------------------------------------------
def load_valuation(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker: 6자리 종목코드 (pykrx용)
    반환 컬럼: BPS, PER, PBR, EPS, DIV(배당수익률 %), DPS
    """
    df = stock.get_market_fundamental_by_date(
        fromdate=start.replace("-", ""),
        todate=end.replace("-", ""),
        ticker=ticker,
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# ------------------------------------------------------------------
# 2. 밸류에이션 feature 생성
# ------------------------------------------------------------------
def add_valuation_features(df: pd.DataFrame, valuation_df: pd.DataFrame, zscore_window: int = 252) -> pd.DataFrame:
    """
    df: build_feature_dataset()이 반환한 기존 feature 데이터셋
    valuation_df: load_valuation()이 반환한 일별 PER/PBR/DIV 등

    절대 PER/PBR은 업종/종목마다 기준이 다르므로, 그 종목 자체의 최근 1년(252거래일)
    분포 대비 z-score로 정규화 -- "지금이 이 종목 기준으로 싼 편인지 비싼 편인지"를 나타냄.
    """
    merged = df.join(valuation_df[["PER", "PBR", "DIV"]], how="left")

    merged["per"] = merged["PER"]
    merged["pbr"] = merged["PBR"]
    merged["div"] = merged["DIV"]

    per_mean = merged["PER"].rolling(zscore_window).mean()
    per_std = merged["PER"].rolling(zscore_window).std()
    merged["per_zscore_252d"] = (merged["PER"] - per_mean) / per_std

    pbr_mean = merged["PBR"].rolling(zscore_window).mean()
    pbr_std = merged["PBR"].rolling(zscore_window).std()
    merged["pbr_zscore_252d"] = (merged["PBR"] - pbr_mean) / pbr_std

    return merged


# ------------------------------------------------------------------
# 3. 실행
# ------------------------------------------------------------------
def build_feature_dataset_with_valuation(
    ticker: str = "005930.KS",
    ticker_krx: str = "005930",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
    cost_threshold: float = 0.005,
) -> pd.DataFrame:
    base = build_feature_dataset(
        ticker=ticker, benchmark=benchmark, start=start, end=end,
        horizon=horizon, cost_threshold=cost_threshold,
    )
    valuation_df = load_valuation(ticker_krx, start, end)
    result = add_valuation_features(base, valuation_df)

    valuation_cols = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=valuation_cols)
    return result


# ------------------------------------------------------------------
# 4. 여러 horizon 데이터셋을 한 번에 생성 (밸류에이션 데이터는 한 번만 조회, 재사용)
# ------------------------------------------------------------------
DEFAULT_HORIZON_COST_MAP = {
    1: 0.002,
    3: 0.003,
    5: 0.005,
    10: 0.008,
    20: 0.012,   # [추가] horizon=10에서 강한 신호가 나와서 더 긴 horizon도 확인
}


def build_multi_horizon_valuation_datasets(
    ticker: str,
    ticker_krx: str,
    horizons: list[int] = None,
    horizon_cost_map: dict[int, float] = None,
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
) -> dict[int, pd.DataFrame]:
    if horizons is None:
        horizons = sorted(DEFAULT_HORIZON_COST_MAP.keys())
    if horizon_cost_map is None:
        horizon_cost_map = DEFAULT_HORIZON_COST_MAP

    print(f"=== {ticker_krx} 밸류에이션 데이터 조회 ===")
    valuation_df = load_valuation(ticker_krx, start, end)
    print(f"밸류에이션 원본: {valuation_df.shape[0]}행, {valuation_df.index.min().date()} ~ {valuation_df.index.max().date()}\n")

    datasets = {}
    for horizon in horizons:
        cost_threshold = horizon_cost_map.get(horizon, 0.005)
        print(f"--- horizon={horizon}, cost_threshold={cost_threshold} ---")

        base = build_feature_dataset(
            ticker=ticker, benchmark=benchmark, start=start, end=end,
            horizon=horizon, cost_threshold=cost_threshold,
        )
        result = add_valuation_features(base, valuation_df)
        valuation_cols = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
        result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=valuation_cols)

        print(f"  shape: {result.shape}, 라벨 분포: {result['label'].value_counts(normalize=True).to_dict()}")

        out_path = f"{ticker_krx}_features_with_valuation_h{horizon}.csv"
        result.to_csv(out_path)
        print(f"  저장 완료: {out_path}\n")

        datasets[horizon] = result

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"

    datasets = build_multi_horizon_valuation_datasets(ticker=TICKER, ticker_krx=TICKER_KRX)

    print("=== 전체 horizon별 요약 ===")
    for horizon, dataset in datasets.items():
        print(f"horizon={horizon}: {dataset.shape[0]}행, label=1 비율 {dataset['label'].mean():.3f}")