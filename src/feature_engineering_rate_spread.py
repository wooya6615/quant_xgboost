"""
국고채 10년-3년 장단기 금리차 feature.

[가설] 기준금리는 연 8회만 바뀌는 계단함수라 정보량이 낮고, 짧은 테스트 구간
안에서는 그냥 "그 시기를 가리키는 우회 변수"처럼 작동할 위험이 있음. 대신
국고채 10년-3년 스프레드는 매일 실제로 값이 바뀌고, 경기 확장/침체 기대를
반영하는 전통적 선행지표라 XGBoost 분할 기준으로 정보량이 더 나을 것으로 기대.

데이터 출처: 한국은행 ECOS API, ecos-reader 패키지의 get_treasury_yield().
FX/수급 실험과 동일하게 look-ahead 방지 위해 1일 shift 적용.

사용법 (레포 루트에서):
    python src/feature_engineering_rate_spread.py

전제:
    pip install ecos-reader, .env에 ECOS_API_KEY 설정 (라이브러리 README로 키 이름 재확인),
    ecos.load_env() 호출 필요.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import ecos

from feature_engineering import build_feature_dataset

ecos.load_env()  # .env에서 ECOS_API_KEY 로드 (키 이름은 라이브러리 문서로 재확인)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_rate_spread_data(start: str = "2015-01-01", end: str = "2026-07-18") -> pd.DataFrame:
    start_ecos = start.replace("-", "")  # YYYY-MM-DD -> YYYYMMDD
    end_ecos = end.replace("-", "")

    y10 = ecos.get_treasury_yield(maturity="10Y", start_date=start_ecos, end_date=end_ecos)
    y3 = ecos.get_treasury_yield(maturity="3Y", start_date=start_ecos, end_date=end_ecos)

    y10 = y10.rename(columns={"value": "yield_10y"})[["date", "yield_10y"]]
    y3 = y3.rename(columns={"value": "yield_3y"})[["date", "yield_3y"]]

    merged = pd.merge(y10, y3, on="date", how="inner")
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.set_index("date").sort_index()
    merged["rate_spread_10y3y"] = merged["yield_10y"] - merged["yield_3y"]

    print(f"국고채 스프레드 데이터: {merged.shape[0]}행, "
          f"{merged.index.min().date()} ~ {merged.index.max().date()}")

    return merged


def add_rate_spread_features(df: pd.DataFrame, spread_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    # shift_days=1: FX/수급 실험과 동일한 look-ahead 방지 원칙
    s = spread_df[["rate_spread_10y3y"]].shift(shift_days)

    result = df.join(s, how="left")
    result["rate_spread_10y3y"] = result["rate_spread_10y3y"].ffill()  # 채권시장 휴장일 등 결측 보정
    result["rate_spread_chg_5d"] = result["rate_spread_10y3y"].diff(5)
    result["rate_spread_chg_20d"] = result["rate_spread_10y3y"].diff(20)
    result["rate_spread_ma20"] = result["rate_spread_10y3y"].rolling(20).mean()
    result["rate_spread_deviation_ma20"] = result["rate_spread_10y3y"] - result["rate_spread_ma20"]
    return result


def build_multi_horizon_datasets_rate_spread(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizons: list = None,
    save: bool = True,
) -> dict:
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    spread_df = load_rate_spread_data(start, end)  # 한 번만 조회해서 재사용

    rate_cols = ["rate_spread_10y3y", "rate_spread_chg_5d", "rate_spread_chg_20d", "rate_spread_deviation_ma20"]

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_rate_spread_features(base, spread_df)
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=rate_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = DATA_DIR / f"{ticker_krx}_features_with_rate_spread_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"   # 현대로템 -- 3종목 중 개별 검증이 제일 강했던 종목으로 시작
    TICKER_KRX = "064350"
    HORIZONS = [1, 3, 5, 10, 20]

    build_multi_horizon_datasets_rate_spread(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)