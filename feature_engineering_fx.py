"""
원달러 환율(USD/KRW) feature를 기존 feature_engineering.py 결과에 붙이는 모듈

핵심 아이디어:
    수출 비중이 큰 종목(현대로템 방산/철도, 대한제강 원자재 수입 등)은
    환율 모멘텀/변동성이 가격에 후행 반영될 여지가 있음.
    수급 데이터처럼 "발표 지연"은 없지만, 대신 다른 종류의 룩어헤드 리스크가 있음:

    ⚠️ 캘린더 불일치 주의:
    KRW=X는 사실상 24시간 거래(주말 제외)되는 반면 KRX는 평일 장중에만 거래됨.
    yfinance의 KRW=X 일별 종가는 통상 뉴욕 종가 기준으로 찍히는데, 이 시점은
    한국 장 마감(오후 3:30) '이후'임. 즉 같은 날짜(index)로 단순 join하면
    "그날 한국 장이 끝난 뒤에 결정된 환율 종가"를 그날 예측에 쓰는 셈이 되어
    수급 데이터의 발표 지연 문제와 본질적으로 동일한 룩어헤드가 생김.

    -> 그래서 수급 feature와 동일하게 1일 shift를 적용함.
       (안전한 쪽으로 가정 -- 정확한 타임스탬프 정렬보다 보수적으로 하루 늦춤)

설치:
    (yfinance는 feature_engineering.py에서 이미 사용 중이므로 추가 설치 불필요)

사용법:
    python feature_engineering_fx.py
"""

import pandas as pd
import numpy as np
import yfinance as yf

from feature_engineering import build_feature_dataset


FX_TICKER = "KRW=X"  # 원달러 환율 (USD/KRW)


# ------------------------------------------------------------------
# 1. 환율 데이터 수집
# ------------------------------------------------------------------
def load_fx_data(start: str, end: str, fx_ticker: str = FX_TICKER) -> pd.DataFrame:
    fx = yf.download(fx_ticker, start=start, end=end, auto_adjust=True)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    fx = fx[["Close"]].rename(columns={"Close": "fx_close"})
    fx = fx.dropna()
    fx.index.name = "Date"
    return fx


# ------------------------------------------------------------------
# 2. 환율 feature 생성
# ------------------------------------------------------------------
def add_fx_features(df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    """
    df: build_feature_dataset()이 반환한 기존 feature 데이터셋
    fx_df: load_fx_data()가 반환한 일별 원달러 종가

    ⚠️ 룩어헤드 방지: fx_close는 join 직후 반드시 1일 shift.
       (모듈 docstring의 캘린더 불일치 설명 참고)
    """
    merged = df.join(fx_df, how="left")
    merged["fx_close"] = merged["fx_close"].ffill()  # 휴장일 등으로 생긴 NaN 앞값으로 채움

    fx_known = merged["fx_close"].shift(1)  # 여기서 1일 shift 적용 -- 핵심

    # 원화 약세/강세 모멘텀 (5/10/20일 수익률) -- shift된 값 기준으로 계산해야 룩어헤드 없음
    merged["fx_return_5d"] = fx_known.pct_change(5)
    merged["fx_return_10d"] = fx_known.pct_change(10)
    merged["fx_return_20d"] = fx_known.pct_change(20)

    # 환율 변동성 (20일, 연율화 안 함 -- 가격 feature의 hist_vol_20d와 스케일 다르므로 그대로 둠)
    fx_daily_ret = fx_known.pct_change()
    merged["fx_vol_20d"] = fx_daily_ret.rolling(20).std()

    # 20일 이동평균 대비 이격도 -- "환율이 최근 평균보다 얼마나 튀었나" (평균회귀 신호 가설)
    fx_ma20 = fx_known.rolling(20).mean()
    merged["fx_deviation_ma20"] = (fx_known - fx_ma20) / fx_ma20

    return merged


# ------------------------------------------------------------------
# 3. 실행 -- 기존 build_feature_dataset()에 환율 feature를 얹어서 반환
# ------------------------------------------------------------------
def build_feature_dataset_with_fx(
    ticker: str = "064350.KS",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
    fx_ticker: str = FX_TICKER,
) -> pd.DataFrame:
    base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)

    fx_df = load_fx_data(start, end, fx_ticker)
    result = add_fx_features(base, fx_df)

    fx_cols = ["fx_return_5d", "fx_return_10d", "fx_return_20d", "fx_vol_20d", "fx_deviation_ma20"]
    # 기존 feature_engineering.py와 동일하게 inf/NaN 방어
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=fx_cols)
    return result


# ------------------------------------------------------------------
# 4. [신규] 멀티 horizon 저장
#    train_xgboost_ablation_fx.py의 run_horizon_sweep()이 기대하는
#    {ticker_krx}_features_with_fx_h{horizon}.csv 파일들을 한 번에 만들어줌.
#
#    ⚠️ 주의: build_feature_dataset()는 horizon마다 label(및 그로 인해 dropna되는 행)이
#    달라지므로, horizon마다 build_feature_dataset_with_fx()를 처음부터 다시 호출함.
#    (환율 데이터 자체는 캐싱해서 매번 다시 다운로드하지 않도록 최적화함)
# ------------------------------------------------------------------
def build_multi_horizon_datasets_fx(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizons: list = None,
    fx_ticker: str = FX_TICKER,
    save: bool = True,
) -> dict:
    if horizons is None:
        horizons = [1, 3, 5, 10]

    fx_df = load_fx_data(start, end, fx_ticker)  # horizon 무관 -- 한 번만 다운로드해서 재사용

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_fx_features(base, fx_df)

        fx_cols = ["fx_return_5d", "fx_return_10d", "fx_return_20d", "fx_vol_20d", "fx_deviation_ma20"]
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=fx_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = f"{ticker_krx}_features_with_fx_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "005930.KS"    # 현대로템 (yfinance용) -- 수급 실험에서 효과 컸던 종목으로 시작
    TICKER_KRX = "005930"
    HORIZONS = [1, 3, 5, 10]  # train_xgboost_ablation_fx.py의 run_horizon_sweep() 기본값과 동일

    build_multi_horizon_datasets_fx(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)