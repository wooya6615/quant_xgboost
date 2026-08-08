"""
DART 공시 빈도 feature를 기존 feature_engineering.py 결과에 붙이는 모듈

핵심 아이디어:
    저유동성 종목은 공시 이벤트(자사주매입, 유상증자, 실적 서프라이즈 등) 하나가 가격에
    크게 영향을 주는 경향이 있다는 가설. 이번 버전은 개별 공시의 내용(호재/악재)까지는
    분류하지 않고, 우선 "공시가 얼마나 자주 나오는가(발생 빈도)" 자체를 feature로 씀
    -- 텍스트 감성분석까지 가면 오버스펙이라, 빈도 카운트부터 시작.

    ⚠️ 룩어헤드 방지 -- 지금까지와 다른 종류의 타이밍 리스크:
    DART 공시는 장 시작 전, 장중, 장 마감 후 아무 때나 접수될 수 있음(수급/공매도처럼
    "장 마감 후에만" 발표되는 게 아님). 장중에 나온 공시는 이론적으로 당일 매매에도 쓸 수
    있지만, 그걸 정교하게 구분하려면 공시 접수 '시각'까지 파싱해야 해서 복잡도가 커짐.
    -> 이 버전은 보수적으로 수급/공매도와 동일하게 1일 shift를 적용함 (안전한 쪽으로 가정).
       장중 시각 기반 정밀화는 추후 확장 과제로 남겨둠.

    ⚠️ DART 접수일자(rcept_dt) vs 거래일 불일치:
    DART는 주말/공휴일에도 공시가 접수될 수 있음(예: 금요일 장마감 후 ~ 월요일 장시작 전
    사이 공시). 거래일 기준 rolling window로 집계하려면 접수일자를 거래일 인덱스에 맞게
    ffill/정렬해야 함 -- 아래 add_dart_features()에서 처리.

설치:
    pip install opendartreader --break-system-packages

사용법:
    python feature_engineering_dart.py
"""

import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # DART_API_KEY를 OpenDartReader 생성 전에 반드시 먼저 로드

from opendartreader import OpenDartReader

from feature_engineering import build_feature_dataset


# ------------------------------------------------------------------
# 1. DART 공시 목록 수집
# ------------------------------------------------------------------
def load_dart_disclosure_data(ticker_krx: str, start: str, end: str, sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    ticker_krx: 6자리 KRX 코드 (예: '064350')
    반환: 일별 공시 건수 (disclosure_count) -- 하루에 공시가 없으면 그 날짜는 아예 행이 없음(0건 의미)

    ⚠️ 참고: DART 원본 list() API는 corp_code를 지정하지 않고 조회하면 기간이 3개월로
    제한되지만, 특정 종목(corp_code/종목코드)을 지정하면 긴 기간도 한 번에 조회 가능함
    (OpenDartReader가 내부적으로 페이지네이션도 처리). 그래도 혹시 모를 응답 누락에
    대비해 1년 단위로 청크 분할해서 요청함 (공매도 실험 때 배운 방어적 패턴 재사용).
    """
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise ValueError("DART_API_KEY가 .env에 없습니다. https://opendart.fss.or.kr 에서 발급 후 추가하세요.")

    dart = OpenDartReader(api_key)

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    frames = []
    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end_dt)

        try:
            df = dart.list(
                ticker_krx,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                frames.append(df)
                print(f"  {chunk_start.date()}~{chunk_end.date()}: {len(df)}건")
            else:
                print(f"  {chunk_start.date()}~{chunk_end.date()}: 0건")
        except Exception as e:
            print(f"  {chunk_start.date()}~{chunk_end.date()}: 실패 ({type(e).__name__}: {e}) -- 건너뜀")

        time.sleep(sleep_sec)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not frames:
        raise ValueError(f"{ticker_krx}: DART 공시 데이터가 비어있습니다.")

    raw = pd.concat(frames, ignore_index=True)

    # rcept_dt: 접수일자 (YYYYMMDD 문자열)
    raw["rcept_dt"] = pd.to_datetime(raw["rcept_dt"], format="%Y%m%d")

    daily_count = raw.groupby("rcept_dt").size().rename("disclosure_count")
    daily_count.index.name = "Date"

    return daily_count.to_frame()


# ------------------------------------------------------------------
# 2. DART 공시 빈도 feature 생성
# ------------------------------------------------------------------
def add_dart_features(df: pd.DataFrame, disclosure_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    """
    df: build_feature_dataset()이 반환한 기존 feature 데이터셋 (거래일 인덱스)
    disclosure_df: load_dart_disclosure_data()가 반환한 일별 공시 건수 (주말/공휴일 포함 가능)

    ⚠️ 거래일 정렬: disclosure_df는 주말에도 값이 있을 수 있는데, join 대상인 df는 거래일만
    있음. 주말 공시는 "다음 거래일(월요일) 장 시작 전에 이미 알려진 정보"이므로, reindex로
    거래일에 맞춰 정렬하면 자연스럽게 다음 거래일로 넘어감 (별도 처리 불필요 -- reindex가
    없는 날짜의 값을 그냥 버리고, 그 정보는 join 시점에 가장 가까운 다음 거래일의 rolling
    윈도우에 반영됨).
    """
    merged = df.join(disclosure_df, how="left")
    merged["disclosure_count"] = merged["disclosure_count"].fillna(0)
    merged["disclosure_count"] = pd.to_numeric(merged["disclosure_count"], errors="coerce").astype("float64")

    count_known = merged["disclosure_count"].shift(shift_days)

    merged["dart_count_3d"] = count_known.rolling(3).sum()
    merged["dart_count_5d"] = count_known.rolling(5).sum()
    merged["dart_count_20d"] = count_known.rolling(20).sum()

    # 급증 신호: 최근 5일 공시 건수가 최근 60일 평균(5일 환산) 대비 몇 배인지
    baseline_5d_equivalent = count_known.rolling(60).mean() * 5
    merged["dart_burst_ratio_5d"] = merged["dart_count_5d"] / baseline_5d_equivalent.replace(0, np.nan)

    return merged


# ------------------------------------------------------------------
# 3. 실행 -- 기존 build_feature_dataset()에 DART 공시 빈도 feature를 얹어서 반환
# ------------------------------------------------------------------
def build_feature_dataset_with_dart(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
    shift_days: int = 1,
) -> pd.DataFrame:
    base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)

    disclosure_df = load_dart_disclosure_data(ticker_krx, start, end)
    result = add_dart_features(base, disclosure_df, shift_days=shift_days)

    dart_cols = ["dart_count_3d", "dart_count_5d", "dart_count_20d", "dart_burst_ratio_5d"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=dart_cols)
    return result


# ------------------------------------------------------------------
# 4. 멀티 horizon 저장
# ------------------------------------------------------------------
def build_multi_horizon_datasets_dart(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizons: list = None,
    shift_days: int = 1,
    save: bool = True,
) -> dict:
    if horizons is None:
        horizons = [1, 3, 5, 10]

    disclosure_df = load_dart_disclosure_data(ticker_krx, start, end)  # horizon 무관 -- 한 번만 조회해서 재사용

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_dart_features(base, disclosure_df, shift_days=shift_days)

        dart_cols = ["dart_count_3d", "dart_count_5d", "dart_count_20d", "dart_burst_ratio_5d"]
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=dart_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = f"{ticker_krx}_features_with_dart_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"    # 현대로템 -- 저유동성 종목에서 지금까지 효과가 컸던 종목으로 시작
    TICKER_KRX = "064350"
    HORIZONS = [1, 3, 5, 10]

    build_multi_horizon_datasets_dart(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)