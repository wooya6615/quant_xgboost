"""
DART 대량보유상황보고(5% Rule) feature를 기존 feature_engineering.py 결과에 붙이는 모듈
(feature_engineering_dart.py를 재사용하되, 전체 공시가 아니라 지분공시 중
 "대량보유상황보고서"만 필터링 -- 방금 실험에서 "빈도만으론 약하다"는 결과가 나와서,
 걸러야 할 신호를 걸러보는 후속 실험)

핵심 아이디어:
    일반 공시(자사주매입, 실적, 계약 등)를 전부 세는 건 노이즈가 컸을 가능성이 있음.
    대량보유상황보고서는 국민연금/기관/개인 대주주가 5% 이상 지분을 보유/변동시켰을 때
    의무적으로 내는 공시라, "확신을 가진 대형 자금의 포지션 변화"를 더 직접적으로 포착함.

    ⚠️ 이 이벤트는 훨씬 드묾 (일반 공시는 연 평균 75건이었는데, 대량보유상황보고서는
    한 종목 기준 연 수 건 수준일 수 있음). 그래서 feature 설계를 짧은 rolling count
    대신 "마지막 신고 이후 며칠 지났는지(recency)" + "긴 window 누적 건수" 위주로 바꿈.

    ⚠️ 방향(매수/매도) 미분류: report_nm(보고서명)만으로는 "5% 넘게 샀다"인지
    "5% 밑으로 팔았다"인지 구분이 안 됨 -- 그러려면 공시 원문(document.xml)을 파싱해서
    보고사유(신규/변동, 보유비율 증가/감소)까지 읽어야 하는데, 이번 버전은 우선
    "이벤트 발생 자체"만 feature화함. 방향 분류는 추후 확장 과제로 남겨둠.

    ⚠️ 룩어헤드 방지: feature_engineering_dart.py와 동일하게 1일 shift 적용
    (장중 발표 가능성을 보수적으로 처리).

설치:
    (feature_engineering_dart.py에서 이미 opendartreader 사용 중이므로 추가 설치 불필요)

사용법:
    python feature_engineering_dart_major_holder.py
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # DART_API_KEY를 OpenDartReader 생성 전에 반드시 먼저 로드

from opendartreader import OpenDartReader

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


MAJOR_HOLDER_KEYWORD = "대량보유상황보고서"


# ------------------------------------------------------------------
# 1. DART 지분공시 목록 수집 -- 대량보유상황보고서만 필터링
# ------------------------------------------------------------------
def load_major_holder_disclosure_data(ticker_krx: str, start: str, end: str, sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    ticker_krx: 6자리 KRX 코드 (예: '064350')
    반환: 일별 대량보유상황보고서 건수 (major_holder_count)
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
            # kind='D' -- 지분공시만 조회 (전체 공시보다 훨씬 적어서 API 호출 비용도 절약됨)
            df = dart.list(
                ticker_krx,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                kind="D",
            )
            if df is not None and not df.empty:
                # report_nm(보고서명)에 "대량보유상황보고서"가 포함된 것만 남김
                # (지분공시엔 "임원ㆍ주요주주특정증권등소유상황보고서" 등 다른 종류도 섞여있음
                #  -- 이건 임원 개인의 소액 지분 변동까지 다 잡혀서 노이즈가 크므로 제외)
                filtered = df[df["report_nm"].str.contains(MAJOR_HOLDER_KEYWORD, na=False)]
                if not filtered.empty:
                    frames.append(filtered)
                print(f"  {chunk_start.date()}~{chunk_end.date()}: 지분공시 {len(df)}건 중 대량보유 {len(filtered)}건")
            else:
                print(f"  {chunk_start.date()}~{chunk_end.date()}: 0건")
        except Exception as e:
            print(f"  {chunk_start.date()}~{chunk_end.date()}: 실패 ({type(e).__name__}: {e}) -- 건너뜀")

        time.sleep(sleep_sec)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not frames:
        print(f"  ⚠️ {ticker_krx}: 기간 내 대량보유상황보고서가 하나도 없습니다. "
              f"이 종목은 이 feature로 검증하기 부적합할 수 있어요.")
        # 빈 데이터프레임 반환 (add_major_holder_features에서 전부 0으로 처리됨)
        return pd.DataFrame(columns=["major_holder_count"]).rename_axis("Date")

    raw = pd.concat(frames, ignore_index=True)
    raw["rcept_dt"] = pd.to_datetime(raw["rcept_dt"], format="%Y%m%d")

    daily_count = raw.groupby("rcept_dt").size().rename("major_holder_count")
    daily_count.index.name = "Date"

    total = int(daily_count.sum())
    print(f"\n  [데이터 품질] {ticker_krx} 전체 기간 대량보유상황보고서 총 {total}건 "
          f"({(end_dt - start_dt).days / 365.25:.1f}년간)")
    if total < 10:
        print("  ⚠️ 10건 미만입니다 -- 이벤트가 너무 희소해서 학습 신호로 쓰기 어려울 수 있어요.")

    return daily_count.to_frame()


# ------------------------------------------------------------------
# 2. 대량보유상황보고서 feature 생성
# ------------------------------------------------------------------
def add_major_holder_features(df: pd.DataFrame, disclosure_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    merged = df.join(disclosure_df, how="left")
    merged["major_holder_count"] = merged["major_holder_count"].fillna(0)
    merged["major_holder_count"] = pd.to_numeric(merged["major_holder_count"], errors="coerce").astype("float64")

    count_known = merged["major_holder_count"].shift(shift_days)

    # 이벤트가 희소하므로 짧은 window(3/5일)보다 긴 window 위주로 설계
    merged["major_holder_count_20d"] = count_known.rolling(20).sum()
    merged["major_holder_count_60d"] = count_known.rolling(60).sum()

    # 최근 신고 발생 여부 (이진 신호) -- 20일 내 신고가 있었는지
    merged["major_holder_flag_20d"] = (merged["major_holder_count_20d"] > 0).astype(int)

    # 마지막 신고 이후 경과일 (recency) -- 신고가 아예 없던 기간엔 매우 큰 값(예: 9999)으로 처리
    event_dates = count_known[count_known > 0].index
    days_since = pd.Series(index=merged.index, dtype="float64")
    last_event = None
    for date in merged.index:
        if date in event_dates:
            last_event = date
        if last_event is not None:
            days_since[date] = (date - last_event).days
        else:
            days_since[date] = 9999  # 아직 한 번도 신고가 없었던 구간
    merged["days_since_major_holder_filing"] = days_since

    return merged


# ------------------------------------------------------------------
# 3. 실행
# ------------------------------------------------------------------
def build_feature_dataset_with_major_holder(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
    shift_days: int = 1,
) -> pd.DataFrame:
    base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)

    disclosure_df = load_major_holder_disclosure_data(ticker_krx, start, end)
    result = add_major_holder_features(base, disclosure_df, shift_days=shift_days)

    mh_cols = ["major_holder_count_20d", "major_holder_count_60d", "major_holder_flag_20d", "days_since_major_holder_filing"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=mh_cols)
    return result


# ------------------------------------------------------------------
# 4. 멀티 horizon 저장
# ------------------------------------------------------------------
def build_multi_horizon_datasets_major_holder(
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

    disclosure_df = load_major_holder_disclosure_data(ticker_krx, start, end)  # 한 번만 조회해서 재사용

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_major_holder_features(base, disclosure_df, shift_days=shift_days)

        mh_cols = ["major_holder_count_20d", "major_holder_count_60d", "major_holder_flag_20d", "days_since_major_holder_filing"]
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=mh_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = DATA_DIR / f"{ticker_krx}_features_with_major_holder_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"    # 현대로템
    TICKER_KRX = "064350"
    HORIZONS = [1, 3, 5, 10]

    build_multi_horizon_datasets_major_holder(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)
    