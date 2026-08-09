"""
국내 공매도(외국인/기관과는 별개, 개별 종목 공매도 거래량) feature 생성

핵심 발견:
    pykrx의 wrapper 함수 get_shorting_volume_by_date()는 긴 기간(약 2년 이상)을 한 번에
    요청하면 내부적으로 깨진 응답을 받아서 KeyError('거래량')로 죽는다.
    반면 그 wrapper가 내부에서 호출하는 raw 함수
    (pykrx.website.krx.get_shorting_trading_value_and_volume_by_date)는
    "한 번에 최대 약 730일(2년)"까지는 정상 응답을 준다는 게 실험으로 확인됨
    (2015-01-01~2016-12-31은 성공, ~2017-01-01부터 실패).

    -> 600일 단위 청크로 나눠서 여러 번 요청 + 이어붙이면 2015~2026 전체 이력 복원 가능.
    -> 단, 공매도 전면 금지 기간(2023-11-06~2025-03-31)에 걸리는 청크는 KRX가 데이터
       자체를 안 주므로 실패하거나 빈 응답이 옴 -- 이건 버그가 아니라 실제 그 기간에
       공매도가 없었다는 뜻이라 그냥 건너뛰면 됨.

⚠️ 로그인 주의: pykrx는 최초 호출 시 KRX_ID/KRX_PW로 로그인 세션을 만듦.
   반드시 load_dotenv()를 pykrx(또는 pykrx.website) import보다 먼저 실행해야 함.

사용법:
    python feature_engineering_short_kr.py
"""

import time
import datetime as dt
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # ⚠️ pykrx.website import보다 반드시 먼저 실행
from pykrx.website import krx

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


CHUNK_DAYS = 600           # 확인된 한계(730일)보다 여유 있게 잡은 청크 크기
BAN_START = "2023-11-06"   # 공매도 전면 금지 시작 (README 기준)
BAN_END = "2025-03-31"     # 공매도 전면 금지 종료 (README 기준)


# ------------------------------------------------------------------
# 1. 청크 단위로 raw 함수 호출 + 이어붙이기
# ------------------------------------------------------------------
def _date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    """YYYY-MM-DD 또는 YYYYMMDD 문자열을 받아 (start, end) 청크 리스트(YYYYMMDD)를 만듦"""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days), end_dt)
        chunks.append((cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def load_short_selling_kr(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    chunk_days: int = CHUNK_DAYS,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    """
    ticker: 6자리 종목코드 (pykrx용, 예: "064350")
    반환: 날짜 인덱스, 컬럼 [short_volume(공매도 거래량), buy_volume(매수 거래량), short_ratio(비중)]

    금지 기간(BAN_START~BAN_END)과 겹치는 청크는 실패해도 정상 -- 건너뛰고 계속 진행.
    """
    chunks = _date_chunks(start, end, chunk_days)
    frames = []
    skipped = []

    for chunk_start, chunk_end in chunks:
        try:
            raw = krx.get_shorting_trading_value_and_volume_by_date(chunk_start, chunk_end, ticker)
            if raw is None or raw.empty:
                skipped.append((chunk_start, chunk_end, "빈 응답"))
                continue

            volume = raw["거래량"].rename(columns={
                "공매도": "short_volume",
                "매수": "buy_volume",
                "비중": "short_ratio",
            })
            frames.append(volume)
            print(f"  {chunk_start}~{chunk_end}: {len(volume)}행 성공")

        except Exception as e:
            skipped.append((chunk_start, chunk_end, f"{type(e).__name__}: {e}"))
            print(f"  {chunk_start}~{chunk_end}: 실패 ({type(e).__name__}) -- 건너뜀 (금지 기간이면 정상)")

        time.sleep(sleep_sec)  # 과도한 연속 요청으로 인한 차단 방지

    if not frames:
        raise ValueError(f"모든 청크가 실패했어요. ticker={ticker} 확인 필요. 실패 내역: {skipped}")

    result = pd.concat(frames)
    result.index.name = "Date"
    result = result[~result.index.duplicated(keep="first")]  # 청크 경계 겹침 방지
    result = result.sort_index()

    if skipped:
        print(f"\n  건너뛴 청크 {len(skipped)}개 (대부분 공매도 금지 기간일 것):")
        for s, e, reason in skipped:
            print(f"    {s}~{e}: {reason}")

    return result


# ------------------------------------------------------------------
# 2. 공매도 feature 생성 (기존 US 버전과 동일한 설계 -- 1일 shift 필수)
# ------------------------------------------------------------------
def add_short_selling_features(df: pd.DataFrame, short_df: pd.DataFrame) -> pd.DataFrame:
    """
    ⚠️ 룩어헤드 방지: 공매도 거래량도 당일 장 마감 후 집계/발표되는 데이터라
    N일차 row에는 N-1일까지 발표분만 들어가야 함 (수급 데이터와 동일한 이유로 1일 shift).
    """
    merged = df.join(short_df, how="left")

    short_volume_known = merged["short_volume"].shift(1)
    short_ratio_known = merged["short_ratio"].shift(1)

    merged["short_qty_3d"] = short_volume_known.rolling(3).sum()
    merged["short_qty_5d"] = short_volume_known.rolling(5).sum()
    merged["short_weight_5d_avg"] = short_ratio_known.rolling(5).mean()

    # 숏커버링 신호 후보: 최근 3일 공매도량 합이 최근 10일 평균(3일 환산)의 절반 이하로 급감
    recent_avg_3d_equivalent = short_volume_known.rolling(10).mean() * 3
    merged["short_covering_signal"] = (
        short_volume_known.rolling(3).sum() < 0.5 * recent_avg_3d_equivalent
    ).astype(int)

    return merged


# ------------------------------------------------------------------
# 3. 실행 -- 기존 build_feature_dataset()에 공매도 feature를 얹어서 반환
# ------------------------------------------------------------------
def build_feature_dataset_with_short_kr(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
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

    short_df = load_short_selling_kr(ticker_krx, start, end)
    result = add_short_selling_features(base, short_df)

    short_cols = ["short_qty_3d", "short_qty_5d", "short_weight_5d_avg", "short_covering_signal"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=short_cols)
    return result


# ------------------------------------------------------------------
# 4. [신규] 여러 horizon 데이터셋을 한 번에 생성
#    -- KRX 공매도 데이터는 horizon과 무관하므로 한 번만 받아서 재사용 (요청 낭비 방지)
# ------------------------------------------------------------------
DEFAULT_HORIZON_COST_MAP = {
    1: 0.002,
    3: 0.003,
    5: 0.005,   # 기존 실험값 (기존엔 fold 6개로 결론 못 냈던 그 실험)
    10: 0.008,
}


def build_multi_horizon_short_datasets(
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

    print(f"=== {ticker_krx} 공매도 이력 1회 조회 (청크 단위) ===")
    short_df = load_short_selling_kr(ticker_krx, start, end)  # 한 번만 fetch
    print(f"공매도 원본 데이터: {short_df.shape[0]}행, {short_df.index.min().date()} ~ {short_df.index.max().date()}\n")

    datasets = {}
    for horizon in horizons:
        cost_threshold = horizon_cost_map.get(horizon, 0.005)
        print(f"--- horizon={horizon}, cost_threshold={cost_threshold} ---")

        base = build_feature_dataset(
            ticker=ticker, benchmark=benchmark, start=start, end=end,
            horizon=horizon, cost_threshold=cost_threshold,
        )
        result = add_short_selling_features(base, short_df)
        short_cols = ["short_qty_3d", "short_qty_5d", "short_weight_5d_avg", "short_covering_signal"]
        result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=short_cols)

        label_dist = result["label"].value_counts(normalize=True).to_dict()
        print(f"  shape: {result.shape}, 라벨 분포: {label_dist}")

        out_path = DATA_DIR / f"{ticker_krx}_features_with_short_kr_h{horizon}.csv"
        result.to_csv(out_path)
        print(f"  저장 완료: {out_path}\n")

        datasets[horizon] = result

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"
    TICKER_KRX = "064350"

    # 단일 horizon만 필요하면:
    # dataset = build_feature_dataset_with_short_kr(ticker=TICKER, ticker_krx=TICKER_KRX, horizon=5)

    # horizon 스윕 (1/3/5/10일) -- KRX 데이터는 한 번만 받고 재사용
    datasets = build_multi_horizon_short_datasets(ticker=TICKER, ticker_krx=TICKER_KRX)

    print("=== 전체 horizon별 요약 ===")
    for horizon, dataset in datasets.items():
        print(f"horizon={horizon}: {dataset.shape[0]}행, label=1 비율 {dataset['label'].mean():.3f}")