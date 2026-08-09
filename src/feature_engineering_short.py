"""
공매도 데이터(ka10014 공매도추이요청, 키움증권 공식 레포 기반) feature를 추가하는 모듈

핵심 아이디어:
    수급 데이터(외국인/기관 순매수) 다음 재료. "공매도가 급증했다가 숏커버링(재매수)되는
    시점"은 국내 시장에서도 잘 알려진 방향성 신호 후보. 공매도를 하려면 먼저 주식을
    빌려야 하므로, 공매도량 자체가 대차거래보다 더 직접적인 공매도 압력 지표임.

전제:
    Kiwoom-Securities/Kiwoom-REST-API 공식 레포의 `kiwoom` 패키지가 설치돼 있어야 함:
        pip install -e "경로\\Kiwoom-REST-API"
    kiwoomcli setup으로 인증이 이미 완료돼 있어야 함 (get_client()가 자격증명을 자동으로 읽어옴).

⚠️ 룩어헤드 주의:
    공매도 데이터의 정확한 공시 시점(당일 장중 실시간 vs 익일 집계)이 명확히 확인되지 않아서,
    수급 데이터(외국인/기관 순매수)와 동일하게 보수적으로 1일 shift를 적용함.
    나중에 실제 공시 시점이 당일 장중이라는 게 확인되면 shift를 제거해도 되지만,
    확인 전까지는 안전한 쪽(늦게 반영)으로 감.

사용법:
    python feature_engineering_short.py
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np
from kiwoom import get_client

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


API_ID = "ka10014"
API_URL = "/api/dostk/shsa"
MAX_PAGES = 10
REQUEST_DELAY_SECONDS = 0.2

COLUMNS = {
    "dt": "일자",
    "trde_qty": "거래량",
    "shrts_qty": "공매도량",
    "trde_wght": "매매비중",          # API가 이미 계산해서 주는 공매도 비중(%)
    "shrts_trde_prica": "공매도거래대금",
}


# ------------------------------------------------------------------
# 1. 공매도 데이터 수집 (examples/국내주식/공매도/get_domestic_short_sale_trend.py 로직을
#    우리 파이프라인에 맞게 재작성 -- DataFrame 반환, Date 인덱스, 컬럼명 통일)
# ------------------------------------------------------------------
def load_short_selling(ticker_krx: str, start: str, end: str) -> pd.DataFrame:
    """
    일별 공매도 데이터를 가져옴.

    ticker_krx: 6자리 종목코드
    start, end: "YYYY-MM-DD" (내부에서 YYYYMMDD로 변환)
    """
    client = get_client()
    body = {
        "stk_cd": ticker_krx,
        "strt_dt": start.replace("-", ""),
        "end_dt": end.replace("-", ""),
        "tm_tp": "1",
    }

    records = []
    next_cont_yn, next_key = None, None

    for page in range(MAX_PAGES):
        response = client.fetch_page(api_id=API_ID, path=API_URL, body=body,
                                      cont_yn=next_cont_yn, next_key=next_key)
        rows = response.body.get("shrts_trnsn", [])
        records.extend(r for r in rows if isinstance(r, dict))

        next_cont_yn = response.continuation.cont_yn
        next_key = response.continuation.next_key
        if next_cont_yn != "Y" or page + 1 >= MAX_PAGES:
            break
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(records).rename(columns=COLUMNS)
    df["일자"] = pd.to_datetime(df["일자"], format="%Y%m%d")
    df = df.set_index("일자").sort_index()  # API가 최신순으로 줄 수 있어 오름차순 정렬 필수
    df.index.name = "Date"

    for col in ["거래량", "공매도량", "매매비중", "공매도거래대금"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "거래량": "short_total_volume",
        "공매도량": "short_qty",
        "매매비중": "short_weight",
        "공매도거래대금": "short_value",
    })
    return df[["short_qty", "short_weight", "short_value", "short_total_volume"]]


# ------------------------------------------------------------------
# 2. 공매도 feature 생성
# ------------------------------------------------------------------
def add_short_selling_features(df: pd.DataFrame, short_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.join(short_df, how="left")

    # ⚠️ 룩어헤드 방지 -- 공시 시점 미확정이라 수급 데이터와 동일하게 보수적으로 shift
    short_qty_known = merged["short_qty"].shift(1)
    short_weight_known = merged["short_weight"].shift(1)

    merged["short_qty_3d"] = short_qty_known.rolling(3).sum()
    merged["short_qty_5d"] = short_qty_known.rolling(5).sum()
    merged["short_weight_5d_avg"] = short_weight_known.rolling(5).mean()

    # 숏커버링 신호 후보: 최근 3일 공매도량 합이 최근 10일 평균의 절반 이하로 급감
    # (공매도가 갑자기 줄어들면 그동안 빌린 주식을 되사는 매수 압력으로 이어질 수 있다는 가설)
    recent_avg_3d_equivalent = short_qty_known.rolling(10).mean() * 3
    merged["short_covering_signal"] = (
        short_qty_known.rolling(3).sum() < 0.5 * recent_avg_3d_equivalent
    ).astype(int)

    return merged


# ------------------------------------------------------------------
# 3. 실행
# ------------------------------------------------------------------
def build_feature_dataset_with_short(
    ticker: str = "005930.KS",
    ticker_krx: str = "005930",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
) -> pd.DataFrame:
    base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
    short_df = load_short_selling(ticker_krx, start, end)
    result = add_short_selling_features(base, short_df)

    short_cols = [
        "short_qty_3d", "short_qty_5d", "short_weight_5d_avg", "short_covering_signal",
    ]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=short_cols)
    return result


if __name__ == "__main__":
    TICKER = "005930.KS"
    TICKER_KRX = "005930"
    HORIZON = 5  # 수급 실험에서 유효했던 horizon 그대로 재현

    dataset = build_feature_dataset_with_short(ticker=TICKER, ticker_krx=TICKER_KRX, horizon=HORIZON)
    print(f"공매도 feature 포함 데이터셋 shape: {dataset.shape}")
    print(f"\n숏커버링 신호 비율:\n{dataset['short_covering_signal'].value_counts(normalize=True)}")
    print(f"\n샘플:\n{dataset[['short_qty_5d', 'short_weight_5d_avg', 'short_covering_signal']].tail()}")

    out_path = DATA_DIR / f"{TICKER_KRX}_features_with_short_h{HORIZON}.csv"
    dataset.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")