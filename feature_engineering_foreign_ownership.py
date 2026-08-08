"""
외국인 보유율(지분율)/한도소진율 feature를 기존 feature_engineering.py 결과에 붙이는 모듈

핵심 아이디어:
    수급(외국인 순매수, flow) 실험과는 다른 종류의 신호. 순매수는 "오늘 얼마나 샀나"인 반면
    보유율/한도소진율은 "지금 얼마나 들고 있나"의 누적 레벨(stock) 데이터.
    -- 매일의 사고파는 노이즈보다 "외국인이 이 종목을 꾸준히 늘리고/줄이고 있는 추세" 자체를
       feature로 만드는 셈이라, 수급 feature와는 독립적인 정보일 가능성이 있음.

    ⚠️ 룩어헤드 방지 -- 발표 지연 이슈, 지금까지와 다른 점 주의:
    pykrx 공식 문서: "외국인 보유수량 및 한도소진율은 장개시 시점 기준(금융감독원 외국인투자
    관리시스템 제공 전일자 확정치)". 이 설명만 보면 수급 데이터와 동일하게 1일 shift로 충분함.

    그런데 pykrx 자체 문서(DeepWiki)에는 "외국인 보유 데이터는 D-2 영업일 데이터이며, D-1에는
    보유수량이 0으로 나온다"는 상반된 설명도 있음. 실제로 어느 쪽이 맞는지는 받아본 데이터에
    0값이 섞여 나오는지로 확인해야 해서, 이 스크립트는:
        1) 원본 데이터에서 보유수량/지분율이 0인 행의 비율을 출력 (데이터 품질 체크)
        2) 기본은 1일 shift를 적용하되, 0값 비율이 유의미하게 높으면 2일 shift로 바꿔야 함을
           경고 메시지로 안내
    로 방어적으로 설계함. (수급/공매도 실험 때는 이런 상충 정보가 없어서 1일 shift만으로 충분했음)

설치:
    (pykrx는 이미 investor/short 실험에서 사용 중이므로 추가 설치 불필요)

사용법:
    python feature_engineering_foreign_ownership.py
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # pykrx import 전에 반드시 먼저 실행 (KRX_ID/KRX_PW 로그인 순서 문제 방지)

from pykrx import stock

from feature_engineering import build_feature_dataset


# ------------------------------------------------------------------
# 1. 외국인 보유율/한도소진율 데이터 수집
# ------------------------------------------------------------------
def load_foreign_ownership_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    ticker: 6자리 KRX 코드 (예: '064350')
    반환 컬럼: 지분율(foreign_ownership_ratio), 한도소진율(foreign_limit_exhaustion)
    """
    fromdate = start.replace("-", "")
    todate = end.replace("-", "")

    raw = stock.get_exhaustion_rates_of_foreign_investment(fromdate, todate, ticker)
    if raw is None or raw.empty:
        raise ValueError(f"{ticker}: 외국인 보유율 데이터가 비어있습니다.")

    print(f"  [디버그] raw 컬럼명: {list(raw.columns)}")

    # pykrx 버전에 따라 '한도소진율'/'한도소진률' 표기가 다를 수 있어 동적으로 매칭
    exhaustion_col = next((c for c in raw.columns if "한도소진" in c), None)
    ownership_col = next((c for c in raw.columns if c == "지분율"), None)

    if ownership_col is None or exhaustion_col is None:
        raise KeyError(
            f"{ticker}: 지분율 또는 한도소진 관련 컬럼을 찾지 못했습니다. "
            f"실제 컬럼명: {list(raw.columns)}"
        )

    df = raw.rename(columns={
        ownership_col: "foreign_ownership_ratio",
        exhaustion_col: "foreign_limit_exhaustion",
    })[["foreign_ownership_ratio", "foreign_limit_exhaustion"]]

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()

    # 데이터 품질 체크: 0값 비율이 높으면 D-2 지연 가능성 -- 콘솔에 경고
    zero_ratio = (df["foreign_ownership_ratio"] == 0).mean()
    print(f"  [데이터 품질] 지분율=0인 행 비율: {zero_ratio:.1%}")
    if zero_ratio > 0.02:
        print("  ⚠️ 0값 비율이 2%를 넘습니다 -- pykrx wiki의 'D-1엔 0으로 나온다'는 설명과 일치할 수 있음.")
        print("     add_foreign_ownership_features()의 shift_days를 1 -> 2로 바꿔서 재확인 권장.")

    return df


# ------------------------------------------------------------------
# 2. 외국인 보유율 feature 생성
# ------------------------------------------------------------------
def add_foreign_ownership_features(df: pd.DataFrame, ownership_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    """
    df: build_feature_dataset()이 반환한 기존 feature 데이터셋
    ownership_df: load_foreign_ownership_data()가 반환한 일별 지분율/한도소진율
    shift_days: 발표 지연 방어용 shift (기본 1일, 데이터 품질 체크에서 0값 비율이 높으면 2로 변경)
    """
    merged = df.join(ownership_df, how="left")
    ratio_numeric = pd.to_numeric(merged["foreign_ownership_ratio"], errors="coerce").astype("float64")
    exhaustion_numeric = pd.to_numeric(merged["foreign_limit_exhaustion"], errors="coerce").astype("float64")
    merged["foreign_ownership_ratio"] = ratio_numeric.ffill()
    merged["foreign_limit_exhaustion"] = exhaustion_numeric.ffill()

    ratio_known = merged["foreign_ownership_ratio"].shift(shift_days)
    exhaustion_known = merged["foreign_limit_exhaustion"].shift(shift_days)

    # 레벨 변화 (수익률이 아니라 %p 차이 -- 이미 비율이므로 diff를 그대로 씀)
    merged["foreign_own_chg_5d"] = ratio_known.diff(5)
    merged["foreign_own_chg_20d"] = ratio_known.diff(20)

    # 가속도 신호: 최근 5일 변화폭이 그 이전 5일 변화폭보다 커지고 있는지 (누적 매수/매도 속도)
    merged["foreign_own_accel_5d"] = merged["foreign_own_chg_5d"].diff(5)

    # 한도소진율 -- 보유한도가 있는 종목에서만 의미 있음(대부분 종목은 한도 자체가 상장주식수와 동일해
    # 지분율과 거의 같은 값일 수 있음). 원 레벨 그대로 사용.
    merged["foreign_limit_exhaustion_level"] = exhaustion_known

    return merged


# ------------------------------------------------------------------
# 3. 실행 -- 기존 build_feature_dataset()에 외국인 보유율 feature를 얹어서 반환
# ------------------------------------------------------------------
def build_feature_dataset_with_foreign_ownership(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 5,
    shift_days: int = 1,
) -> pd.DataFrame:
    base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)

    ownership_df = load_foreign_ownership_data(ticker_krx, start, end)
    result = add_foreign_ownership_features(base, ownership_df, shift_days=shift_days)

    fo_cols = ["foreign_own_chg_5d", "foreign_own_chg_20d", "foreign_own_accel_5d", "foreign_limit_exhaustion_level"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=fo_cols)
    return result


# ------------------------------------------------------------------
# 4. 멀티 horizon 저장 (FX 실험 때 배운 대로, 처음부터 포함)
# ------------------------------------------------------------------
def build_multi_horizon_datasets_foreign_ownership(
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

    ownership_df = load_foreign_ownership_data(ticker_krx, start, end)  # horizon 무관 -- 한 번만 조회해서 재사용

    datasets = {}
    for horizon in horizons:
        base = build_feature_dataset(ticker=ticker, benchmark=benchmark, start=start, end=end, horizon=horizon)
        merged = add_foreign_ownership_features(base, ownership_df, shift_days=shift_days)

        fo_cols = ["foreign_own_chg_5d", "foreign_own_chg_20d", "foreign_own_accel_5d", "foreign_limit_exhaustion_level"]
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=fo_cols)

        datasets[horizon] = merged
        print(f"[horizon={horizon}] shape={merged.shape}")

        if save:
            out_path = f"{ticker_krx}_features_with_foreign_own_h{horizon}.csv"
            merged.to_csv(out_path)
            print(f"  저장 완료: {out_path}")

    return datasets


if __name__ == "__main__":
    TICKER = "005930.KS"
    TICKER_KRX = "005930"
    HORIZONS = [1, 3, 5, 10]

    build_multi_horizon_datasets_foreign_ownership(ticker=TICKER, ticker_krx=TICKER_KRX, horizons=HORIZONS)