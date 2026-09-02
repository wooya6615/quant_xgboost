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

from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # ⚠️ pykrx import보다 반드시 먼저 실행
from pykrx import stock

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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
def add_valuation_features(df: pd.DataFrame, valuation_df: pd.DataFrame, shift_days: int = 1) -> pd.DataFrame:
    """
    [수정] PER<=0(적자 또는 계산불가)을 명시적으로 NaN 처리.

    배경: PER은 연 1회(매년 5월, 사업보고서 검토 후) 갱신되고 다음 해 4월까지
    고정값으로 유지됨. 적자연도(EPS<=0)면 PER이 0으로 기록되고 이 0이 1년
    내내 유지됨. 기존엔 이 0을 그대로 둬서 (1) 모델이 "극단적 저평가"로
    오독할 위험이 있었고 (2) rolling(252) 창이 적자연도로 가득 차면
    std=0 -> z-score NaN/inf -> dropna()로 그 구간 전체가 소실되면서도
    "정상 워밍업 감소"로 오인됐음 (010170 조사에서 발견, PROJECT_SUMMARY 참고).

    [해결 방식] 0으로 채우거나(왜곡) dropna로 통째로 버리는(데이터 손실 심각,
    특히 010170처럼 적자비율 80%+인 종목) 대신, NaN을 그대로 두고 XGBoost의
    네이티브 결측치 처리에 맡김 -- "밸류에이션 정보 없음"을 있는 그대로 표현.
    이 함수 안에서는 dropna를 호출하지 않음 -- 호출부(build_triple_barrier_
    dataset_combined 등)에서 밸류에이션 컬럼을 dropna 대상에서 제외해야 함.
    """
    v = valuation_df.shift(shift_days)

    per_clean = v["PER"].where(v["PER"] > 0, np.nan)

    result = df.copy()
    result["per"] = per_clean
    result["pbr"] = v["PBR"]
    result["div"] = v["DIV"]
    result["is_loss"] = (v["PER"] <= 0).astype(int)  # 적자(계산불가) 여부 -- 명시적 신호로 보존

    result["per_zscore_252d"] = (
        (result["per"] - result["per"].rolling(252, min_periods=60).mean())
        / result["per"].rolling(252, min_periods=60).std()
    )
    result["pbr_zscore_252d"] = (
        (result["pbr"] - result["pbr"].rolling(252, min_periods=60).mean())
        / result["pbr"].rolling(252, min_periods=60).std()
    )
    # ⚠️ 여기서 dropna 안 함 -- NaN은 XGBoost가 학습 중 자체적으로 분기 방향을 결정
    return result


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

        out_path = DATA_DIR / f"{ticker_krx}_features_with_valuation_h{horizon}.csv"
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