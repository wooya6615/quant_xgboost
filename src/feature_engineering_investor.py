"""
수급 데이터(외국인/기관 순매수) feature를 기존 feature_engineering.py 결과에 붙이는 모듈

핵심 아이디어:
    기존 4개 feature 그룹(모멘텀/변동성/거래량/상대강도)은 전부 가격 시계열에서 파생됐음.
    그래서 서로 정보가 겹치고, 결국 "시장 국면을 확인"하는 것 이상을 못했음(vs_base_rate 상관계수 0.79).

    수급 데이터는 가격과 독립된 소스라서, 이 상관관계 구조를 깰 여지가 있음.
    "외국인이 며칠째 순매수 중인데 아직 가격에 안 반영됐다" 같은 정보는
    RSI/MACD로는 절대 못 만들어내는 신호이기 때문.

[수정] horizon 스윕 지원 추가
    - build_feature_dataset_with_investor()에 cost_threshold 파라미터 추가
      (horizon이 짧아질수록 N일 후 수익률 변동폭도 작아지므로 임계값을 같이 낮춰야
       라벨이 한쪽으로 쏠리는 걸 방지할 수 있음)
    - build_multi_horizon_datasets(): 여러 horizon에 대해 한 번에 데이터셋을 만들고
      각각 CSV로 저장하는 함수 추가. 매번 스크립트를 수동으로 고쳐 돌릴 필요 없이
      horizon별 라벨 분포/행 개수를 바로 비교할 수 있게 함.

설치:
    pip install pykrx

사용법:
    python feature_engineering_investor.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 KRX_ID / KRX_PW를 읽어와 os.environ에 등록
# ⚠️ pykrx는 import 시점에 바로 로그인 세션을 생성하므로, load_dotenv()는
#    반드시 "from pykrx import stock"보다 먼저 실행돼야 함 (순서 바꾸면 안 됨)
from pykrx import stock

from feature_engineering import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 1. 수급 데이터 수집
# ------------------------------------------------------------------
def load_investor_flow(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    일별 투자자별 순매수 대금을 가져옴 (단위: 원)

    ticker: 6자리 종목코드 (예: 삼성전자 = "005930")
            주의 -- feature_engineering.py의 yfinance ticker(".KS" 붙은 형태)와 다름.
            pykrx는 순수 6자리 코드만 받음.
    """
    df = stock.get_market_trading_value_by_date(
        fromdate=start.replace("-", ""),
        todate=end.replace("-", ""),
        ticker=ticker,
    )
    # pykrx 컬럼명: 금융투자/보험/투신/사모/은행/기타금융/연기금 등/기타법인/개인/외국인/기타외국인/전체
    # 필요한 것만 추려서 컬럼명 통일
    df = df.rename(columns={"기관합계": "inst_net", "외국인합계": "foreign_net"})
    df = df[["inst_net", "foreign_net"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


# ------------------------------------------------------------------
# 2. 수급 feature 생성
# ------------------------------------------------------------------
def add_investor_features(df: pd.DataFrame, investor_df: pd.DataFrame) -> pd.DataFrame:
    """
    df: build_feature_dataset()이 반환한 기존 feature 데이터셋 (Close, Volume 등 포함해야 함)
    investor_df: load_investor_flow()가 반환한 일별 순매수 데이터

    거래대금(Close * Volume) 대비 비율로 정규화 -- 종목마다 시가총액이 달라서
    순매수 금액 절대값만 쓰면 대형주/소형주 비교가 안 됨. 다종목 풀링 시 필수.
    """
    merged = df.join(investor_df, how="left")

    # ⚠️ 룩어헤드 방지: 수급 데이터는 당일 장 마감 후(저녁)에 발표됨.
    # 반면 backtest_simulation.py는 "당일 종가 매매"를 가정하므로,
    # N일차 row에는 N일 당일 수급이 아니라 N-1일까지 발표분만 들어가야 함.
    # (가격 feature는 종가 시점에 이미 확정된 정보라 당일 종가 가정이 성립하지만,
    #  수급 데이터는 그 종가 시점엔 아직 발표 전이라 하루 shift가 필수)
    foreign_net_known = merged["foreign_net"].shift(1)
    inst_net_known = merged["inst_net"].shift(1)

    # 순매수 3/5일 누적 (하루치 노이즈 방지) -- shift된 값 기준으로 계산
    merged["foreign_net_3d"] = foreign_net_known.rolling(3).sum()
    merged["foreign_net_5d"] = foreign_net_known.rolling(5).sum()
    merged["inst_net_3d"] = inst_net_known.rolling(3).sum()
    merged["inst_net_5d"] = inst_net_known.rolling(5).sum()

    # 거래대금 대비 순매수 강도 (정규화) -- 종목 간 비교/풀링 시 이 컬럼을 우선 사용
    daily_value = merged["Close"] * merged["Volume"]
    trading_value_20d = daily_value.rolling(20).sum()
    merged["foreign_net_ratio_5d"] = merged["foreign_net_5d"] / trading_value_20d
    merged["inst_net_ratio_5d"] = merged["inst_net_5d"] / trading_value_20d

    # 외국인+기관 동반 순매수 여부 (둘 다 순매수일 때만 1 -- 두 세력이 같은 방향일 때가 더 강한 신호라는 가설)
    merged["smart_money_aligned"] = (
        (merged["foreign_net_3d"] > 0) & (merged["inst_net_3d"] > 0)
    ).astype(int)

    return merged


# ------------------------------------------------------------------
# 3. 실행 -- 기존 build_feature_dataset()에 수급 feature를 얹어서 반환
# ------------------------------------------------------------------
def build_feature_dataset_with_investor(
    ticker: str = "005930.KS",
    ticker_krx: str = "005930",       # pykrx용 6자리 코드 (yfinance ticker와 별개로 넘겨야 함)
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 10,
    cost_threshold: float = 0.005,   # [추가] horizon 짧을수록 낮춰서 넘길 것
) -> pd.DataFrame:
    base = build_feature_dataset(
        ticker=ticker, benchmark=benchmark, start=start, end=end,
        horizon=horizon, cost_threshold=cost_threshold,
    )

    # 수급 데이터는 종목당 한 번만 받으면 되고 horizon과 무관하므로 그대로 재사용 가능
    investor_df = load_investor_flow(ticker_krx, start, end)
    result = add_investor_features(base, investor_df)

    investor_cols = [
        "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
        "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
    ]
    # 기존 feature_engineering.py와 동일하게 inf/NaN 방어
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=investor_cols)
    return result


# ------------------------------------------------------------------
# 4. [신규] 여러 horizon용 데이터셋을 한 번에 생성
#    - horizon이 짧아질수록 라벨 임계값도 낮춰야 하므로, horizon:cost_threshold 매핑을 인자로 받음
#    - 기본값은 대략적인 가이드라인일 뿐 -- 실제로는 생성 후 라벨 분포를 보고 조정 권장
# ------------------------------------------------------------------
DEFAULT_HORIZON_COST_MAP = {
    1: 0.002,   # 1일 변동폭은 작아서 임계값도 낮춤
    3: 0.003,
    5: 0.005,   # 기존 실험값 그대로
    10: 0.008,  # 누적 기간이 길어지는 만큼 임계값도 높임
}


def build_multi_horizon_datasets(
    ticker: str,
    ticker_krx: str,
    horizons: list[int] = None,
    horizon_cost_map: dict[int, float] = None,
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
) -> dict[int, pd.DataFrame]:
    """
    horizons에 지정한 모든 horizon에 대해 데이터셋을 만들고 각각 CSV로 저장.
    반환값은 {horizon: DataFrame} 딕셔너리 -- 바로 이어서 ablation/훈련에 쓸 수 있음.
    """
    if horizons is None:
        horizons = sorted(DEFAULT_HORIZON_COST_MAP.keys())
    if horizon_cost_map is None:
        horizon_cost_map = DEFAULT_HORIZON_COST_MAP

    datasets = {}
    for horizon in horizons:
        cost_threshold = horizon_cost_map.get(horizon, 0.005)
        print(f"\n--- horizon={horizon}, cost_threshold={cost_threshold} 생성 중 ---")

        dataset = build_feature_dataset_with_investor(
            ticker=ticker, ticker_krx=ticker_krx, benchmark=benchmark,
            start=start, end=end, horizon=horizon, cost_threshold=cost_threshold,
        )

        label_dist = dataset["label"].value_counts(normalize=True).to_dict()
        print(f"  shape: {dataset.shape}, 라벨 분포: {label_dist}")

        out_path = DATA_DIR / f"{ticker_krx}_features_with_investor_h{horizon}.csv"
        dataset.to_csv(out_path)
        print(f"  저장 완료: {out_path}")

        datasets[horizon] = dataset

    return datasets


if __name__ == "__main__":
    TICKER = "064350.KS"    # 현대로템 (yfinance용) -- 저유동성 종목에서 효과 더 컸던 종목
    TICKER_KRX = "064350"   # 현대로템 (pykrx/KIS용 6자리 코드)

    # 단일 horizon만 필요하면 이렇게:
    # dataset = build_feature_dataset_with_investor(ticker=TICKER, ticker_krx=TICKER_KRX, horizon=5)

    # horizon 스윕 (1/3/5/10일) -- 결과 나오면 train_xgboost_ablation.py의 run_horizon_sweep()으로 이어서 비교
    datasets = build_multi_horizon_datasets(ticker=TICKER, ticker_krx=TICKER_KRX)

    print("\n=== 전체 horizon별 요약 ===")
    for horizon, dataset in datasets.items():
        print(f"horizon={horizon}: {dataset.shape[0]}행, "
              f"label=1 비율 {dataset['label'].mean():.3f}")