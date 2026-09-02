"""
Triple-barrier 라벨 + 수급(외국인/기관 순매수) feature COMBINED 데이터셋 생성.

배경: 지금까지 [배포 부적합] 판정을 받은 3종목 풀링 triple-barrier 전략(BASE+VALUATION)은
PBO(pt_sl 축) 95.6%로 무너졌음. 그런데 그 실험에는 이 프로젝트에서 유일하게 국면 제외
후에도 살아남았던 신호인 수급 데이터(feature_engineering_investor.py)가 한 번도 결합된
적이 없음 -- 수급 검증은 별도 라인(XGBoost 이진분류, 현대로템/삼성전자 단일종목)에서만
이뤄졌었음. 이 스크립트는 "triple-barrier 라벨링 + 수급 feature" 조합을 처음으로 만듦.

구조는 feature_engineering_triple_barrier.py의 build_triple_barrier_dataset_combined()와
동일 -- VALUATION 자리에 INVESTOR를 그대로 교체한 것. pt_sl=(2,1), num_days=20,
D+1 종가체결 + High/Low 장중터치 반영까지 기존 컨벤션 100% 유지 (전략 자체는 안 바꾸고
feature 소스만 바꿔서 비교해야 공정한 비교가 됨).

⚠️ 수급 데이터는 pykrx 사용 -- .env에 KRX_ID/KRX_PW 필요, load_dotenv()가
   pykrx import보다 먼저 실행돼야 함 (feature_engineering_investor.py가 이미 그렇게 처리함).

사용법 (레포 루트에서):
    python src/feature_engineering_triple_barrier_investor.py

전제:
    feature_engineering.py, feature_engineering_investor.py, labeling_triple_barrier.py와
    같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    load_data, add_momentum_features, add_volatility_features,
    add_volume_features, add_relative_strength_features, add_label,
)
from feature_engineering_investor import load_investor_flow, add_investor_features
from labeling_triple_barrier import build_shifted_barrier_labels

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]

FEATURE_COLS_INVESTOR = [
    "foreign_net_3d", "foreign_net_5d", "inst_net_3d", "inst_net_5d",
    "foreign_net_ratio_5d", "inst_net_ratio_5d", "smart_money_aligned",
]


def build_triple_barrier_dataset_investor(
    ticker: str = "064350.KS",
    ticker_krx: str = "064350",
    benchmark: str = "^KS11",
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon_fixed: int = 20,
    cost_threshold: float = 0.012,
    pt_sl: tuple = (2, 1),
    vol_span: int = 20,
    num_days: int = 20,
) -> pd.DataFrame:
    """
    BASE + INVESTOR(외국인/기관 순매수 7개) feature까지 합친 triple-barrier 데이터셋.
    feature_engineering_investor.py의 load_investor_flow()/add_investor_features()를
    그대로 재사용 -- 1일 shift(발표 지연 방지)는 add_investor_features() 내부에서 이미 처리됨.
    이건 triple-barrier의 체결 지연(D일 신호 -> D+1일 종가체결)과는 별개의 문제라
    두 shift가 겹쳐도 이중 방어일 뿐 문제 없음.
    """
    df, bench = load_data(ticker, benchmark, start, end)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_relative_strength_features(df, bench)

    # 기존 고정 horizon 라벨 (비교 기준선, BASE+VALUATION 버전과 동일하게 유지)
    df = add_label(df, horizon=horizon_fixed, cost_threshold=cost_threshold)
    df = df.rename(columns={"label": "label_fixed"})

    investor_df = load_investor_flow(ticker_krx, start, end)
    df = add_investor_features(df, investor_df)

    # triple-barrier 라벨 (체결 지연 1일 + D+1 종가체결로 확정 + High/Low로 장중 터치 반영)
    close = df["Close"]
    high = df["High"] if "High" in df.columns else None
    low = df["Low"] if "Low" in df.columns else None
    if high is None or low is None:
        print("경고: 데이터에 High/Low 컬럼이 없어서 종가만으로 배리어 터치 판정함 "
              "(장중 터치 놓칠 수 있음)")
    bins = build_shifted_barrier_labels(
        close, vol_span=vol_span, num_days=num_days, pt_sl=pt_sl, high=high, low=low,
    )

    df["label_tb"] = bins["label"]
    df["ret_tb"] = bins["ret"]
    touch_pos = close.index.get_indexer(bins["touch_time"])
    entry_pos = close.index.get_indexer(bins.index)
    df["holding_rows_tb"] = pd.Series(touch_pos - entry_pos, index=bins.index)

    feature_cols = [
        "Close", "Volume",
        *FEATURE_COLS_BASE, *FEATURE_COLS_INVESTOR,
        "future_return", "label_fixed",
        "label_tb", "ret_tb", "holding_rows_tb",
    ]

    result = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # BASE+VALUATION 버전과 동일한 원칙: 두 라벨 + 전체 feature가 다 유효한 행만 남김
    result = result.dropna(
        subset=FEATURE_COLS_BASE + FEATURE_COLS_INVESTOR + ["label_fixed", "label_tb"]
    )
    return result


if __name__ == "__main__":
    VALIDATED_TICKERS = {
        "064350": "064350.KS",  # 현대로템
        "052690": "052690.KS",  # 한전기술
        "118990": "118990.KQ",  # 모트렉스 (코스닥, .KQ 접미사)
    }
    PT_SL = (2, 1)
    NUM_DAYS = 20
    CONFIG_LABEL = f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"  # 기존 BASE+VALUATION 버전과 동일한 라벨 규약

    for TICKER_KRX, TICKER in VALIDATED_TICKERS.items():
        print(f"\n{'#' * 60}\n# {TICKER_KRX} ({TICKER}) -- BASE + INVESTOR\n{'#' * 60}")

        dataset = build_triple_barrier_dataset_investor(
            ticker=TICKER, ticker_krx=TICKER_KRX, pt_sl=PT_SL, num_days=NUM_DAYS,
        )
        print(f"생성된 데이터셋 shape: {dataset.shape}")

        tb_dist = dataset["label_tb"].value_counts(normalize=True)
        print(f"triple-barrier 라벨(label_tb) 분포:\n{tb_dist}")
        if tb_dist.get(0.0, 0) < 0.02:
            print("경고: label_tb=0 비율이 여전히 2% 미만 -- 배리어가 너무 타이트할 수 있음.")

        out_path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{CONFIG_LABEL}_investor.csv"
        dataset.to_csv(out_path)
        print(f"저장 완료: {out_path}")

    print("\n\n3종목 전부 생성 완료 -- backtest_triple_barrier_pooled_investor.py로")
    print("5-seed 풀링 백테스트, 이어서 compute_pbo_investor.py로 PBO 검증할 것.")