"""
Production 아티팩트 생성 -- 현대로템/한전기술/모트렉스 3종목 각각의 최종 XGBoost
모델을 전체 히스토리로 학습해서 joblib + 메타데이터 JSON으로 저장.

기존 train_final_model.py(1차, 현대로템 단독)의 컨벤션을 계승 -- 이번엔 3종목
풀링 후보(docs/candidate_model_triple_barrier_pooled.md)를 실제 배포 가능한
형태로 만드는 2차 버전.

[중요] 여기서 학습하는 모델은 walk-forward 백테스트용이 아니라 "지금 시점 기준
가장 최신 정보까지 다 써서 학습한, 앞으로 실전에 쓸 모델"임. 그러니까 지금까지
검증에 썼던 walk-forward 방식(과거 300일로 학습 -> 다음 60일 예측 반복)과 달리,
전체 데이터를 다 학습에 씀 -- 이건 "미래를 안다"는 게 아니라, 실전 배포 시점에는
그 시점까지의 모든 과거 데이터를 다 쓰는 게 당연하기 때문 (walk-forward는 과거
시뮬레이션을 위한 방법론이었을 뿐, 실전 배포에서는 불필요한 제약임).

산출물:
    models/{ticker}_triple_barrier_model.joblib   -- 학습된 XGBClassifier
    models/production_metadata.json                -- 설정/검증이력/사용법 기록

사용법 (레포 루트에서):
    python src/build_production_models.py
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS_BASE = [
    "return_5d", "return_10d", "return_20d", "rsi_14", "macd_hist",
    "hist_vol_20d", "bb_width", "bb_position", "atr_14",
    "volume_ratio_20d", "obv_change_20d",
    "excess_return_5d", "excess_return_20d",
]
FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

CONFIG_LABEL = "pt2sl1_nd20_hl"
TICKER_NAMES = {
    "064350": "현대로템",
    "052690": "한전기술",
    "118990": "모트렉스",
}
THRESHOLD = 0.65
PT_SL = (2, 1)
NUM_DAYS = 20
ROUND_TRIP_COST = 0.002
WEIGHT_PER_TICKER = round(1 / len(TICKER_NAMES), 4)

MODEL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42,
)


def load_dataset(ticker_krx: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{CONFIG_LABEL}_valuation.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def train_final_model(ticker_krx: str) -> dict:
    df = load_dataset(ticker_krx)
    X = df[FEATURE_COLS_COMBINED]
    y = df["label_tb_binary"]

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    model_path = MODELS_DIR / f"{ticker_krx}_triple_barrier_model.joblib"
    joblib.dump(model, model_path)

    return {
        "ticker": ticker_krx,
        "name": TICKER_NAMES[ticker_krx],
        "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
        "n_training_rows": len(df),
        "training_date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "label_distribution": y.value_counts(normalize=True).round(4).to_dict(),
    }


if __name__ == "__main__":
    print("=== Production 모델 학습 (3종목, 전체 히스토리) ===\n")

    per_ticker_info = []
    for ticker_krx in TICKER_NAMES:
        print(f"[{TICKER_NAMES[ticker_krx]} ({ticker_krx})] 학습 중...")
        info = train_final_model(ticker_krx)
        per_ticker_info.append(info)
        print(f"  -> {info['model_path']} 저장 완료 "
              f"({info['n_training_rows']}행, {info['training_date_range'][0]}~{info['training_date_range'][1]})")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "FINAL_VERDICT": (
            "[배포 부적합] pt_sl 축 PBO=95.6%로 심각한 과적합 위험 확인됨 "
            "(compute_pbo_v3.py). 채택값 pt_sl=(2,1)은 IS 1등 빈도 4위/5개(25/252회)에 "
            "불과하고, 이미 폐기했던 pt_sl=(1,1)이 97회로 가장 자주 1등. 5-seed/국면검증/"
            "체결타이밍수정을 다 통과했어도 이 결과는 배포하지 말 것 -- 참고용으로만 취급. "
            "자세한 내용은 docs/candidate_model_triple_barrier_pooled.md 참고."
        ),
        "strategy": {
            "labeling": "triple-barrier (López de Prado)",
            "pt_sl": list(PT_SL),
            "num_days": NUM_DAYS,
            "vol_scaling": "daily_vol * sqrt(num_days)",
            "entry_timing": "D일 종가까지 정보로 신호 -> D+1일 종가 체결 (MOC 주문)",
            "exit_detection": "High/Low 반영 -- 장중 배리어 터치 감지, 청산가는 배리어 트리거 가격",
            "features": FEATURE_COLS_COMBINED,
            "model_params": MODEL_PARAMS,
            "entry_threshold": THRESHOLD,
            "round_trip_cost_assumed": ROUND_TRIP_COST,
        },
        "portfolio": {
            "tickers": [{"code": t, "name": TICKER_NAMES[t], "weight": WEIGHT_PER_TICKER}
                        for t in TICKER_NAMES],
            "rebalancing": "동일가중, 각 종목 독립 모델의 신호에 따라 개별 진입/청산",
        },
        "validation_summary": {
            "note": "docs/candidate_model_triple_barrier_pooled.md 참고 -- 아래는 요약",
            "pooled_5seed_win_rate_full_period": "5/5 (blended net 512.0~943.9% vs blended Buy&Hold 485.1%, 고정 구간 기준)",
            "pooled_trade_count_range": "255~259건",
            "per_ticker_individual_caveat": "개별 종목 단독으로는 완벽하지 않음 (한전기술 전체기간 2/5, 모트렉스 2025이전절단 3/5) -- 반드시 3종목 포트폴리오로만 유효",
            "pbo_threshold_axis_only": "6.7% (compute_pbo.py, 낮음)",
            "pbo_threshold_x_portfolio_axis": "81.3% (compute_pbo_v2.py, 28개 후보, 매우 높음) -- 검증 범위를 넓히자 정반대 결론. 채택한 조합(th0.65, 3종목전체)은 구간별 IS 최고 후보 목록에 상위권으로 등장하지 않음",
        },
        "critical_warning": (
            "확장 PBO(threshold x 종목조합) 결과가 81.3%로 매우 높게 나옴 -- 이 문제 "
            "공간 자체가 노이즈가 크다는 뜻. 이 모델은 '완전히 검증된 결과'가 아니라 "
            "'여러 방법론적 결함을 고치고도 살아남았지만 문제 공간의 노이즈가 크다는 "
            "게 확인된 잠정적 후보'로 취급할 것. 실거래 투입 전 (1) pt_sl/num_days까지 "
            "포함한 완전한 PBO, (2) 소액 paper trading 기간, (3) 정기 재검증 필요."
        ),
        "known_limitations": [
            "pt_sl/num_days/체결방식 축까지 포함한 완전한 PBO는 미실시",
            "7종목 시도 중 3종목만 최종 통과 -- 통하는 종목의 공통 조건 미규명",
            "실거래 슬리피지(주문 크기 대비 유동성 부족) 미반영",
            "포지션 사이징/자금관리 규칙 미검증",
        ],
        "models": per_ticker_info,
    }

    metadata_path = MODELS_DIR / "production_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n메타데이터 저장 완료: {metadata_path}")
    print("\n=== 완료 ===")
    print(f"모델 3개 + 메타데이터 1개가 {MODELS_DIR}에 저장됐어.")
    print("실전 사용 시: 매일 장 마감 후 각 종목의 최신 feature를 계산해서 모델에 넣고,")
    print(f"proba >= {THRESHOLD}면 다음날(D+1) 종가에 진입, 나머지는 대기.")