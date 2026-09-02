"""
Production 아티팩트 생성 -- 064350(현대로템), BASE 전용(밸류에이션 제외),
triple-barrier(pt_sl=(2,1), num_days=20), threshold=0.60.

build_production_models.py(3종목 COMBINED, [배포 부적합])의 컨벤션을 계승 --
이번엔 BASE 재검증에서 유일하게 살아남은 064350 단독 후보를 실전 배포 가능한
형태로 만듦 (docs/PROJECT_SUMMARY.md "[최종 확정] 064350 BASE 전용..." 참고).

[중요] walk-forward 백테스트용이 아니라 지금 시점까지의 전체 히스토리로 학습한
실전 배포용 모델임. 검증 때 썼던 walk-forward 분할은 과거 시뮬레이션을 위한
방법론이었을 뿐, 실전 배포에서는 사용 가능한 데이터를 전부 학습에 씀.

산출물:
    models/064350_triple_barrier_base_model.joblib
    models/064350_triple_barrier_base_metadata.json

사용법 (레포 루트에서):
    python src/build_production_model_064350_base.py
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from feature_engineering_triple_barrier import FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TICKER_KRX = "064350"
TICKER_NAME = "현대로템"
PT_SL = (2, 1)
NUM_DAYS = 20
THRESHOLD = 0.60  # backtest_base_only_triple_barrier.py + PBO 재검증으로 확정된 값
ROUND_TRIP_COST = 0.002

MODEL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42,  # ablation/backtest 때와 동일 시드 -- 재현성
)

MODEL_PATH = MODELS_DIR / f"{TICKER_KRX}_triple_barrier_base_model.joblib"
METADATA_PATH = MODELS_DIR / f"{TICKER_KRX}_triple_barrier_base_metadata.json"


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_pt2sl1_nd20_hl_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def train_final_model() -> dict:
    df = load_dataset()
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]

    print(f"=== {TICKER_NAME} ({TICKER_KRX}) BASE 전용 최종 모델 학습 ===")
    print(f"학습 데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})")
    print(f"라벨 분포: {y.value_counts(normalize=True).round(4).to_dict()}\n")

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS_BASE).sort_values(ascending=False)
    print("=== Feature Importance (전체 데이터 기준 재학습) ===")
    print(importance.round(4).to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "ticker_krx": TICKER_KRX,
        "ticker_name": TICKER_NAME,
        "feature_cols": FEATURE_COLS_BASE,
        "feature_set": "BASE_ONLY (밸류에이션 제외)",
        "label": "label_tb_binary (triple-barrier, label_tb > 0 -> 1)",
        "pt_sl": list(PT_SL),
        "num_days": NUM_DAYS,
        "fill_method": "D+1 종가체결 + High/Low 장중터치 반영 (최종 확정판)",
        "threshold": THRESHOLD,
        "round_trip_cost": ROUND_TRIP_COST,
        "model_path": str(MODEL_PATH.relative_to(MODELS_DIR.parent)),
        "xgb_params": MODEL_PARAMS,
        "n_training_rows": int(df.shape[0]),
        "training_date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "label_distribution": y.value_counts(normalize=True).round(4).to_dict(),
        "feature_importance": importance.round(4).to_dict(),
        "validation_summary": {
            "5seed_full_period": "5/5 (net_total_return 1260~1591%, B&H 기준 1219%)",
            "5seed_regime_excl_2025": "5/5 (net_total_return 276~453%, B&H excl-2025 198%)",
            "dominant_year_trade_level_excl": "2025년 거래 제외해도 절대수익률 379~526%로 견고, "
                                               "벤치마크 무너지지 않음",
            "threshold_axis_pbo": "19.4% (CSCV, 5개 후보 0.50~0.70, 10구간/252조합) -- 낮음",
            "pbo_is_preference_note": "IS에서는 0.50/0.55가 더 자주 최고로 뽑혔으나(126,99/252 "
                                       "vs 채택값 0.60은 16/252), 실제 5-seed 재검증에서 0.50은 "
                                       "전체기간 4/5로 근소 미달, 0.60은 5/5로 더 견고함을 직접 확인.",
        },
        "FINAL_VERDICT": (
            "[잠정 통과 -- 완전 배포 적합 아님] 5-seed, 거래단위 국면검증(자체 최대기여연도 "
            "2025 제외), threshold 축 PBO(19.4%, 낮음)까지 통과. 추가로 pt_sl 축도 별도 "
            "검증 완료: 대안 (1,1)은 5-seed 자체를 통과 못함(최대 2/5), (3,1)은 5-seed/"
            "국면검증/집중도까지 다 통과했지만 그 threshold(0.50) 자체의 축 PBO가 86.9%로 "
            "최악 수준이라 탈락 -- (2,1)/threshold=0.60이 두 대안보다 더 견고함을 직접 "
            "확인함(compute_pbo_pt_sl_064350.py, backtest_pt1sl1_064350.py, "
            "compute_pbo_threshold_pt3sl1_064350.py). 다만 pt_sl x threshold 전체 조합을 "
            "하나의 CSCV 후보 풀로 묶은 완전한 결합 PBO는 여전히 미실시이고, num_days=20 "
            "자체는 한 번도 검증 축에 넣어본 적 없음(예전 세션 고정값 상속)."
        ),
        "known_limitations": [
            "pt_sl x threshold 완전 결합 PBO 미실시 (각 축을 상대 축 고정한 채 따로 "
            "검증했음 -- pt_sl 축은 threshold=0.60 고정, threshold 축은 pt_sl=(2,1) 또는 "
            "(3,1) 고정)",
            "num_days=20 자체는 검증 축에 넣어본 적 없음 -- 다음 단계 후보",
            "052690/118990 탈락으로 종목 1개 단독 후보 -- 거래 수(전체기간 108~111건)가 "
            "예전 3종목 풀링(178~188건) 대비 적음",
            "밸류에이션 없이 통과했다는 게, 예전 COMBINED 결과가 착시였을 가능성을 시사하나 "
            "이 가설 자체는 별도로 검증 안 함 (COMBINED 라인은 무효 처리되어 재실행 안 함)",
            "실거래 슬리피지(주문 크기 대비 유동성 부족) 미반영",
            "포지션 사이징/자금관리 규칙 미검증",
        ],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장 완료: {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    metadata = train_final_model()
    print(f"\n{'=' * 60}\n{metadata['FINAL_VERDICT']}\n{'=' * 60}")