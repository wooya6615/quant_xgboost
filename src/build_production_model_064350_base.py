"""
Production 아티팩트 생성 -- 064350(현대로템), BASE 전용(밸류에이션 제외),
triple-barrier(pt_sl=(2,1), num_days=30), threshold=0.60.

[교체 이력] 이 스크립트는 원래 num_days=20으로 만들었던 첫 production 모델을
대체함. num_days 축 PBO(compute_pbo_num_days_064350.py, PBO 17.5%/낮음,
logit +2.92)에서 num_days=30이 IS 최고빈도 81%로 압도적이었고, 직접 5-seed
검증(backtest_num_days30_064350.py, threshold 5개 전부 5/5) + 그 자체의
threshold 축 PBO(compute_pbo_threshold_nd30_064350.py, PBO 18.3%/낮음,
IS 1위가 채택 threshold=0.60과 정확히 일치)까지 전부 모순 없이 통과해서
num_days=20 -> 30으로 최종 교체함. 상세 근거는 docs/PROJECT_SUMMARY.md
"[num_days 축 검증] nd=30으로 최종 교체..." 섹션 참고.

build_production_models.py(3종목 COMBINED, [배포 부적합])의 컨벤션을 계승 --
BASE 재검증에서 유일하게 살아남은 064350 단독 후보를 실전 배포 가능한 형태로
만듦.

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
NUM_DAYS = 30  # [교체됨] 기존 20 -> num_days 축 PBO + 직접 검증으로 30 확정
THRESHOLD = 0.60  # nd=30 기준 threshold 축 PBO에서도 IS 1위와 일치 확인됨
ROUND_TRIP_COST = 0.002

MODEL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42,  # ablation/backtest 때와 동일 시드 -- 재현성
)

MODEL_PATH = MODELS_DIR / f"{TICKER_KRX}_triple_barrier_base_model.joblib"
METADATA_PATH = MODELS_DIR / f"{TICKER_KRX}_triple_barrier_base_metadata.json"


def config_label() -> str:
    return f"pt{PT_SL[0]}sl{PT_SL[1]}_nd{NUM_DAYS}_hl"


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / f"{TICKER_KRX}_features_triple_barrier_{config_label()}_base.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if "label_tb_binary" not in df.columns:
        df["label_tb_binary"] = (df["label_tb"] > 0).astype(int)
    return df


def train_final_model() -> dict:
    df = load_dataset()
    X = df[FEATURE_COLS_BASE]
    y = df["label_tb_binary"]

    print(f"=== {TICKER_NAME} ({TICKER_KRX}) BASE 전용 최종 모델 학습 (num_days={NUM_DAYS}) ===")
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
        "superseded_config": {
            "note": "이 모델은 num_days=20으로 만들었던 첫 production 후보를 대체함",
            "old_pt_sl": [2, 1], "old_num_days": 20, "old_threshold": 0.60,
            "replacement_reason": "num_days 축 PBO(17.5%,낮음)+직접5-seed+threshold축 "
                                    "PBO(18.3%,낮음, IS1위=threshold=0.60 일치)까지 전부 "
                                    "모순 없이 통과 -- docs/PROJECT_SUMMARY.md 참고",
        },
        "validation_summary": {
            "5seed_full_period": "5/5 (threshold 5개 0.50~0.70 전부 5/5 통과, "
                                  "th=0.60 평균 net_total_return 4201%)",
            "5seed_regime_excl_dominant_year": "2024년(자체 최대기여연도) 제외, 5/5 통과, "
                                                "절대수익률 599~1088%, B&H(524%) 안 무너짐",
            "trade_concentration": "상위5개 거래 제외해도 부호 유지(+411~445%), 상위5개 거래 "
                                    "수익률이 시드 간 사실상 동일 -- 우연 아닌 근거",
            "num_days_axis_pbo": "17.5%(낮음), logit +2.920, IS 1위=30(81%, 채택값과 일치)",
            "threshold_axis_pbo_at_nd30": "18.3%(낮음), logit +1.542, IS 1위=threshold 0.60"
                                           "(60%, 채택값과 정확히 일치 -- 모순 없음)",
            "pt_sl_axis_pbo_at_nd20": "39.3%(중간) -- num_days=20 기준으로 검증했던 값, "
                                       "nd=30 기준으로는 재확인 안 함(다음 단계 후보)",
        },
        "FINAL_VERDICT": (
            "[잠정 통과 -- 완전 배포 적합 아님, 그러나 파라미터 검증은 이번 세션 기준 가장 "
            "충실하게 완료] pt_sl=(2,1)/num_days=30/threshold=0.60. num_days 축 PBO, "
            "직접 5-seed, 국면제외(자체 최대기여연도), 거래집중도, threshold 축 PBO까지 "
            "전부 모순 없이 통과한 유일한 조합 -- 이전 nd=20 버전보다 절대수익률도 훨씬 "
            "높고(4201% vs 1445%) 검증 일관성도 더 좋음(IS 선택과 백테스트 선택이 threshold "
            "축에서 정확히 일치). 다만 pt_sl 축은 num_days=20 기준으로만 검증했고 nd=30 "
            "기준으로는 재확인 안 함, pt_sl x num_days x threshold 완전 3축 결합 PBO도 "
            "미실시 -- 완전 배포 적합 판정까지는 아님."
        ),
        "known_limitations": [
            "pt_sl 축은 num_days=20 기준으로만 검증됨 (PBO 39.3%) -- num_days=30 기준으로 "
            "pt_sl 축을 재확인하지 않음, 다음 단계 후보",
            "pt_sl x num_days x threshold 완전 3축 결합 PBO 미실시 (2축씩만 결합 확인)",
            "052690/118990 탈락으로 종목 1개 단독 후보 -- 거래 수(nd=30 기준 전체기간 75~78건)가 "
            "예전 3종목 풀링(178~188건) 대비 적고, nd=20(106~111건)보다도 적음(보유기간이 "
            "길어져서 거래 회전율 자체가 낮아짐)",
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