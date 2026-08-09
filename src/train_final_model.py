"""
최종 프로덕션 모델: 현대로템(064350), BASE + 밸류에이션 COMBINED, horizon=20

선정 근거 (PROJECT_SUMMARY.md 참고):
    - 밸류에이션 실험에서 h=20이 COMBINED-BASE AUC 단조증가 패턴의 정점이었고,
      2025년(이례적 강세장)을 제외해도 COMBINED가 Buy & Hold를 이기는 게 재현됨
      (2015~2024년만으로: BASE -5.0% / COMBINED 64.1% / Buy&Hold 29.5%).
    - 지금까지 검증된 조합 중 "국면 우연이 아닌 진짜 edge"라는 근거가 가장 탄탄한 케이스.

⚠️ 이 스크립트가 지금까지의 ablation/backtest 스크립트와 다른 점:
    - 지금까지는 전부 Walk-Forward로 "검증"만 했음 (매 fold마다 새로 학습 -> 평가).
    - 여기서는 실제 배포를 가정하고, 사용 가능한 전체 히스토리로 최종 모델을 1번 학습해서
      저장함. 이 모델이 나중에 "다음 20일 방향"을 예측하는 데 쓰일 프로덕션 아티팩트임.
    - 하이퍼파라미터는 ablation 때 검증에 썼던 것과 동일하게 유지 (재현성 확보).

⚠️ 알아둬야 할 한계 (PROJECT_SUMMARY.md 4장 참고):
    - "AUC가 개선됐다"와 "backtest가 Buy&Hold를 이겼다"는 별개 결론이며, 이 모델도
      Buy & Hold 대비 절대 우위를 보장하지 않음 (BASE보다 낫다는 상대적 우위만 확인됨).
    - Walk-forward 검증 당시의 AUC(약 0.58~0.62대)가 미래에도 그대로 재현된다는 보장은 없음
      -- 시장 국면이 바뀌면(예: 밸류에이션 팩터가 안 먹히는 구간) 성능이 저하될 수 있음.
    - horizon=20(약 1개월)짜리 예측이라 신호 빈도가 낮음 -- 하루하루 트레이딩용이 아니라
      포지션을 몇 주 단위로 들고 가는 전략에 맞는 모델임.

사용법:
    python train_final_model.py              # 학습 + 저장
    python train_final_model.py --predict     # 저장된 모델로 최신 데이터 기준 예측만 실행

전제:
    feature_engineering_valuation.py, train_xgboost_valuation_ablation.py와 같은 폴더에 있어야 함.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from feature_engineering_valuation import build_feature_dataset_with_valuation
from train_xgboost_valuation_ablation import FEATURE_COLS_BASE, FEATURE_COLS_VALUATION


# ------------------------------------------------------------------
# 모델 설정 -- ablation/backtest 때 검증한 값과 반드시 일치시킬 것
# ------------------------------------------------------------------
TICKER = "064350.KS"
TICKER_KRX = "064350"
TICKER_NAME = "현대로템"
HORIZON = 20
COST_THRESHOLD = 0.012  # DEFAULT_HORIZON_COST_MAP[20] (feature_engineering_valuation.py)
FEATURE_COLS = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION  # COMBINED

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,  # ablation 기본 시드와 동일 -- 배포 모델도 하나의 시드로 고정
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / f"final_model_{TICKER_KRX}_h{HORIZON}.joblib"
METADATA_PATH = MODEL_DIR / f"final_model_{TICKER_KRX}_h{HORIZON}_metadata.json"


# ------------------------------------------------------------------
# 1. 전체 히스토리로 최종 모델 학습
# ------------------------------------------------------------------
def train_final_model(end: str = None):
    end = end or datetime.today().strftime("%Y-%m-%d")

    print(f"=== {TICKER_NAME} ({TICKER_KRX}) 최종 모델 학습 ===")
    print(f"horizon={HORIZON}, feature 수={len(FEATURE_COLS)}, 데이터 종료일={end}\n")

    df = build_feature_dataset_with_valuation(
        ticker=TICKER, ticker_krx=TICKER_KRX, horizon=HORIZON,
        cost_threshold=COST_THRESHOLD, end=end,
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["label"])
    print(f"학습 데이터: {df.shape[0]}행 ({df.index.min().date()} ~ {df.index.max().date()})")

    # ⚠️ horizon=20짜리 라벨은 데이터 맨 끝 20일치가 미래 정보 없이는 계산 불가 -- 이미
    # build_feature_dataset 내부에서 해당 구간은 dropna로 제거됨. 즉 "가장 최근 데이터"는
    # 실제로는 학습에 쓸 수 있는 마지막 라벨 확정 시점보다 과거임 (아래 predict_latest에서
    # 이 갭을 어떻게 다루는지 별도 설명).
    label_dist = df["label"].value_counts(normalize=True).to_dict()
    print(f"라벨 분포: {label_dist}\n")

    X = df[FEATURE_COLS]
    y = df["label"]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("=== Feature Importance (전체 데이터 기준 재학습) ===")
    print(importance.round(4).to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")

    metadata = {
        "ticker_krx": TICKER_KRX,
        "ticker_name": TICKER_NAME,
        "horizon": HORIZON,
        "cost_threshold": COST_THRESHOLD,
        "feature_cols": FEATURE_COLS,
        "xgb_params": XGB_PARAMS,
        "trained_at": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "train_data_start": str(df.index.min().date()),
        "train_data_end": str(df.index.max().date()),
        "train_n_rows": int(df.shape[0]),
        "label_distribution": label_dist,
        "feature_importance": importance.round(4).to_dict(),
        "validated_performance_reference": {
            "note": "walk-forward 검증 당시(ablation) 성능 -- 재학습된 이 모델의 실제 성능이 "
                     "아니라 참고용 기준선입니다. PROJECT_SUMMARY.md 2-3절 참고.",
            "combined_vs_base_auc_diff_h20": 0.0585,  # 대한제강 값은 별도 -- 로템은 아래 갱신 필요시 조정
            "backtest_note": "2025년 제외 재검증: BASE -5.0% / COMBINED 64.1% / Buy&Hold 29.5%",
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장 완료: {METADATA_PATH}")

    return model, df


# ------------------------------------------------------------------
# 2. 저장된 모델로 최신 데이터 기준 예측
# ------------------------------------------------------------------
def predict_latest():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"저장된 모델이 없습니다 ({MODEL_PATH}). 먼저 train_final_model()을 실행하세요.")
        return None

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"=== {metadata['ticker_name']} 최신 데이터 기준 예측 ===")
    print(f"모델 학습 시점: {metadata['trained_at']}\n")

    # ⚠️ 핵심 버그 수정: build_feature_dataset()은 마지막에 df[feature_cols].dropna()를
    # 호출하는데, feature_cols에 'label'/'future_return'이 포함돼 있음. 라벨은
    # horizon(20)일 후 수익률로 계산되므로 데이터 맨 끝 20영업일치는 항상 NaN이고,
    # 그 결과 build_feature_dataset()을 그대로 쓰면 최근 20영업일이 통째로 사라짐
    # (실제로 이전 버전에서 "기준일 2026-07-09"로 오늘보다 한 달 전이 나온 원인).
    #
    # 예측 시점에는 라벨이 애초에 필요 없으므로, add_label()을 거치지 않고 개별
    # feature 함수만 직접 호출해서 라벨 종속적인 dropna를 우회함.
    from feature_engineering import (
        load_data, add_momentum_features, add_volatility_features,
        add_volume_features, add_relative_strength_features,
    )
    from feature_engineering_valuation import load_valuation, add_valuation_features

    end = datetime.today().strftime("%Y-%m-%d")
    price_df, bench_df = load_data(TICKER, "^KS11", "2015-01-01", end)
    print(f"[진단] 가격 데이터 마지막 날짜: {price_df.index.max().date()}")

    price_df = add_momentum_features(price_df)
    price_df = add_volatility_features(price_df)
    price_df = add_volume_features(price_df)
    price_df = add_relative_strength_features(price_df, bench_df)
    # add_label()은 호출하지 않음 -- 예측 단계에서는 라벨 불필요, 이게 우회의 핵심

    valuation_df = load_valuation(TICKER_KRX, "2015-01-01", end)
    print(f"[진단] 밸류에이션 데이터 마지막 날짜: {valuation_df.index.max().date()}")
    print(f"[진단] 밸류에이션 데이터 최근 5행:\n{valuation_df.tail()}\n")

    full = add_valuation_features(price_df, valuation_df)
    full = full.replace([np.inf, -np.inf], np.nan)

    feature_cols = metadata["feature_cols"]

    # 어느 feature 그룹이 최신 날짜를 깎아먹는지 진단
    price_feature_cols = [c for c in feature_cols if c not in ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]]
    valuation_feature_cols = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]
    print(f"[진단] 가격 feature만으로 dropna한 마지막 날짜: "
          f"{full.dropna(subset=price_feature_cols).index.max().date()}")
    print(f"[진단] 밸류에이션 feature만으로 dropna한 마지막 날짜: "
          f"{full.dropna(subset=valuation_feature_cols).index.max().date()}\n")

    latest_valid = full.dropna(subset=feature_cols).iloc[[-1]]

    if latest_valid.empty:
        print("예측 가능한 최신 데이터가 없습니다 (feature 계산에 필요한 rolling window 부족).")
        return None

    latest_date = latest_valid.index[-1].date()
    proba = model.predict_proba(latest_valid[feature_cols])[0, 1]

    print(f"기준일: {latest_date}")
    print(f"COMBINED 모델의 {HORIZON}일 후 상승(라벨=1) 예측 확률: {proba:.4f}")

    print("\n--- 현재 밸류에이션 지표 ---")
    print(f"PER: {latest_valid['per'].values[0]:.2f}배 (z-score: {latest_valid['per_zscore_252d'].values[0]:+.2f})")
    print(f"PBR: {latest_valid['pbr'].values[0]:.2f}배 (z-score: {latest_valid['pbr_zscore_252d'].values[0]:+.2f})")
    print(f"배당수익률(DIV): {latest_valid['div'].values[0]:.2f}%")
    print("(z-score가 양수면 최근 1년 평균보다 비싼 편, 음수면 싼 편)")

    if proba >= 0.55:
        print("→ 신호: 매수 후보 (threshold 0.55 기준, ablation/backtest 검증 때와 동일 기준)")
    else:
        print("→ 신호: 관망 (threshold 미달)")

    print("\n⚠️ 이건 검증된 백테스트 threshold(0.55)를 그대로 적용한 참고 신호일 뿐, "
          "실거래 매수/매도 판단으로 곧바로 쓰지 말 것. PROJECT_SUMMARY.md의 한계 섹션 참고.")

    return proba


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict_latest()
    else:
        train_final_model()
        print("\n\n예측을 바로 보고 싶으면: python train_final_model.py --predict")