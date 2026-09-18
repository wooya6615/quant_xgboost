# 사전등록: KAN vs XGBoost(BASE) vs 동일가중 — 064350 triple-barrier

브랜치: `experiment/kan-base-classifier`
작성일: 2026-09-18 (결과 확인 전, 이후 값 변경 금지 — 변경 필요시 addendum 남길 것)

## 배경

quant_cnn_chart(CNN), quant_seq_model(GRU/TCN) 모두 시퀀스/이미지 형태로 표현력을
늘리는 방향이었으나 동일가중 대비 일관된 우위를 못 보임. KAN은 방향이 다름 —
시퀀스가 아니라 **스칼라 피처 각각에 대해 학습되는 비선형 곡선(스플라인)**을 엣지에
두는 구조라, XGBoost가 쓰는 BASE 13개 피처에 그대로 적용 가능하고 엣지 함수를
시각화하면 해석까지 가능함. "모델을 더 유연하게 만들면 신호가 보이는지"를
XGBoost와 가장 가까운 조건(같은 피처, 같은 라벨, 같은 종목)에서 확인.

## 고정 파라미터

- 대상: 064350(현대로템) 단독 — production까지 간 유일한 종목이라 XGBoost와
  가장 공정한 비교 기준
- 라벨 프레임: triple-barrier, pt_sl=(2,1), num_days=30 (production과 동일 설정,
  `feature_engineering_triple_barrier.build_triple_barrier_dataset` 재사용)
- 피처: `FEATURE_COLS_BASE` (13개), 윈도우 없이 스칼라 그대로 (KAN은 시퀀스 구조가
  아니므로 GRU/TCN처럼 20일 펼치기 안 함 — XGBoost와 입력을 동일하게 맞추는 게 목적)
- Walk-forward: TRAIN=300 / TEST=60 / STEP=60 / EMBARGO=NUM_DAYS(=30)
  (기존 verify_base_only_triple_barrier.py와 동일한 스케일)
- 5-seed: 42 / 1 / 7 / 123 / 2024
- KAN 구조: `[13, 16, 2]`, grid_size=5, spline_order=3, base_activation=SiLU
  (하나의 설정으로 고정 후 seed 안정성부터 — ChartCNN/GRU/TCN 때와 동일 원칙)
- 정규화: fold별 train 통계로 z-score 후 [-1,1] soft-clip (spline grid 범위에 맞춤,
  train→test 누수 방지는 기존 normalize_fold와 동일 원칙)
- 학습: Adam lr=1e-3, epoch=30 (KAN이 spline 계산 때문에 MLP/GRU보다 느려서
  기존 EPOCHS=10보다 늘림 — 결과 보고 나서 조정하지 말 것, 필요하면 addendum),
  batch_size=64, 정규화 손실(entropy+activation) 가중치 1e-4 고정
- XGBoost 비교군: 기존 스크립트들과 동일한 XGB_PARAMS(n_estimators=200, max_depth=4,
  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)

## 1차 판정 기준

vs_base_rate(동일가중 대비 accuracy) 5/5 seed 전부 양수인지만 먼저 확인.
- KAN 5/5 실패, XGBoost도 5/5 실패 → 이 피처/라벨 조합 자체가 약한 신호라는
  기존 결론(학습된 교훈)을 재확인한 것으로 보고 접음
- KAN만 5/5 통과 → triple-barrier 실제 백테스트(비용 반영, threshold sweep,
  run_seq_experiment_triple_barrier.py와 동일 절차) 단계로 진행
- 둘 다 통과 → AUC/vs_base_rate 크기 비교 후 더 나은 쪽으로 백테스트 진행

AUC 개선은 채택 근거가 아님 — 최종 판단은 항상 비용 반영 백테스트 기준.