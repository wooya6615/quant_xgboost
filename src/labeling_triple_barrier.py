"""
Triple-barrier 라벨링 (Marcos Lopez de Prado, "Advances in Financial Machine Learning" 3장 방식).
기존 실험들의 "N일 후 방향 이진분류" 라벨을 대체하기 위한 범용 모듈 -- 특정 종목에 묶이지
않고, close 가격 시리즈(DatetimeIndex)만 주어지면 어떤 종목에도 적용 가능.

핵심 아이디어:
    "N일 후 오르나 내리나"가 아니라, "익절선(pt) / 손절선(sl) / 최대 보유기간(수직 배리어)
    중 뭘 먼저 치는가"로 라벨을 정의. 실제 트레이딩 로직(목표가/손절가 설정)에 훨씬 가까움.
    지금까지 quant_xgboost/quant_lightgbm/quant_ranking_kr에서 반복된 실패 패턴
    ("AUC/edge는 잡히는데 실전 백테스트/국면 검증에서 무너진다")이 라벨 정의 자체의
    문제일 수 있다는 가설을 검증하기 위한 첫 단계.

라벨 정의 (get_bins 반환):
    label =  1  익절 배리어를 먼저 침 (수익 실현, side 보정 후 기준)
    label = -1  손절 배리어를 먼저 침 (손실 확정)
    label =  0  수직 배리어(최대 보유기간)까지 둘 다 안 침, 그 시점 수익률이 정확히 0

사용 흐름 (1차 라벨링, side 없이 -- 순수 방향 분류 라벨 대체용):
    1. vol = get_daily_volatility(close)                       변동성 추정 (배리어 폭 결정)
    2. t1  = get_vertical_barrier(close, t_events, num_days)    수직 배리어 시점
    3. events = apply_triple_barrier(close, t_events, pt_sl, vol, t1=t1)
    4. bins = get_bins(events, close)                           -1/0/1 라벨 + 실현수익률

사용 흐름 (meta-labeling -- 1차 신호가 이미 있을 때):
    bins = get_meta_labels(primary_side, close, pt_sl, vol, num_days)
    -> bins["meta_label"] (0/1)을 2차 모델의 y로 사용

quant_xgboost 기존 컨벤션과의 접점:
    - FEATURE_COLS_BASE(모멘텀/변동성/거래량 13개)는 그대로 재사용 가능 -- 이 모듈은
      라벨(y)만 바꾸고 feature(X)는 건드리지 않음
    - t1(배리어를 친 실제 날짜)을 walk_forward_splits()의 embargo 대신 정밀하게 쓰면
      라벨 겹침 방지를 더 정확히 할 수 있음 (다음 단계에서 통합 예정)

주의:
    - close는 DatetimeIndex(오름차순 정렬)여야 함. yfinance/pykrx 데이터 로드 후
      df.sort_index() 거친 Close 시리즈를 그대로 넣으면 됨.
    - 아직 실제 종목 데이터로 검증 전 상태 -- 파일 하단 __main__에 합성 데이터로 만든
      자체 점검(sanity check)만 포함. 실제 종목 적용 전 이 점검부터 돌려서 라벨 분포가
      상식적인지(예: 상승 추세 데이터에 양의 라벨이 더 많이 나오는지) 확인할 것.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 1. 변동성 추정 -- 배리어 폭(pt_sl * target)을 동적으로 조정하는 데 씀.
#    고정폭 %를 쓰면 변동성 낮은 시기엔 너무 헐렁하고 높은 시기엔 너무 타이트해짐.
# ------------------------------------------------------------------
def get_daily_volatility(close: pd.Series, span: int = 20) -> pd.Series:
    """지수가중 이동 표준편차로 일별 변동성 추정."""
    prev_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prev_idx = prev_idx[prev_idx > 0]
    prev_dates = pd.Series(
        close.index[prev_idx - 1],
        index=close.index[close.shape[0] - prev_idx.shape[0]:],
    )
    daily_returns = close.loc[prev_dates.index] / close.loc[prev_dates.values].values - 1
    return daily_returns.ewm(span=span).std()


# ------------------------------------------------------------------
# 2. 수직 배리어(최대 보유기간) 시점 계산
# ------------------------------------------------------------------
def get_vertical_barrier(close: pd.Series, t_events: pd.DatetimeIndex, num_days: int) -> pd.Series:
    """
    각 이벤트 시점 t로부터 num_days 거래일 후를 수직 배리어로 지정.
    데이터 끝을 넘어가는 이벤트는 자동으로 제외됨 (아직 미결이라 라벨링 불가).
    """
    barrier_idx = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    valid = barrier_idx < close.shape[0]
    barrier_idx = barrier_idx[valid]
    return pd.Series(close.index[barrier_idx], index=t_events[valid])


# ------------------------------------------------------------------
# 3. 익절/손절/수직 배리어 중 뭘 먼저 쳤는지 탐색
# ------------------------------------------------------------------
def apply_triple_barrier(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple,
    target: pd.Series,
    min_ret: float = 0.0,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """
    pt_sl: (익절 배수, 손절 배수). target(변동성 추정치)에 곱해서 실제 배리어 %를 정함.
           예: (2, 1)이면 손절 대비 익절을 2배 넓게 잡는 비대칭 배리어. 0으로 주면 그
           배리어는 비활성화(수직 배리어까지 계속 보유).
    target: get_daily_volatility() 등에서 나온 이벤트별 변동성 추정치.
    min_ret: target이 이 값보다 작은(너무 조용한) 이벤트는 제외.
    t1: get_vertical_barrier() 결과. None이면 수직 배리어 없이 데이터 끝까지 봄.
    side: meta-labeling용 1(매수)/-1(매도) 방향. None이면 방향 무관하게 순수 가격
          움직임만 봄 (1차 라벨링).
    """
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]

    if t1 is None:
        t1 = pd.Series(close.index[-1], index=target.index)
    _side = pd.Series(1.0, index=target.index) if side is None else side.reindex(target.index).fillna(0)

    events = pd.concat({"t1": t1, "target": target, "side": _side}, axis=1).dropna(subset=["target", "t1"])

    out = pd.DataFrame(index=events.index)
    out["t1"] = events["t1"]
    out["side"] = events["side"]
    out["pt_touch"] = pd.NaT
    out["sl_touch"] = pd.NaT

    for loc, barrier_time in events["t1"].items():
        path_prices = close.loc[loc:barrier_time]
        if len(path_prices) < 2:
            continue
        path_returns = (path_prices / close.loc[loc] - 1) * events.at[loc, "side"]

        if pt_sl[0] > 0:
            pt_level = pt_sl[0] * events.at[loc, "target"]
            hits = path_returns[path_returns > pt_level]
            if not hits.empty:
                out.at[loc, "pt_touch"] = hits.index.min()

        if pt_sl[1] > 0:
            sl_level = -pt_sl[1] * events.at[loc, "target"]
            hits = path_returns[path_returns < sl_level]
            if not hits.empty:
                out.at[loc, "sl_touch"] = hits.index.min()

    out["touch_time"] = out[["pt_touch", "sl_touch", "t1"]].min(axis=1)
    return out


# ------------------------------------------------------------------
# 4. 최종 라벨(-1/0/1) + 실현 수익률 계산
# ------------------------------------------------------------------
def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    - touch_time에 실제 도달한 가격으로 수익률 계산 (진입가 대비, side 보정 반영)
    - side가 주어졌던 경우(meta-labeling), ret은 이미 "그 베팅이 이겼는가"를 나타냄
    """
    events = triple_barrier_events.dropna(subset=["touch_time"]).copy()
    out = pd.DataFrame(index=events.index)
    out["touch_time"] = events["touch_time"]
    out["side"] = events["side"]

    entry_px = close.loc[events.index]
    exit_px = close.loc[events["touch_time"].values].values
    out["ret"] = (exit_px / entry_px.values - 1) * events["side"].values
    out["label"] = np.sign(out["ret"]).astype(int)
    return out


# ------------------------------------------------------------------
# 5. Meta-labeling 진입점
# ------------------------------------------------------------------
def get_meta_labels(
    primary_side: pd.Series,
    close: pd.Series,
    pt_sl: tuple,
    target: pd.Series,
    num_days: int,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    1차 신호(primary_side: -1/0/1, 예를 들면 단순 모멘텀 부호)가 있을 때,
    "그 신호를 따라갔다면 익절선을 손절선보다 먼저 쳤는가"를 0/1로 라벨링.

    아이디어: 1차 모델(간단한 룰 기반)의 방향 정확도가 낮아도, 2차 모델이
    "이 신호를 실제로 딸지 말지"만 판단하게 하면 지금까지 반복된 "AUC는 되는데
    실전에서 못 이긴다"는 문제를 우회할 수 있을지 확인.

    반환 컬럼:
        meta_label: 1(베팅 성공, 익절 먼저 침) / 0(베팅 실패, 손절 먼저 침 또는 본전 이하)
        primary_side: 원래 1차 신호 방향 (참고용)
    """
    t_events = primary_side[primary_side != 0].index
    t1 = get_vertical_barrier(close, t_events, num_days)
    events = apply_triple_barrier(
        close, t_events, pt_sl, target, min_ret=min_ret, t1=t1,
        side=primary_side.loc[t_events],
    )
    bins = get_bins(events, close)
    bins["meta_label"] = (bins["label"] > 0).astype(int)
    bins["primary_side"] = primary_side.loc[bins.index]
    return bins


# ------------------------------------------------------------------
# 자체 점검 (실제 종목 데이터 없이도 돌아가는 sanity check)
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 합성 데이터 1: 뚜렷한 상승 추세 + 노이즈 -- 라벨이 양의 방향으로 쏠려야 정상
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    uptrend = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.015, n)))
    close_up = pd.Series(uptrend, index=dates)

    vol = get_daily_volatility(close_up, span=20)
    t_events = close_up.index[20:-15]  # 변동성 워밍업 구간 + 끝단 여유분 제외
    t1 = get_vertical_barrier(close_up, t_events, num_days=10)
    events = apply_triple_barrier(close_up, t_events, pt_sl=(1, 1), target=vol, t1=t1)
    bins = get_bins(events, close_up)

    print("=== 점검 1: 상승 추세 합성 데이터 (1차 라벨링, side 없음) ===")
    print(f"이벤트 수: {len(bins)}")
    print(bins["label"].value_counts().sort_index())
    print(f"라벨 1(익절) 비율: {(bins['label'] == 1).mean():.1%} -- 상승 추세니까 0.5보다 커야 정상")

    # 합성 데이터 2: meta-labeling -- 1차 신호를 "항상 매수(side=1)"로 주고
    # 상승 추세에서는 성공률이 높고, 하락 추세를 섞은 데이터에서는 낮아지는지 확인
    downtrend = 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.015, n)))
    close_down = pd.Series(downtrend, index=dates)

    always_buy = pd.Series(1, index=close_down.index[20:-15])
    vol_down = get_daily_volatility(close_down, span=20)
    meta_bins = get_meta_labels(always_buy, close_down, pt_sl=(1, 1), target=vol_down, num_days=10)

    print("\n=== 점검 2: 하락 추세 합성 데이터 + '항상 매수' 1차 신호 (meta-labeling) ===")
    print(f"이벤트 수: {len(meta_bins)}")
    print(f"meta_label=1(베팅 성공) 비율: {meta_bins['meta_label'].mean():.1%} "
          f"-- 하락 추세에서 '항상 매수'했으니 0.5보다 작아야 정상")

    assert (bins["label"] == 1).mean() > 0.5, "상승 추세인데 익절 비율이 절반 이하 -- 로직 점검 필요"
    assert meta_bins["meta_label"].mean() < 0.5, "하락 추세 + 항상매수인데 성공률이 절반 이상 -- 로직 점검 필요"
    print("\n두 점검 모두 통과 -- 기본 방향성(상승/하락 반응)은 정상 작동")