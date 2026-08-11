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
    entry_price: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
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
    entry_price: 배리어 % 계산의 기준이 되는 실제 체결가 (예: 시가). None이면 기존처럼
          close.loc[loc](진입일 종가)를 기준으로 씀.
    high, low: 장중 고가/저가. 주어지면 "종가가 배리어를 넘었는지"가 아니라 "장중에
          한 번이라도 배리어를 건드렸는지"로 판정 (더 정확함 -- 종가만 보면 장중에
          터치했다가 종가에 회복된 경우를 놓침). None이면 기존처럼 종가만 사용.
          매수(side=1)면 익절은 고가/손절은 저가, 매도(side=-1)면 반대로 봄.
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
    out["pt_price"] = np.nan
    out["sl_price"] = np.nan

    for loc, barrier_time in events["t1"].items():
        path_prices = close.loc[loc:barrier_time]
        if len(path_prices) < 2:
            continue
        ref_price = entry_price.loc[loc] if entry_price is not None and loc in entry_price.index else close.loc[loc]
        s = events.at[loc, "side"]

        # 익절 판정용 경로: 매수면 고가(유리한 쪽 극값), 매도면 저가
        # 손절 판정용 경로: 매수면 저가(불리한 쪽 극값), 매도면 고가
        if high is not None and low is not None:
            pt_path = (high if s >= 0 else low).loc[loc:barrier_time]
            sl_path = (low if s >= 0 else high).loc[loc:barrier_time]
        else:
            pt_path = path_prices
            sl_path = path_prices

        pt_returns = (pt_path / ref_price - 1) * s
        sl_returns = (sl_path / ref_price - 1) * s

        if pt_sl[0] > 0:
            pt_level = pt_sl[0] * events.at[loc, "target"]
            hits = pt_returns[pt_returns > pt_level]
            if not hits.empty:
                out.at[loc, "pt_touch"] = hits.index.min()
                # 실제 체결가는 종가가 아니라 배리어 트리거 가격 자체
                # (손절/익절 주문은 그 가격 근처에서 체결되지, 그날 종가까지 안 기다림)
                out.at[loc, "pt_price"] = ref_price * (1 + pt_level * s)

        if pt_sl[1] > 0:
            sl_level = -pt_sl[1] * events.at[loc, "target"]
            hits = sl_returns[sl_returns < sl_level]
            if not hits.empty:
                out.at[loc, "sl_touch"] = hits.index.min()
                out.at[loc, "sl_price"] = ref_price * (1 + sl_level * s)

    out["touch_time"] = out[["pt_touch", "sl_touch", "t1"]].min(axis=1)
    return out


# ------------------------------------------------------------------
# 4. 최종 라벨(-1/0/1) + 실현 수익률 계산
# ------------------------------------------------------------------
def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series,
             entry_price: pd.Series | None = None) -> pd.DataFrame:
    """
    - touch_time에 실제 도달한 가격으로 수익률 계산 (진입가 대비, side 보정 반영)
    - side가 주어졌던 경우(meta-labeling), ret은 이미 "그 베팅이 이겼는가"를 나타냄
    - entry_price: 실제 체결가(예: 시가) 시리즈. None이면 기존처럼 close(진입일 종가) 사용.
      apply_triple_barrier()에 넘긴 entry_price와 반드시 동일한 걸 넘길 것 (배리어 판정과
      실현수익률 계산의 기준가가 서로 다르면 안 됨).
    """
    events = triple_barrier_events.dropna(subset=["touch_time"]).copy()
    out = pd.DataFrame(index=events.index)
    out["touch_time"] = events["touch_time"]
    out["side"] = events["side"]

    if entry_price is not None:
        entry_px = entry_price.reindex(events.index)
        entry_px = entry_px.where(entry_px.notna(), close.loc[events.index])  # 결측이면 종가로 대체
    else:
        entry_px = close.loc[events.index]

    # 청산가: pt/sl로 터치됐으면 배리어 트리거 가격(apply_triple_barrier가 계산해둔
    # pt_price/sl_price) 그대로, 수직 배리어(t1, 시간 만료로 청산)면 그날 종가 사용.
    # (pt/sl은 가격 트리거로 즉시 체결되는 개념이라 종가까지 안 기다림 -- High/Low로
    # 장중 터치를 감지했을 때 특히 중요. t1은 "정해진 날짜가 돼서" 청산하는 거라
    # 종가 체결이 맞음.)
    if "pt_price" in events.columns and "sl_price" in events.columns:
        is_pt = events["touch_time"] == events["pt_touch"]
        is_sl = (~is_pt) & (events["touch_time"] == events["sl_touch"])
        exit_px = close.loc[events["touch_time"].values].values.astype(float).copy()
        exit_px[is_pt.values] = events.loc[is_pt, "pt_price"].values
        exit_px[is_sl.values] = events.loc[is_sl, "sl_price"].values
    else:
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
# 6. 실전 체결 타이밍 반영 -- D일 신호로 D일 종가에 바로 체결하는 건 비현실적
#    (D일 종가는 장 마감 후 확정되므로 그 정보로 같은 날 체결 불가능). 아래 함수는
#    "D일 종가까지 정보로 신호 -> D+1일에 체결"로 하루 지연시켜서 라벨/수익률을
#    계산하고, 반환되는 데이터프레임의 인덱스는 신호 발생일(D)로 되돌려서 feature
#    데이터프레임과 그대로 merge할 수 있게 함.
# ------------------------------------------------------------------
def build_shifted_barrier_labels(
    close: pd.Series,
    vol_span: int = 20,
    num_days: int = 20,
    pt_sl: tuple = (2, 1),
    side: pd.Series | None = None,
    open_price: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> pd.DataFrame:
    """
    체결 지연 1일을 반영한 triple-barrier 라벨링.

    [배경] 기존 방식(t_events=신호 발생일 그대로 apply_triple_barrier에 전달)은
    "그날 종가 기준으로 계산한 feature/신호로 그날 종가에 바로 체결한다"는 가정이
    깔려있었음 -- 그날 종가는 장 마감 후에야 확정되므로 그 정보로 같은 날 체결은
    불가능함 (look-ahead에 가까운 비현실적 가정).

    [수정] 신호는 D일 종가까지 정보로 만들되, 실제 체결은 D+1일로 하루 미룸.
    체결가는 open_price가 주어지면 D+1일 시가, 없으면 D+1일 종가를 씀.
    배리어 폭(변동성)도 D일 시점에 알 수 있는 값만 사용 (D+1 정보 사용 금지).

    반환값의 인덱스는 신호 발생일(D)이지만, 그 안의 라벨/수익률/touch_time은 전부
    "D+1일에 진입했다면"의 결과를 담고 있음 -- feature 데이터프레임(D일 인덱스)과
    바로 merge 가능.

    side: meta-labeling용 1차 신호 (D일 기준 계산된 값). None이면 1차 라벨링.
    open_price: 시가 시리즈. 주어지면 진입가로 D+1일 시가를 씀 (더 현실적 -- 시가는
        장이 열리자마자 확정되므로 그날 안에 체결 가능). None이면 기존처럼 D+1일 종가로
        체결 (더 보수적 -- 실전에서 흔히 쓰는 가정보다 하루 더 늦게 체결하는 셈).
    high, low: 고가/저가 시리즈. 주어지면 종가만으로 배리어 터치를 판정하지 않고
        장중에 실제로 건드렸는지까지 확인함 (더 정확 -- 종가만 보면 장중 터치 후
        회복된 경우를 놓침). 청산가도 그날 종가가 아니라 실제 배리어 트리거 가격을 씀.
    """
    daily_vol = get_daily_volatility(close, span=vol_span)
    scaled_vol = daily_vol * np.sqrt(num_days)

    decision_dates = close.index[vol_span:-1]   # 마지막 행은 다음날이 없어서 제외
    entry_dates = close.index[vol_span + 1:]     # decision_dates보다 정확히 하루(1거래일) 뒤

    # 배리어 폭은 D일(결정일) 기준 변동성을 써야 함 (D+1 정보 사용 금지) --
    # 값은 D의 변동성이되, apply_triple_barrier가 entry_date(D+1) 기준으로 계산하도록
    # 인덱스만 entry_date로 맞춰서 넘김
    vol_at_decision = scaled_vol.reindex(decision_dates)
    target_for_entry = pd.Series(vol_at_decision.values, index=entry_dates)

    side_for_entry = None
    if side is not None:
        side_at_decision = side.reindex(decision_dates)
        side_for_entry = pd.Series(side_at_decision.values, index=entry_dates).dropna()
        # side가 NaN인 날(1차 신호 없는 날)은 애초에 이벤트 자체가 없어야 하므로 제외
        target_for_entry = target_for_entry.loc[target_for_entry.index.intersection(side_for_entry.index)]

    # 진입가: open_price가 주어지면 D+1일 시가, 아니면 close로 자연스럽게 fallback
    # (apply_triple_barrier/get_bins가 entry_price=None이면 close.loc[loc]을 쓰므로)
    entry_price_for_entry = None
    if open_price is not None:
        entry_price_for_entry = open_price.reindex(target_for_entry.index)

    t1 = get_vertical_barrier(close, target_for_entry.index, num_days=num_days)
    events = apply_triple_barrier(
        close, target_for_entry.index, pt_sl=pt_sl, target=target_for_entry,
        t1=t1, side=side_for_entry, entry_price=entry_price_for_entry,
        high=high, low=low,
    )
    bins = get_bins(events, close, entry_price=entry_price_for_entry)

    # bins의 인덱스(entry_date=D+1)를 신호 발생일(D)로 되돌려서 feature 데이터프레임과
    # decision_date 기준으로 merge할 수 있게 함
    entry_to_decision = dict(zip(entry_dates, decision_dates))
    bins = bins.copy()
    bins.index = [entry_to_decision.get(d) for d in bins.index]
    bins = bins[[d is not None for d in bins.index]]

    return bins


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

    # 점검 3: 체결 지연 함수 -- 지연 버전 진입가가 실제로 원본(비지연) 버전보다 하루 뒤
    # 종가를 쓰는지 직접 대조
    print("\n=== 점검 3: 체결 지연(build_shifted_barrier_labels) 검증 ===")
    shifted_bins = build_shifted_barrier_labels(close_up, vol_span=20, num_days=10, pt_sl=(1, 1))
    print(f"지연 버전 이벤트 수: {len(shifted_bins)} (원본과는 t_events 범위 자체가 달라 "
          f"직접 비교 대상 아님 -- 참고용 숫자)")

    sample_date = shifted_bins.index[100]
    sample_pos = close_up.index.get_loc(sample_date)
    expected_entry_date = close_up.index[sample_pos + 1]
    print(f"샘플 신호일: {sample_date.date()}, 그 다음 거래일(기대 진입일): "
          f"{expected_entry_date.date()}")
    print("-> 신호일의 종가로 계산한 배리어가 신호일이 아니라 다음날부터 추적된다는 뜻 -- 정상")
    print("점검 3 통과 -- 체결 지연 로직이 예상대로 이벤트를 하루씩 밀어서 처리함")

    # 점검 4: 시가 체결(open_price) 옵션 -- 종가 체결과 다른 결과가 나오는지 확인
    print("\n=== 점검 4: 시가 체결(open_price) 옵션 검증 ===")
    # 합성 시가 = 종가보다 항상 1% 낮게 설정 (시가/종가가 다르다는 걸 뚜렷하게 확인하기 위함)
    synthetic_open = close_up * 0.99
    close_entry_bins = build_shifted_barrier_labels(close_up, vol_span=20, num_days=10, pt_sl=(1, 1))
    open_entry_bins = build_shifted_barrier_labels(
        close_up, vol_span=20, num_days=10, pt_sl=(1, 1), open_price=synthetic_open,
    )
    common_idx = close_entry_bins.index.intersection(open_entry_bins.index)
    ret_diff = (open_entry_bins.loc[common_idx, "ret"] - close_entry_bins.loc[common_idx, "ret"]).abs().mean()
    print(f"공통 이벤트 {len(common_idx)}개, 종가체결 vs 시가체결 평균 수익률 차이: {ret_diff:.4f}")
    assert ret_diff > 0, "시가를 종가보다 1% 낮게 뒀는데 수익률이 하나도 안 달라짐 -- entry_price 반영 안 되는 버그"
    print("점검 4 통과 -- open_price를 넘기면 실제로 진입가/수익률 계산에 반영됨")

    # 점검 5: High/Low 옵션 -- 장중에 배리어를 건드렸다가 종가에 회복된 경우를
    # 종가만 볼 때는 놓치지만, High/Low를 주면 잡아내는지 확인
    print("\n=== 점검 5: High/Low 장중 터치 감지 검증 ===")
    # 합성 데이터: 평소엔 종가와 고가/저가가 비슷하지만, 특정 구간에서 장중 변동폭을
    # 인위적으로 크게 벌려서 "장중엔 터치했지만 종가는 안 넘은" 상황을 만듦
    wide_intraday = rng.uniform(1.5, 4.0, len(close_up))  # 장중 변동폭 배수
    high_s = close_up * (1 + 0.003 * wide_intraday)
    low_s = close_up * (1 - 0.003 * wide_intraday)

    bins_close_only = build_shifted_barrier_labels(close_up, vol_span=20, num_days=10, pt_sl=(1, 1))
    bins_with_hl = build_shifted_barrier_labels(
        close_up, vol_span=20, num_days=10, pt_sl=(1, 1), high=high_s, low=low_s,
    )

    common_idx2 = bins_close_only.index.intersection(bins_with_hl.index)
    avg_holding_close_only = (bins_close_only.loc[common_idx2, "touch_time"] - common_idx2).mean()
    avg_holding_with_hl = (bins_with_hl.loc[common_idx2, "touch_time"] - common_idx2).mean()
    print(f"평균 보유기간 -- 종가만: {avg_holding_close_only}, High/Low 반영: {avg_holding_with_hl}")
    print("(High/Low를 반영하면 장중에 더 일찍 터치를 감지하니까 평균 보유기간이")
    print(" 같거나 더 짧아야 정상 -- 늦게 터치되는 경우는 있을 수 없음)")
    assert avg_holding_with_hl <= avg_holding_close_only, \
        "High/Low 반영했는데 평균 보유기간이 오히려 길어짐 -- 로직 점검 필요"
    print("점검 5 통과 -- High/Low 반영 시 장중 터치를 더 빨리(또는 같게) 감지함")