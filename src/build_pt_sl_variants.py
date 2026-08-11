"""
pt_sl(손익비) 여러 후보를 최종 파이프라인(D+1 종가체결 + High/Low 반영)으로 생성.

배경: compute_pbo_v2.py(종목조합 축)는 우리가 실제로 "여러 조합을 뒤져보고 고른"
게 아니라 이론적 근거(국면 분산)로 정한 축이었음. 반면 pt_sl은 이 세션에서
실제로 (1,1) -> (2,1)로 스윕하며 탐색했던 축이라, PBO가 원래 잡아야 하는
"탐색 후 최고를 고르는" 상황에 훨씬 정확히 들어맞음.

pt_sl 후보 -- 손절(sl)은 1로 고정, 익절(pt)만 넓혀가며 탐색했던 이 세션의
실제 흐름을 재현: (1,1), (1.5,1), (2,1)[최종 채택], (2.5,1), (3,1)

사용법 (레포 루트에서):
    python src/build_pt_sl_variants.py

전제:
    feature_engineering_triple_barrier.py, feature_engineering_valuation.py,
    labeling_triple_barrier.py와 같은 폴더(src/)에 있어야 함.
"""

from pathlib import Path

from feature_engineering_triple_barrier import build_triple_barrier_dataset_combined

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VALIDATED_TICKERS = {
    "064350": "064350.KS",
    "052690": "052690.KS",
    "118990": "118990.KQ",
}
NUM_DAYS = 20
PT_SL_CANDIDATES = [(1, 1), (1.5, 1), (2, 1), (2.5, 1), (3, 1)]


def config_label(pt_sl: tuple) -> str:
    return f"pt{pt_sl[0]}sl{pt_sl[1]}_nd{NUM_DAYS}_hl"


if __name__ == "__main__":
    for ticker_krx, ticker in VALIDATED_TICKERS.items():
        for pt_sl in PT_SL_CANDIDATES:
            label = config_label(pt_sl)
            out_path = DATA_DIR / f"{ticker_krx}_features_triple_barrier_{label}_valuation.csv"
            if out_path.exists():
                print(f"[{ticker_krx}, pt_sl={pt_sl}] 이미 존재함 -- 건너뜀: {out_path.name}")
                continue

            print(f"[{ticker_krx}, pt_sl={pt_sl}] 생성 중...")
            dataset = build_triple_barrier_dataset_combined(
                ticker=ticker, ticker_krx=ticker_krx, pt_sl=pt_sl, num_days=NUM_DAYS,
            )
            dataset.to_csv(out_path)
            print(f"  -> 저장 완료: {out_path.name} ({dataset.shape[0]}행)")

    print("\n전체 완료 -- compute_pbo_v3.py로 pt_sl 축 PBO 계산할 것.")