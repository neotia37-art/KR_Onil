#!/usr/bin/env python3
"""
CANSLIM-KR 분산일(Distribution Day) 임계값 백테스트 골격
=======================================================

목적
----
오닐/IBD 원본 임계값(분산일 5개 → 조정 등)이 한국 시장에 최적인지
2015~2025 스타일 데이터로  empirically 검증·최적화하기 위한 골격 코드.

현재 상태
--------
- 실제 KRX 데이터를 받는 부분은 주석으로 남겨 두었음 (데이터 소스에 따라 교체)
- 바로 실행 가능하도록 **합성(synthetic) 데이터 생성기**를 포함
- 합성 데이터는 상승 추세 + 조정 + 거래량 급증 패턴을 재현

사용법
------
    python canslim_backtest.py

결과
----
- 그리드 서치로 최적 임계값 추천
- 성과 지표 (CAGR, MDD, 승률, 노출 비율 등)
- HTML 앱에 바로 넣을 수 있는 DEFAULT_THRESHOLDS 형태의 출력

다음 단계 (실제 데이터)
-----------------------
1. KOSPI 일별 데이터 수집 (OHLCV)
   - pykrx, FinanceDataReader, 또는 KRX API
2. load_real_data() 함수만 교체
3. 유니버스/개별 종목 백테스트로 확장 가능
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# 1. 분산일 계산 엔진 (오닐/IBD 규칙 충실 재현)
# ─────────────────────────────────────────────────────────────

@dataclass
class DistDayParams:
    """분산일 정의 파라미터"""
    price_drop_pct: float = 0.002          # 0.2% 이상 하락
    volume_higher_than_prev: bool = True   # 전일 대비 거래량 증가 필수
    lookback: int = 25                     # 최근 N거래일


def calculate_distribution_days(
    df: pd.DataFrame,
    params: DistDayParams,
) -> pd.Series:
    """
    일별 분산일 여부(0/1)와 롤링 카운트를 계산한다.

    입력 df 필수 컬럼: ['close', 'volume']
    인덱스: DatetimeIndex (거래일)

    반환: rolling distribution day count (lookback 기준)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("인덱스는 DatetimeIndex여야 합니다.")

    close = df["close"]
    volume = df["volume"]

    # 전일 대비 수익률
    ret = close.pct_change()

    # 가격 조건: 0.2% 이상 하락
    price_cond = ret <= -params.price_drop_pct

    # 거래량 조건: 전일보다 많음
    if params.volume_higher_than_prev:
        vol_cond = volume > volume.shift(1)
    else:
        # 대안: 50일 평균 대비 등 (현재는 사용 안 함)
        vol_cond = volume > volume.rolling(50).mean()

    is_dist_day = (price_cond & vol_cond).astype(int)

    # 최근 lookback일 합계
    dist_count = is_dist_day.rolling(window=params.lookback, min_periods=1).sum()

    return dist_count


def derive_market_state(
    dist_count: float,
    pressure_threshold: int = 3,
    correction_threshold: int = 5,
) -> str:
    """분산일 개수로 시장 상태 결정"""
    if dist_count >= correction_threshold:
        return "MARKET_IN_CORRECTION"
    if dist_count >= pressure_threshold:
        return "UPTREND_UNDER_PRESSURE"
    return "CONFIRMED_UPTREND"


# ─────────────────────────────────────────────────────────────
# 2. 백테스트 엔진 (시장 타이밍만 테스트)
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    params: Dict
    cagr: float
    mdd: float
    sharpe: float
    win_rate: float
    time_in_market: float
    final_equity: float
    n_trades: int
    total_days: int


def run_market_timing_backtest(
    df: pd.DataFrame,
    dist_params: DistDayParams,
    pressure_th: int = 3,
    correction_th: int = 5,
    exposure_map: Optional[Dict[str, float]] = None,
) -> BacktestResult:
    """
    분산일 기반 시장 타이밍 백테스트.

    - CONFIRMED_UPTREND      → 노출 100%
    - UPTREND_UNDER_PRESSURE → 노출 50% (또는 exposure_map)
    - MARKET_IN_CORRECTION   → 노출 0%

    단순 buy & hold 대비 성과를 측정한다.
    """
    if exposure_map is None:
        exposure_map = {
            "CONFIRMED_UPTREND": 1.0,
            "UPTREND_UNDER_PRESSURE": 0.5,
            "MARKET_IN_CORRECTION": 0.0,
            "RALLY_ATTEMPT": 0.3,
        }

    dist_count = calculate_distribution_days(df, dist_params)
    states = dist_count.apply(
        lambda x: derive_market_state(x, pressure_th, correction_th)
    )
    exposure = states.map(exposure_map).fillna(0.0)

    # 일별 수익률
    daily_ret = df["close"].pct_change().fillna(0.0)

    # 전략 수익률 = 시장 수익률 × 노출 비중
    strategy_ret = daily_ret * exposure

    # 누적 자산
    equity = (1 + strategy_ret).cumprod()
    final_equity = float(equity.iloc[-1])

    # CAGR
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = final_equity ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0.0

    # MDD
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = float(drawdown.min())

    # 샤프 (간단 버전, 무위험 0 가정, 연율화)
    if strategy_ret.std() > 0:
        sharpe = float(strategy_ret.mean() / strategy_ret.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # 승률 (양의 수익률 일 비율, 노출 > 0인 날만)
    active = exposure > 0
    if active.sum() > 0:
        win_rate = float((strategy_ret[active] > 0).mean())
    else:
        win_rate = 0.0

    time_in_market = float(exposure.mean())

    # 대략적인 포지션 변경 횟수
    n_trades = int((exposure.diff().abs() > 0.1).sum())

    return BacktestResult(
        params={
            "price_drop_pct": dist_params.price_drop_pct,
            "lookback": dist_params.lookback,
            "pressure_th": pressure_th,
            "correction_th": correction_th,
        },
        cagr=cagr,
        mdd=mdd,
        sharpe=sharpe,
        win_rate=win_rate,
        time_in_market=time_in_market,
        final_equity=final_equity,
        n_trades=n_trades,
        total_days=len(df),
    )


# ─────────────────────────────────────────────────────────────
# 3. 합성 데이터 생성기 (바로 실행 가능하도록)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_kospi(
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    한국 시장 특성을 어느 정도 모방한 합성 KOSPI 데이터 생성.

    - 장기 상승 추세
    - 간헐적 조정 (거래량 급증 + 하락)
    - 변동성 클러스터링 느낌
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # 기본 드리프트 + 변동성
    mu = 0.00035          # 대략 연 9% 정도
    sigma = 0.011

    rets = rng.normal(mu, sigma, n)

    # 조정 구간 삽입 (랜덤하게 몇 번)
    n_corrections = 12
    for _ in range(n_corrections):
        start_idx = rng.integers(50, n - 40)
        length = rng.integers(8, 25)
        # 하락 + 변동성 확대
        rets[start_idx:start_idx + length] = rng.normal(-0.004, 0.018, length)

    close = 2000 * np.cumprod(1 + rets)   # 2015년 근처 2000pt 가정

    # 거래량: 기본 + 하락일에 더 커지는 경향
    base_vol = rng.lognormal(mean=15.5, sigma=0.35, size=n)
    vol = base_vol.copy()
    for i in range(1, n):
        if rets[i] < -0.002:
            vol[i] *= rng.uniform(1.15, 1.9)   # 하락일 거래량 증가

    df = pd.DataFrame(
        {"close": close, "volume": vol},
        index=dates,
    )
    df.index.name = "date"
    return df


# ─────────────────────────────────────────────────────────────
# 4. 그리드 서치 최적화
# ─────────────────────────────────────────────────────────────

def grid_search(
    df: pd.DataFrame,
    price_drop_grid: List[float] = None,
    lookback_grid: List[int] = None,
    pressure_grid: List[int] = None,
    correction_grid: List[int] = None,
) -> pd.DataFrame:
    """간단한 그리드 서치. 결과는 샤프 비율 기준으로 정렬."""
    if price_drop_grid is None:
        price_drop_grid = [0.001, 0.002, 0.003]
    if lookback_grid is None:
        lookback_grid = [20, 25, 30]
    if pressure_grid is None:
        pressure_grid = [2, 3, 4]
    if correction_grid is None:
        correction_grid = [4, 5, 6, 7]

    results = []
    combos = list(itertools.product(
        price_drop_grid, lookback_grid, pressure_grid, correction_grid
    ))

    print(f"총 {len(combos)}개 조합 테스트 중...")

    for i, (pdrop, lb, pth, cth) in enumerate(combos):
        if pth >= cth:          # 논리적으로 무의미한 조합 스킵
            continue
        dist_params = DistDayParams(
            price_drop_pct=pdrop,
            lookback=lb,
        )
        res = run_market_timing_backtest(
            df, dist_params, pressure_th=pth, correction_th=cth
        )
        results.append(asdict(res))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)} 완료")

    res_df = pd.DataFrame(results)
    # 파라미터가 dict라서 펼침
    params_df = pd.json_normalize(res_df["params"])
    res_df = pd.concat([params_df, res_df.drop(columns=["params"])], axis=1)

    # 샤프 + MDD를 고려한 간단한 점수 (예시)
    res_df["score"] = res_df["sharpe"] * 0.6 + (-res_df["mdd"]) * 0.4
    res_df = res_df.sort_values("score", ascending=False).reset_index(drop=True)
    return res_df


# ─────────────────────────────────────────────────────────────
# 5. 실제 데이터 로더 골격 (교체 지점)
# ─────────────────────────────────────────────────────────────

def load_real_kospi_data(
    start: str = "2015-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    실제 데이터를 불러오는 함수.
    아래 중 하나를 선택해서 구현하면 됩니다.

    예시 1) FinanceDataReader
        import FinanceDataReader as fdr
        df = fdr.DataReader("KS11", start, end)
        df = df.rename(columns={"Close": "close", "Volume": "volume"})
        return df[["close", "volume"]].dropna()

    예시 2) pykrx
        from pykrx import stock
        df = stock.get_index_ohlcv(start.replace("-", ""), end.replace("-", ""), "1001")
        df = df.rename(columns={"종가": "close", "거래량": "volume"})
        return df[["close", "volume"]]

    현재는 NotImplemented로 남겨 둠.
    """
    raise NotImplementedError(
        "실제 데이터 로더를 구현하세요. "
        "FinanceDataReader 또는 pykrx 예시를 참고."
    )


# ─────────────────────────────────────────────────────────────
# 6. 메인 실행
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CANSLIM-KR 분산일 임계값 백테스트 (골격)")
    print("=" * 60)

    # ── 데이터 로드 ──────────────────────────────────────────
    USE_REAL_DATA = False   # True로 바꾸고 load_real_kospi_data 구현

    if USE_REAL_DATA:
        print("실제 KOSPI 데이터 로딩...")
        df = load_real_kospi_data("2015-01-01", "2025-12-31")
    else:
        print("합성 데이터 생성 중 (2015-2025 스타일)...")
        df = generate_synthetic_kospi("2015-01-01", "2025-12-31", seed=42)
        print(f"  → {len(df)} 거래일 생성 완료")

    print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"시작 종가: {df['close'].iloc[0]:.1f}  →  끝 종가: {df['close'].iloc[-1]:.1f}")
    print()

    # ── 벤치마크: 단순 Buy & Hold ────────────────────────────
    bh_ret = df["close"].pct_change().fillna(0)
    bh_equity = (1 + bh_ret).cumprod()
    bh_cagr = bh_equity.iloc[-1] ** (1 / ((df.index[-1] - df.index[0]).days / 365.25)) - 1
    bh_mdd = ((bh_equity - bh_equity.cummax()) / bh_equity.cummax()).min()
    print(f"[Buy & Hold] CAGR {bh_cagr*100:.2f}%  |  MDD {bh_mdd*100:.2f}%")
    print()

    # ── 오닐 원본 파라미터로 한 번 돌려보기 ──────────────────
    print("오닐 원본 파라미터 (0.2%, lookback=25, pressure=3, correction=5) 테스트...")
    onil_params = DistDayParams(price_drop_pct=0.002, lookback=25)
    onil_res = run_market_timing_backtest(df, onil_params, pressure_th=3, correction_th=5)
    print(f"  CAGR {onil_res.cagr*100:.2f}%  |  MDD {onil_res.mdd*100:.2f}%  |  "
          f"Sharpe {onil_res.sharpe:.2f}  |  시장 노출 {onil_res.time_in_market*100:.1f}%")
    print()

    # ── 그리드 서치 ──────────────────────────────────────────
    print("그리드 서치 시작...")
    results = grid_search(df)

    print("\n" + "=" * 60)
    print("상위 10개 파라미터 조합 (score = 0.6*Sharpe + 0.4*(-MDD))")
    print("=" * 60)
    display_cols = [
        "price_drop_pct", "lookback", "pressure_th", "correction_th",
        "cagr", "mdd", "sharpe", "time_in_market", "score"
    ]
    print(results[display_cols].head(10).to_string(
        float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.2f}"
    ))

    best = results.iloc[0]
    print("\n" + "-" * 60)
    print("추천 임계값 (이번 합성 데이터 기준)")
    print("-" * 60)
    print(f"  price_drop_pct     : {best['price_drop_pct']}")
    print(f"  lookback           : {int(best['lookback'])}")
    print(f"  pressure_th        : {int(best['pressure_th'])}")
    print(f"  correction_th      : {int(best['correction_th'])}")
    print(f"  → CAGR {best['cagr']*100:.2f}%  |  MDD {best['mdd']*100:.2f}%  |  Sharpe {best['sharpe']:.2f}")

    # HTML 앱에 바로 붙여넣을 수 있는 형태
    print("\n" + "=" * 60)
    print("HTML 앱 DEFAULT_THRESHOLDS 에 넣을 값")
    print("=" * 60)
    print(f"""
const DEFAULT_THRESHOLDS = {{
  correctionDays: {int(best['correction_th'])},
  pressureDays: {int(best['pressure_th'])},
  priceDropPct: {best['price_drop_pct']},
  lookbackDays: {int(best['lookback'])},
  volumeHigherThanPrev: true,
}};
""")

    # 결과 저장
    results.to_csv("backtest_results.csv", index=False)
    print("전체 결과 → backtest_results.csv 저장 완료")
    print("\n실제 데이터로 돌리려면 load_real_kospi_data()를 구현하고 USE_REAL_DATA=True로 바꾸세요.")


if __name__ == "__main__":
    main()
