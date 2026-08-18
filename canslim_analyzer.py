#!/usr/bin/env python3
"""
CANSLIM-KR 실전 종목 분석기 v3
================================

기능
----
1. 실적(C/A) : DART OpenAPI (선택)
2. 기관/외국인 수급(I) : 네이버 금융 스크래핑
3. 베이스 패턴 : 가격+거래량 휴리스틱
4. 관심종목 파일(watchlist.txt) 지원 → 일괄 분석 + 점수순 정렬
5. HTML 앱에 바로 붙여넣을 수 있는 JSON 출력

사용법
------
    # 개별 종목
    python canslim_analyzer.py 005930 000660

    # 관심종목 파일 사용 (한 줄에 코드 하나)
    python canslim_analyzer.py --watchlist watchlist.txt

    # DART 키 + 관심종목
    export DART_API_KEY=키
    python canslim_analyzer.py --watchlist watchlist.txt

    # 결과 점수순 정렬은 기본 동작
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import FinanceDataReader as fdr
except ImportError:
    raise SystemExit("pip install finance-datareader")

try:
    import OpenDartReader
    HAS_DART = True
except ImportError:
    HAS_DART = False

_dart = None
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def init_dart(api_key: Optional[str] = None) -> bool:
    global _dart
    key = api_key or os.environ.get("DART_API_KEY") or os.environ.get("OPENDART_API_KEY")
    if not key or not HAS_DART:
        _dart = None
        return False
    try:
        _dart = OpenDartReader(key)
        return True
    except Exception as e:
        print(f"[DART] 초기화 실패: {e}")
        _dart = None
        return False


# ─────────────────────────────────────────────────────────────
# 기관/외국인 수급 (네이버)
# ─────────────────────────────────────────────────────────────

def fetch_investor_flow(code: str, days: int = 20) -> Dict[str, Any]:
    """
    네이버 금융 외국인/기관 순매매 페이지에서 최근 수급 추출.
    반환: 기관/외국인 순매수 합계, 연속 순매수 일수 등
    """
    url = f"https://finance.naver.com/item/frgn.naver?code={code}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        dfs = pd.read_html(StringIO(r.text))
        # 보통 3번째 테이블
        raw = None
        for d in dfs:
            cols = str(d.columns)
            if "기관" in cols and "외국인" in cols and "순매매량" in cols:
                raw = d
                break
        if raw is None:
            return {"ok": False, "error": "테이블 없음"}

        # MultiIndex 정리
        df = raw.copy()
        df.columns = ["_".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
        # 날짜 있는 행만
        date_col = [c for c in df.columns if "날짜" in c][0]
        df = df.dropna(subset=[date_col])
        df = df[df[date_col].astype(str).str.contains(r"\d{4}", na=False)]

        inst_col = [c for c in df.columns if "기관" in c and "순매매" in c][0]
        frgn_col = [c for c in df.columns if "외국인" in c and "순매매" in c][0]

        def to_num(x):
            try:
                return float(str(x).replace(",", "").replace("▲", "").replace("↓", "").strip())
            except Exception:
                return 0.0

        inst = df[inst_col].map(to_num)
        frgn = df[frgn_col].map(to_num)

        # 최근 N거래일 (페이지에 보통 20일 정도)
        n = min(days, len(inst))
        inst_sum = float(inst.iloc[:n].sum())
        frgn_sum = float(frgn.iloc[:n].sum())

        # 연속 순매수 일수 (외국인 기준)
        streak = 0
        for val in frgn.iloc[:n]:
            if val > 0:
                streak += 1
            else:
                break

        return {
            "ok": True,
            "inst_net_20d": inst_sum,
            "frgn_net_20d": frgn_sum,
            "frgn_streak": streak,
            "days": n,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def score_investor(flow: Dict) -> Tuple[int, Dict, List[str]]:
    """I 점수 계산"""
    notes = []
    detail = {}
    if not flow.get("ok"):
        return 50, {"데이터": "수급 수집 실패"}, ["네이버 수급 페이지 확인 필요"]

    inst = flow["inst_net_20d"]
    frgn = flow["frgn_net_20d"]
    streak = flow["frgn_streak"]

    detail["기관 20일 순매수"] = int(inst)
    detail["외국인 20일 순매수"] = int(frgn)
    detail["외국인 연속 순매수일"] = streak

    score = 50
    # 동시 순매수면 가산
    if inst > 0 and frgn > 0:
        score += 25
        notes.append("기관·외국인 동시 순매수")
    elif frgn > 0:
        score += 12
        notes.append("외국인 순매수")
    elif inst > 0:
        score += 8
        notes.append("기관 순매수")
    else:
        score -= 15
        notes.append("기관·외국인 모두 순매도 우세")

    if streak >= 5:
        score += 10
        notes.append(f"외국인 {streak}일 연속 순매수")
    elif streak >= 3:
        score += 5

    score = int(np.clip(score, 15, 95))
    return score, detail, notes


# ─────────────────────────────────────────────────────────────
# DART 실적 (기존 유지, 간략화)
# ─────────────────────────────────────────────────────────────

def fetch_earnings(stock_code: str, years: int = 3) -> Dict[str, Any]:
    if _dart is None:
        return {}
    result = {"source": "DART", "quarters": [], "annual": []}
    try:
        current_year = datetime.now().year
        for y in range(current_year, current_year - years - 1, -1):
            try:
                fs = _dart.finstate(stock_code, y, reprt_code="11011")
                if fs is None or (hasattr(fs, "empty") and fs.empty):
                    continue
                if "fs_div" in fs.columns:
                    cfs = fs[fs["fs_div"] == "CFS"]
                    if not cfs.empty:
                        fs = cfs
                def pick(df, keywords):
                    for kw in keywords:
                        row = df[df["account_nm"].str.contains(kw, na=False)]
                        if not row.empty:
                            val = row.iloc[0].get("thstrm_amount")
                            try:
                                return float(str(val).replace(",", ""))
                            except Exception:
                                pass
                    return None
                sales = pick(fs, ["매출액", "수익(매출액)", "영업수익"])
                ni = pick(fs, ["당기순이익", "당기순이익(손실)", "연결당기순이익"])
                equity = pick(fs, ["자본총계", "지배기업 소유주지분"])
                if ni is not None or sales is not None:
                    result["annual"].append({"year": y, "sales": sales, "net_income": ni, "equity": equity})
            except Exception:
                continue
        for y in [current_year, current_year - 1]:
            for rc, label in [("11013", "1Q"), ("11012", "반기"), ("11014", "3Q")]:
                try:
                    fs = _dart.finstate(stock_code, y, reprt_code=rc)
                    if fs is None or (hasattr(fs, "empty") and fs.empty):
                        continue
                    if "fs_div" in fs.columns:
                        cfs = fs[fs["fs_div"] == "CFS"]
                        if not cfs.empty:
                            fs = cfs
                    def pick(df, keywords):
                        for kw in keywords:
                            row = df[df["account_nm"].str.contains(kw, na=False)]
                            if not row.empty:
                                val = row.iloc[0].get("thstrm_amount")
                                try:
                                    return float(str(val).replace(",", ""))
                                except Exception:
                                    pass
                        return None
                    ni = pick(fs, ["당기순이익", "당기순이익(손실)"])
                    if ni is not None:
                        result["quarters"].append({"year": y, "period": label, "net_income": ni})
                except Exception:
                    continue
    except Exception as e:
        result["error"] = str(e)
    return result


def score_from_earnings(earn: Dict) -> Tuple[int, int, Dict, List[str]]:
    notes, detail_c, detail_a = [], {}, {}
    score_c, score_a = 55, 55
    annuals = earn.get("annual", [])
    quarters = earn.get("quarters", [])
    if not annuals and not quarters:
        notes.append("DART 실적 없음 — 직접 확인")
        return score_c, score_a, {"C": detail_c, "A": detail_a}, notes

    if len(annuals) >= 2:
        annuals = sorted(annuals, key=lambda x: x["year"], reverse=True)
        latest, prev = annuals[0], annuals[1]
        if latest.get("net_income") and prev.get("net_income") and prev["net_income"] != 0:
            ni_yoy = (latest["net_income"] - prev["net_income"]) / abs(prev["net_income"])
            detail_a["순이익 YoY"] = round(ni_yoy, 3)
            detail_a["기준연도"] = latest["year"]
            if ni_yoy >= 0.25: score_a = 90
            elif ni_yoy >= 0.15: score_a = 78
            elif ni_yoy >= 0.05: score_a = 65
            elif ni_yoy >= 0: score_a = 55
            else:
                score_a = 35
                notes.append(f"연간 순이익 감소 ({ni_yoy*100:.1f}%)")
        if latest.get("sales") and prev.get("sales") and prev["sales"]:
            detail_a["매출 YoY"] = round((latest["sales"] - prev["sales"]) / abs(prev["sales"]), 3)
        if latest.get("net_income") and latest.get("equity") and latest["equity"]:
            roe = latest["net_income"] / latest["equity"]
            detail_a["ROE(추정)"] = round(roe, 3)
            if roe >= 0.17:
                score_a = min(95, score_a + 8)
                notes.append("ROE 17% 이상")
            elif roe < 0.08:
                score_a = max(30, score_a - 10)

    if quarters:
        q = quarters[0]
        detail_c["기준분기"] = f"{q['year']} {q['period']}"
        detail_c["순이익"] = q.get("net_income")
        score_c = 70 if (q.get("net_income") or 0) > 0 else 40
        notes.append("최근 분기 흑자" if score_c >= 70 else "최근 분기 부진 가능성")

    return score_c, score_a, {"C": detail_c, "A": detail_a}, notes


# ─────────────────────────────────────────────────────────────
# 베이스 패턴 (가격+거래량)
# ─────────────────────────────────────────────────────────────

def detect_base_pattern(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) < 80:
        return {"found": False, "pattern": "NONE", "status": "", "notes": ["데이터 부족"]}

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    volume = df["volume"].astype(float)

    win = min(260, len(df))
    h, l, c, v = high.iloc[-win:], low.iloc[-win:], close.iloc[-win:], volume.iloc[-win:]
    left_portion = int(win * 0.55)
    left_high_idx = h.iloc[:left_portion].idxmax()
    left_high = float(h.loc[left_high_idx])

    if len(h.loc[left_high_idx:]) < 15:
        return {"found": False, "pattern": "NONE", "status": "", "notes": ["고점 이후 부족"]}

    cup_low_idx = l.loc[left_high_idx:].idxmin()
    cup_low = float(l.loc[cup_low_idx])
    depth = (left_high - cup_low) / left_high if left_high else 0

    handle_high = float(h.loc[cup_low_idx:].max()) if len(c.loc[cup_low_idx:]) > 5 else left_high
    handle_low = float(l.loc[cup_low_idx:].min()) if len(c.loc[cup_low_idx:]) > 5 else cup_low
    handle_depth = (handle_high - handle_low) / handle_high if handle_high else 0

    try:
        vol_dryup = v.loc[cup_low_idx:].mean() < v.loc[:cup_low_idx].mean() * 0.9
    except Exception:
        vol_dryup = False

    last = float(c.iloc[-1])
    pivot = left_high
    pct_from_pivot = (last / pivot - 1) if pivot else 0
    vol_ma50 = volume.rolling(50).mean().iloc[-1]
    recent_vol_ratio = float(volume.iloc[-5:].mean() / vol_ma50) if vol_ma50 else 1.0
    weeks = round(win / 5.0, 1)
    notes, stage = [], 1

    if depth < 0.08:
        found, pattern, status = False, "NONE", ""
        notes.append("조정 깊이 너무 얕음")
    elif depth > 0.55:
        found, pattern = True, "CUP_HANDLE"
        status = "FAILED" if pct_from_pivot < -0.1 else "BUILDING"
        notes.append(f"깊은 베이스 {depth*100:.0f}% (실패 위험)")
        stage = 2
    else:
        found, pattern = True, "CUP_HANDLE"
        if 0 <= pct_from_pivot <= 0.05 and recent_vol_ratio >= 1.3:
            status = "BREAKOUT"
            notes.append("피벗 돌파 + 거래량 증가")
        elif 0 <= pct_from_pivot <= 0.08:
            status = "BREAKOUT"
            notes.append("피벗 돌파 구간")
        elif -0.12 <= pct_from_pivot < 0:
            status = "HANDLE"
            notes.append("손잡이 + 거래량 감소" if vol_dryup else "손잡이 구간")
        else:
            status = "BUILDING"
            notes.append("베이스 형성 중")

    return {
        "found": found, "pattern": pattern if found else "NONE", "weeks": weeks,
        "depth": round(depth, 3), "pivot": round(pivot), "low": round(cup_low),
        "left_high": round(left_high), "handle_depth": round(handle_depth, 3),
        "stage": stage, "status": status, "notes": notes,
    }


# ─────────────────────────────────────────────────────────────
# 시장 / 종목 공통
# ─────────────────────────────────────────────────────────────

def fetch_kospi(start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader("KS11", start, end)
    col_map = {"Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["close", "volume"]].dropna()


def calc_distribution_days(df: pd.DataFrame, price_drop_pct: float = 0.002, lookback: int = 25):
    ret = df["close"].pct_change()
    is_dd = ((ret <= -price_drop_pct) & (df["volume"] > df["volume"].shift(1))).astype(int)
    count = is_dd.rolling(lookback, min_periods=1).sum()
    current = int(count.iloc[-1]) if len(count) else 0
    recent = is_dd.iloc[-lookback:]
    dates = [d.strftime("%Y-%m-%d") for d in recent[recent == 1].index]
    return current, dates


def derive_state(dist_days: int, pressure: int = 3, correction: int = 5) -> str:
    if dist_days >= correction: return "MARKET_IN_CORRECTION"
    if dist_days >= pressure: return "UPTREND_UNDER_PRESSURE"
    return "CONFIRMED_UPTREND"


def market_analysis(start: str, end: str) -> Dict[str, Any]:
    kdf = fetch_kospi(start, end)
    dist_days, dist_dates = calc_distribution_days(kdf)
    state = derive_state(dist_days)
    ma50 = kdf["close"].rolling(50).mean().iloc[-1]
    ma200 = kdf["close"].rolling(200).mean().iloc[-1]
    last = kdf["close"].iloc[-1]
    high_252 = kdf["close"].iloc[-252:].max() if len(kdf) >= 252 else kdf["close"].max()
    max_exposure = {"CONFIRMED_UPTREND": 1.0, "UPTREND_UNDER_PRESSURE": 0.5, "MARKET_IN_CORRECTION": 0.0}.get(state, 0.5)
    commentary = []
    if dist_days >= 5: commentary.append(f"분산일 {dist_days}개 — 시장 조정. 신규 매수 금지")
    elif dist_days >= 3: commentary.append(f"분산일 {dist_days}개 — 상승 추세·압박. 신규 매수 축소")
    else: commentary.append(f"분산일 {dist_days}개 — 확인된 상승 추세")
    if last > ma50 and last > ma200: commentary.append("50일·200일선 위 유지")
    elif last < ma50: commentary.append("50일선 아래 — 단기 약세")
    return {
        "index_name": "KOSPI", "state": state, "distribution_days": dist_days,
        "distribution_dates": dist_dates[-8:], "ftd_date": None, "ftd_gain": None, "days_since_ftd": None,
        "above_ma50": bool(last > ma50), "above_ma200": bool(last > ma200),
        "ma50_above_ma200": bool(ma50 > ma200), "pct_from_high": round(last / high_252 - 1, 3),
        "max_exposure": max_exposure, "gate_multiplier": 0.85 if state == "UPTREND_UNDER_PRESSURE" else 1.0,
        "commentary": commentary,
    }


def get_stock_name(code: str) -> str:
    try:
        for market in ("KOSPI", "KOSDAQ"):
            listing = fdr.StockListing(market)
            row = listing[listing["Code"] == code]
            if not row.empty:
                return str(row.iloc[0]["Name"])
    except Exception:
        pass
    return code


def fetch_stock(code: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(code, start, end)
    col_map = {"Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "close" not in df.columns:
        raise RuntimeError(f"{code}: close 없음")
    for col, fallback in [("volume", 0), ("high", "close"), ("low", "close")]:
        if col not in df.columns:
            df[col] = df["close"] if fallback == "close" else 0
    return df.dropna(subset=["close"])


def calc_rs_rating(stock_close: pd.Series, kospi_close: pd.Series, window: int = 126) -> float:
    common = stock_close.index.intersection(kospi_close.index)
    if len(common) < window + 5:
        return 50.0
    s = stock_close.reindex(common).ffill()
    k = kospi_close.reindex(common).ffill()
    rel = (s.iloc[-1] / s.iloc[-window] - 1) - (k.iloc[-1] / k.iloc[-window] - 1)
    return float(np.clip(50 + rel * 180, 1, 99))


def analyze_one_stock(code: str, start: str, end: str, kospi_close: pd.Series, market_state: Dict) -> Dict[str, Any]:
    name = get_stock_name(code)
    df = fetch_stock(code, start, end)
    if len(df) < 40:
        raise RuntimeError(f"{code} ({name}): 데이터 부족")

    close, volume = df["close"], df["volume"]
    last = float(close.iloc[-1])
    high_52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    pct_from_high = last / high_52 if high_52 else 0
    rs = calc_rs_rating(close, kospi_close)
    vol_ma50 = volume.rolling(50).mean().iloc[-1]
    vol_ratio = float(volume.iloc[-1] / vol_ma50) if vol_ma50 else 1.0
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50

    # C/A
    earn = fetch_earnings(code)
    score_c, score_a, earn_detail, earn_notes = score_from_earnings(earn)

    # I (수급)
    flow = fetch_investor_flow(code)
    score_i, i_detail, i_notes = score_investor(flow)
    time.sleep(0.4)  # 네이버 예의

    # 기술적
    score_n = int(np.clip(35 + (pct_from_high - 0.65) * 160, 15, 96))
    score_s = int(np.clip(35 + (vol_ratio - 0.8) * 25, 20, 92))
    score_l = int(np.clip(rs, 5, 99))

    total = round(score_c*0.18 + score_a*0.14 + score_n*0.18 + score_s*0.10 + score_l*0.22 + score_i*0.18, 1)
    grade = "A+" if total >= 85 else "A" if total >= 78 else "B+" if total >= 70 else "B" if total >= 60 else "C"

    base = detect_base_pattern(df)
    pivot = base.get("pivot") or round(high_52)
    status = base.get("status") or ""
    if status == "BREAKOUT":
        plan_status, reason = "매수 가능 구간", "피벗 돌파 구간. 거래량 확인 후 진입."
    elif status == "HANDLE":
        plan_status, reason = "돌파 대기", f"피벗({pivot:,.0f}) 근접. 대량 돌파 대기."
    elif status == "BUILDING":
        plan_status, reason = "베이스 형성 중", "베이스 미완성."
    else:
        plan_status, reason = "대기", "명확한 돌파 신호 약함."

    max_exp = market_state.get("max_exposure", 0.5)
    allow_pct = f"{int(max_exp*10)}% (시장상태 반영)" if max_exp < 1 else "10~15%"

    factors = {
        "C": {"key": "C", "score": score_c, "detail": earn_detail.get("C", {"데이터": "DART 미연결"}), "notes": [n for n in earn_notes if "분기" in n or "DART" in n]},
        "A": {"key": "A", "score": score_a, "detail": earn_detail.get("A", {"데이터": "DART 미연결"}), "notes": [n for n in earn_notes if "연간" in n or "ROE" in n or "순이익" in n]},
        "N": {"key": "N", "score": score_n, "detail": {"52주 고점 대비": round(pct_from_high, 3), "above_ma50": bool(last > ma50)}, "notes": ["신고가 근접도"]},
        "S": {"key": "S", "score": score_s, "detail": {"당일 거래량/50일평균": round(vol_ratio, 2)}, "notes": ["거래량 상대 강도"]},
        "L": {"key": "L", "score": score_l, "detail": {"RS Rating (프록시)": round(rs, 1)}, "notes": ["KOSPI 대비 상대강도 추정"]},
        "I": {"key": "I", "score": score_i, "detail": i_detail, "notes": i_notes},
    }

    warnings = []
    if _dart is None:
        warnings.append("DART 키 없음 → C/A 중립. 실적 직접 확인")
    if market_state.get("state") == "MARKET_IN_CORRECTION":
        warnings.append("시장 조정 — 신규 매수 금지 권고")
    if base.get("depth", 0) > 0.4:
        warnings.append("베이스 깊이 과다 — 실패 위험")
    if not flow.get("ok"):
        warnings.append("수급 데이터 수집 실패 — 네이버에서 직접 확인")

    mkt = "KOSPI"
    try:
        if code in fdr.StockListing("KOSDAQ")["Code"].values:
            mkt = "KOSDAQ"
    except Exception:
        pass

    return {
        "code": code, "name": name, "market": mkt, "date": end.replace("-", ""),
        "total_score": total, "grade": grade,
        "verdict": "매수 후보 (최종 확인 후)" if total >= 78 and plan_status == "매수 가능 구간" else "관찰 대상" if total >= 65 else "제외/보류",
        "factors": factors, "base": base,
        "trade_plan": {
            "현재가": round(last), "피벗(매수기준가)": pivot,
            "매수구간": [pivot, round(pivot * 1.05)], "상태": plan_status,
            "손절가": round(pivot * 0.92), "1차 목표": round(pivot * 1.22),
            "손익비": round((1.22 - 1) / 0.08, 2), "허용비중": allow_pct, "사유": reason,
        },
        "market_state": {"state": market_state.get("state"), "distribution_days": market_state.get("distribution_days"), "max_exposure": market_state.get("max_exposure")},
        "warnings": warnings,
    }


def load_watchlist(path: str) -> List[str]:
    codes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 코드만 추출 (이름 있어도 앞 6자리 숫자)
            m = re.search(r"(\d{6})", line)
            if m:
                codes.append(m.group(1))
            elif line.isdigit() and len(line) <= 6:
                codes.append(line.zfill(6))
    return list(dict.fromkeys(codes))  # 중복 제거, 순서 유지


def analyze(codes: List[str], lookback_days: int = 450, dart_key: Optional[str] = None) -> Dict[str, Any]:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    dart_ok = init_dart(dart_key)
    print(f"DART 연동: {'성공' if dart_ok else '키 없음 (C/A 중립)'}")
    print(f"기간: {start} ~ {end}")
    print("시장 분석 중...")
    mkt = market_analysis(start, end)
    print(f"  → {mkt['state']} | 분산일 {mkt['distribution_days']}개")

    kospi_close = fetch_kospi(start, end)["close"]
    stocks = []
    for i, code in enumerate(codes, 1):
        code = code.strip().zfill(6)
        print(f"[{i}/{len(codes)}] {code} ...", end=" ", flush=True)
        try:
            s = analyze_one_stock(code, start, end, kospi_close, mkt)
            stocks.append(s)
            print(f"{s['name']:10s} {s['total_score']:5.1f} ({s['grade']}) {s['trade_plan']['상태']}")
        except Exception as e:
            print(f"실패 — {e}")

    # 점수순 정렬 (높은 순)
    stocks.sort(key=lambda x: x["total_score"], reverse=True)

    return {
        "기준일": end.replace("-", ""),
        "유니버스": len(codes),
        "RS통과": len([s for s in stocks if s["factors"]["L"]["score"] >= 70]),
        "시장판정": {"KOSPI": mkt},
        "종목": stocks,
        "_meta": {
            "note": "C/A=DART(선택), I=네이버 수급, 베이스=가격·거래량 휴리스틱. 실전 전 재확인 필수.",
            "dart_enabled": dart_ok,
            "sorted_by": "total_score desc",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "canslim_analyzer.py v3",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="CANSLIM-KR 분석기 v3")
    parser.add_argument("codes", nargs="*", help="종목코드들")
    parser.add_argument("--watchlist", "-w", help="관심종목 파일 (한 줄에 코드 하나)")
    parser.add_argument("--days", type=int, default=450)
    parser.add_argument("--dart-key", default=None)
    parser.add_argument("--out", default="canslim_result.json")
    args = parser.parse_args()

    codes = list(args.codes or [])
    if args.watchlist:
        if not os.path.exists(args.watchlist):
            raise SystemExit(f"관심종목 파일 없음: {args.watchlist}")
        codes = load_watchlist(args.watchlist) + codes
        print(f"관심종목 파일에서 {len(codes)}개 로드")

    if not codes:
        raise SystemExit("종목코드를 넣거나 --watchlist 파일을 지정하세요.")

    result = analyze(codes, lookback_days=args.days, dart_key=args.dart_key)

    print("\n" + "=" * 60)
    print("점수순 정렬 결과 (상위)")
    print("=" * 60)
    for s in result["종목"][:15]:
        print(f"  {s['total_score']:5.1f} {s['grade']:3s} {s['code']} {s['name']:12s} {s['trade_plan']['상태']}")

    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\nJSON 저장 → {args.out}")
    print("HTML 앱 「JSON 불러오기」에 붙여넣으세요.")


if __name__ == "__main__":
    main()
