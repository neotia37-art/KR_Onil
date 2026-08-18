#!/usr/bin/env python3
"""
CANSLIM-KR Streamlit 대시보드
----------------------------
기존 canslim_analyzer.py 로직을 그대로 사용합니다.

실행:
    streamlit run app.py

배포 (Streamlit Cloud):
    - GitHub에 app.py, canslim_analyzer.py, requirements.txt, watchlist.txt 업로드
    - Streamlit Cloud에서 해당 레포 연결
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st

# 분석기 모듈
try:
    import canslim_analyzer as ca
except ImportError:
    st.error("canslim_analyzer.py 가 같은 폴더에 있어야 합니다.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CANSLIM 한국",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 간단한 다크 톤 CSS
st.markdown("""
<style>
    .stApp { background-color: #101119; color: #EDEBE8; }
    .metric-card {
        background: #191B25; border-radius: 10px; padding: 14px 16px;
        border: 1px solid #2C3042; margin-bottom: 8px;
    }
    .score-high { color: #E5484D; font-weight: 700; }
    .score-mid  { color: #F2C14E; font-weight: 700; }
    .score-low  { color: #4C86F0; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("CANSLIM 한국")
    st.caption("실전 종목 분석 대시보드")

    st.subheader("종목 입력")
    default_codes = ""
    if os.path.exists("watchlist.txt"):
        try:
            codes_from_file = ca.load_watchlist("watchlist.txt")
            default_codes = " ".join(codes_from_file)
        except Exception:
            pass

    codes_text = st.text_area(
        "종목코드 (공백 또는 줄바꿈)",
        value=default_codes,
        height=100,
        help="예: 005930 000660 035420",
    )

    # ★★★ 여기에 키를 넣으세요 (따옴표 필수!)
    DART_API_KEY = "2a36b26b39fff9c0cad83a71452cfb6ae0a6a9d3"

    dart_key = st.text_input(
        "DART API 키 (선택)",
        type="password",
        help="https://opendart.fss.or.kr/ 에서 발급",
        value=DART_API_KEY,
    )

    lookback = st.slider("조회 기간 (일)", 200, 600, 450, 50)

    run_btn = st.button("분석 실행", type="primary", use_container_width=True)

    st.divider()
    st.caption("결과 JSON 다운로드 / 관심종목 저장도 가능합니다.")

# ─────────────────────────────────────────────────────────────
# 세션 상태
# ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "selected" not in st.session_state:
    st.session_state.selected = None


def parse_codes(text: str) -> List[str]:
    import re
    codes = []
    for part in re.split(r"[\s,;]+", text.strip()):
        m = re.search(r"(\d{6})", part)
        if m:
            codes.append(m.group(1))
    return list(dict.fromkeys(codes))


# ─────────────────────────────────────────────────────────────
# 분석 실행
# ─────────────────────────────────────────────────────────────
if run_btn:
    codes = parse_codes(codes_text)
    if not codes:
        st.warning("종목코드를 입력하세요.")
    else:
        with st.spinner(f"{len(codes)}개 종목 분석 중... (수급·실적 조회 포함)"):
            try:
                result = ca.analyze(codes, lookback_days=lookback, dart_key=dart_key or None)
                st.session_state.result = result
                st.session_state.selected = None
                st.success(f"완료 — {len(result.get('종목', []))}개 종목, 점수순 정렬")
            except Exception as e:
                st.error(f"분석 실패: {e}")


result = st.session_state.result


# ─────────────────────────────────────────────────────────────
# 시장 게이트
# ─────────────────────────────────────────────────────────────
if result:
    mkt = result.get("시장판정", {}).get("KOSPI", {})
    state = mkt.get("state", "")
    dist = mkt.get("distribution_days", 0)
    state_label = {
        "CONFIRMED_UPTREND": ("확인된 상승 추세", "#E5484D"),
        "UPTREND_UNDER_PRESSURE": ("상승 추세 · 압박", "#F2C14E"),
        "MARKET_IN_CORRECTION": ("시장 조정", "#4C86F0"),
        "RALLY_ATTEMPT": ("반등 시도", "#F2C14E"),
    }.get(state, (state, "#8A90A2"))

    st.markdown(f"### M · 시장 방향")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("상태", state_label[0])
    c2.metric("분산일", f"{dist}개 / 25일")
    c3.metric("허용 비중", f"{mkt.get('max_exposure', 0)*100:.0f}%")
    c4.metric("기준일", result.get("기준일", "-"))

    for line in mkt.get("commentary", []):
        st.caption(f"· {line}")

    if state == "MARKET_IN_CORRECTION":
        st.warning("시장이 조정 국면입니다. 오닐 규칙상 신규 매수를 하지 않는 것이 원칙입니다.")

    st.divider()

    # ─────────────────────────────────────────────────────────
    # 종목 리스트 (점수순)
    # ─────────────────────────────────────────────────────────
    stocks = result.get("종목", [])
    st.subheader(f"종목 리스트  ({len(stocks)}개 · 점수순)")

    # 필터
    filter_opt = st.radio("필터", ["전체", "매수 가능 구간", "돌파 대기", "A등급 이상"], horizontal=True)

    def pass_filter(s):
        plan = s.get("trade_plan", {})
        if filter_opt == "매수 가능 구간":
            return plan.get("상태") == "매수 가능 구간"
        if filter_opt == "돌파 대기":
            return plan.get("상태") == "돌파 대기"
        if filter_opt == "A등급 이상":
            return s.get("grade", "") in ("A+", "A")
        return True

    filtered = [s for s in stocks if pass_filter(s)]

    if not filtered:
        st.info("조건에 맞는 종목이 없습니다.")
    else:
        for s in filtered:
            score = s.get("total_score", 0)
            grade = s.get("grade", "")
            color_cls = "score-high" if score >= 78 else ("score-mid" if score >= 65 else "score-low")
            plan = s.get("trade_plan", {})
            base = s.get("base", {})

            with st.container():
                cols = st.columns([3, 1, 1, 2, 2])
                with cols[0]:
                    st.markdown(f"**{s.get('name', '')}**  `{s.get('code')}` · {s.get('market')}")
                with cols[1]:
                    st.markdown(f"<span class='{color_cls}'>{score}</span> {grade}", unsafe_allow_html=True)
                with cols[2]:
                    rs = s.get("factors", {}).get("L", {}).get("detail", {}).get("RS Rating (프록시)", "-")
                    st.caption(f"RS {rs}")
                with cols[3]:
                    st.caption(plan.get("상태", "-"))
                with cols[4]:
                    if st.button("상세", key=f"btn_{s['code']}"):
                        st.session_state.selected = s["code"]

                # 간단한 팩터 바
                factors = s.get("factors", {})
                fcols = st.columns(6)
                for i, k in enumerate(["C", "A", "N", "S", "L", "I"]):
                    sc = factors.get(k, {}).get("score", 0)
                    fcols[i].progress(min(sc / 100, 1.0), text=f"{k} {sc}")

                st.markdown("---")

    # ─────────────────────────────────────────────────────────
    # 상세 패널
    # ─────────────────────────────────────────────────────────
    if st.session_state.selected:
        sel = next((x for x in stocks if x["code"] == st.session_state.selected), None)
        if sel:
            st.subheader(f"상세 · {sel['name']} ({sel['code']})")
            if st.button("목록으로"):
                st.session_state.selected = None
                st.rerun()

            p1, p2, p3 = st.columns(3)
            p1.metric("총점", f"{sel['total_score']} ({sel['grade']})")
            p2.metric("현재가", f"{sel['trade_plan'].get('현재가', 0):,}")
            p3.metric("피벗", f"{sel['trade_plan'].get('피벗(매수기준가)', 0):,}")

            st.markdown(f"**판정:** {sel.get('verdict')}")
            st.markdown(f"**매매 상태:** {sel['trade_plan'].get('상태')} — {sel['trade_plan'].get('사유')}")

            # 팩터 상세
            st.markdown("#### 팩터 상세")
            for k in ["C", "A", "N", "S", "L", "I"]:
                f = sel["factors"].get(k, {})
                with st.expander(f"{k} · 점수 {f.get('score', '-')}"):
                    st.json(f.get("detail", {}))
                    for n in f.get("notes", []):
                        st.caption(f"· {n}")

            # 베이스
            st.markdown("#### 베이스")
            st.json(sel.get("base", {}))

            # 경고
            if sel.get("warnings"):
                st.markdown("#### 주의")
                for w in sel["warnings"]:
                    st.warning(w)

    # ─────────────────────────────────────────────────────────
    # 다운로드
    # ─────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "결과 JSON 다운로드",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"canslim_result_{result.get('기준일', '')}.json",
            mime="application/json",
        )
    with col_b:
        # 관심종목 텍스트로 저장용
        codes_only = "\n".join(s["code"] for s in stocks)
        st.download_button(
            "분석된 코드 목록 다운로드",
            data=codes_only,
            file_name="analyzed_codes.txt",
            mime="text/plain",
        )

else:
    st.info("왼쪽에서 종목코드를 입력하고 **분석 실행**을 누르세요.")
    st.markdown("""
    **사용 예시**
    - `005930 000660 035420`
    - 또는 `watchlist.txt`에 넣어두면 자동으로 불러옵니다.

    **DART 키**가 있으면 실적(C/A)이 실제 숫자로 반영됩니다.  
    없어도 기술적 분석 + 네이버 수급(I) + 베이스는 동작합니다.
    """)
