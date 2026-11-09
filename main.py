import json
import os

import pandas as pd
import streamlit as st

# ============================================================
# 기본 설정 (st.set_page_config는 반드시 첫 번째 스트림릿 명령이어야 함)
# ============================================================
APP_TITLE = "매출 성장 업종 예측 실습실"
APP_ICON = "📈"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 파일 경로는 main.py가 있는 폴더를 기준으로 잡는다 (실행 위치와 상관없이 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sales.csv")
RECORD_DIR = os.path.join(BASE_DIR, "내_기록")
INTRO_JSON = os.path.join(RECORD_DIR, "데이터소개.json")


# ============================================================
# 공통 함수
# ============================================================
@st.cache_data
def load_data():
    """sales.csv를 읽어온다. 이 파일은 UTF-8이 아니라 cp949 인코딩이다."""
    return pd.read_csv(DATA_PATH, encoding="cp949")


def quarter_label(code):
    """20262 -> '2026년 2분기' 처럼 읽기 쉬운 글자로 바꾼다 (이후 단계에서도 사용)."""
    code = int(code)
    return f"{code // 10}년 {code % 10}분기"


def load_record():
    """내_기록 폴더의 json을 읽어온다. 파일이 없거나 깨져 있으면 빈 기록을 돌려준다."""
    record = {}
    if os.path.exists(INTRO_JSON):
        try:
            with open(INTRO_JSON, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                record = loaded
        except (json.JSONDecodeError, OSError):
            record = {}
    if not isinstance(record.get("meanings"), dict):
        record["meanings"] = {}
    if not isinstance(record.get("source"), str):
        record["source"] = ""
    return record


def save_record(record):
    """기록을 내_기록 폴더의 json 파일로 저장한다. 성공하면 True."""
    try:
        os.makedirs(RECORD_DIR, exist_ok=True)
        with open(INTRO_JSON, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
        return False


def build_column_table(df, meanings):
    """열 이름 · 우리말 뜻 · 값의 종류 · 빈 값 개수 표를 만든다."""
    rows = []
    for col in df.columns:
        kind = "숫자" if pd.api.types.is_numeric_dtype(df[col]) else "글자"
        rows.append(
            {
                "열 이름": col,
                "우리말 뜻": meanings.get(col, ""),
                "값의 종류": f"{kind} · 서로 다른 값 {df[col].nunique():,}종류",
                "빈 값 개수": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def big_card(container, label, value):
    """큰 숫자 카드 한 개를 그린다 (라이트/다크 모드 모두에서 보이도록 투명 배경 사용)."""
    html = (
        '<div style="padding:18px 12px;border-radius:12px;'
        'background:rgba(128,128,128,0.12);text-align:center;">'
        f'<div style="font-size:0.95rem;opacity:0.75;">{label}</div>'
        f'<div style="font-size:2.4rem;font-weight:700;line-height:1.3;">{value}</div>'
        "</div>"
    )
    container.markdown(html, unsafe_allow_html=True)


# ============================================================
# 1. 데이터 소개 화면
# ============================================================
def page_intro(df):
    # 처음 한 번만 저장된 기록을 불러온다 (새로고침하면 다시 json에서 불러옴)
    if "intro_record" not in st.session_state:
        record = load_record()
        st.session_state["intro_record"] = record
        st.session_state["meaning_base"] = build_column_table(df, record["meanings"])
        st.session_state["source_text"] = record["source"]

    record = st.session_state["intro_record"]

    st.subheader("1. 데이터 소개")
    st.write(
        "서울시 상권분석서비스의 **자치구별 추정매출** 데이터예요. "
        "한 행은 '한 분기 동안 한 자치구에서 한 업종이 올린 매출'을 뜻하고, "
        "금액과 건수를 요일·시간대·성별·연령대로 나눠 담고 있어요."
    )

    # ----- 큰 숫자 카드 네 개 -----
    cols = st.columns(4)
    big_card(cols[0], "전체 행 수", f"{len(df):,}")
    big_card(cols[1], "열 개수", f"{df.shape[1]:,}")
    big_card(cols[2], "자치구 수", f"{df['자치구_코드_명'].nunique():,}")
    big_card(cols[3], "업종 수", f"{df['서비스_업종_코드_명'].nunique():,}")

    # ----- 데이터 기간 -----
    first_q = df["기준_년분기_코드"].min()
    last_q = df["기준_년분기_코드"].max()
    st.markdown(
        f"**데이터 기간:** {quarter_label(first_q)} ~ {quarter_label(last_q)} "
        f"(총 {df['기준_년분기_코드'].nunique()}개 분기)"
    )

    # ----- 열 설명 표 -----
    st.markdown("### 열 설명 표")
    st.caption("'우리말 뜻' 칸은 비어 있어요. 칸을 더블클릭해서 직접 적고, 아래 저장 버튼을 눌러 주세요.")
    edited = st.data_editor(
        st.session_state["meaning_base"],
        key="meaning_editor",
        hide_index=True,
        height=600,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="직접 적어 넣는 칸"),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
    )

    if st.button("우리말 뜻 저장", key="save_meanings"):
        meanings = {}
        for _, row in edited.iterrows():
            text = row["우리말 뜻"]
            if isinstance(text, str) and text.strip():
                meanings[row["열 이름"]] = text.strip()
        record["meanings"] = meanings
        if save_record(record):
            st.success(f"우리말 뜻 {len(meanings)}개를 저장했어요.")

    # ----- 처음 다섯 줄 -----
    st.markdown("### 데이터 처음 다섯 줄")
    st.dataframe(df.head(5))

    # ----- 데이터 출처 -----
    st.markdown("### 데이터 출처")
    st.text_area(
        "출처를 직접 적어 주세요.",
        key="source_text",
        height=120,
        placeholder="예: 서울 열린데이터광장, 서울시 상권분석서비스(추정매출-자치구)",
    )
    if st.button("출처 저장", key="save_source"):
        record["source"] = st.session_state["source_text"]
        if save_record(record):
            st.success("출처를 저장했어요.")


# ============================================================
# 앱 실행 부분
# ============================================================
st.title(f"{APP_ICON} {APP_TITLE}")

# 데이터 읽기 (파일이 없거나 인코딩이 다르면 안내 후 멈춘다)
try:
    data = load_data()
except FileNotFoundError:
    st.error("sales.csv 파일을 찾을 수 없어요. main.py와 같은 폴더(저장소)에 올렸는지 확인해 주세요.")
    st.stop()
except pd.errors.EmptyDataError:
    size = os.path.getsize(DATA_PATH) if os.path.exists(DATA_PATH) else 0
    st.error(
        f"sales.csv 파일은 있지만 내용이 비어 있어요 (현재 크기: {size:,} 바이트). "
        "GitHub 저장소에서 sales.csv를 다시 올려 주세요. 원본은 약 17MB(17,323,614바이트)예요."
    )
    st.stop()
except UnicodeDecodeError:
    st.error("sales.csv를 cp949로 읽지 못했어요. 파일이 바뀌지 않았는지 확인해 주세요.")
    st.stop()

# 사이드바 메뉴 (다음 단계에서 이 딕셔너리에 화면을 계속 추가하면 된다)
PAGES = {
    "1. 데이터 소개": page_intro,
}

st.sidebar.title(f"{APP_ICON} 메뉴")
menu = st.sidebar.radio("화면 선택", list(PAGES.keys()))

PAGES[menu](data)
