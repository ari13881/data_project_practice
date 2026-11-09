
import json
import os
import glob

import pandas as pd
import streamlit as st

# ============================================================
# 기본 설정
# ============================================================
APP_TITLE = "매출 성장 업종 예측 실습실"
APP_ICON = "📈"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# ============================================================
# 파일 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 여러 CSV 파일이 들어 있는 폴더
DATA_DIR = os.path.join(BASE_DIR, "sales.csv")

# 사용자 기록 저장 폴더
RECORD_DIR = os.path.join(BASE_DIR, "내_기록")
INTRO_JSON = os.path.join(RECORD_DIR, "데이터소개.json")


# ============================================================
# 공통 함수
# ============================================================
@st.cache_data
def load_data():
    """sales.csv 폴더 안의 모든 CSV 파일을 읽어 하나로 합친다."""

    # 폴더 안의 모든 CSV 파일 찾기
    csv_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.csv"))
    )

    if not csv_files:
        raise FileNotFoundError("CSV 파일이 없습니다.")

    # 각각의 CSV 파일 읽기
    dataframes = []

    for file in csv_files:
        df_part = pd.read_csv(file, encoding="cp949")
        dataframes.append(df_part)

    # 모든 파일을 하나의 데이터프레임으로 합치기
    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


def quarter_label(code):
    """20262 -> 2026년 2분기"""
    code = int(code)
    return f"{code // 10}년 {code % 10}분기"


def load_record():
    """저장된 JSON 기록을 불러온다."""

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
    """기록을 JSON 파일로 저장한다."""

    try:
        os.makedirs(RECORD_DIR, exist_ok=True)

        with open(INTRO_JSON, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return True

    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
        return False


def build_column_table(df, meanings):
    """열 이름, 우리말 뜻, 값의 종류, 빈 값 개수 표를 만든다."""

    rows = []

    for col in df.columns:
        kind = "숫자" if pd.api.types.is_numeric_dtype(df[col]) else "글자"

        rows.append({
            "열 이름": col,
            "우리말 뜻": meanings.get(col, ""),
            "값의 종류": f"{kind} · 서로 다른 값 {df[col].nunique():,}종류",
            "빈 값 개수": int(df[col].isna().sum()),
        })

    return pd.DataFrame(rows)


def big_card(container, label, value):
    """큰 숫자 카드"""

    html = (
        '<div style="padding:18px 12px;border-radius:12px;'
        'background:rgba(128,128,128,0.12);text-align:center;">'
        f'<div style="font-size:0.95rem;opacity:0.75;">{label}</div>'
        f'<div style="font-size:2.4rem;font-weight:700;line-height:1.3;">{value}</div>'
        '</div>'
    )

    container.markdown(html, unsafe_allow_html=True)


# ============================================================
# 1. 데이터 소개 화면
# ============================================================
def page_intro(df):

    # 저장된 기록 불러오기
    if "intro_record" not in st.session_state:
        record = load_record()

        st.session_state["intro_record"] = record
        st.session_state["meaning_base"] = build_column_table(
            df, record["meanings"]
        )
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
        f"**데이터 기간:** {quarter_label(first_q)} ~ "
        f"{quarter_label(last_q)} "
        f"(총 {df['기준_년분기_코드'].nunique()}개 분기)"
    )

    # ----- 열 설명 표 -----
    st.markdown("### 열 설명 표")

    st.caption(
        "'우리말 뜻' 칸은 비어 있어요. 칸을 더블클릭해서 직접 적고, "
        "아래 저장 버튼을 눌러 주세요."
    )

    edited = st.data_editor(
        st.session_state["meaning_base"],
        key="meaning_editor",
        hide_index=True,
        height=600,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="직접 적어 넣는 칸"
            ),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn(
                "빈 값 개수",
                format="%d"
            ),
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

# 데이터 읽기
try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "sales.csv 폴더 또는 폴더 안의 CSV 파일을 찾을 수 없어요. "
        "main.py와 같은 위치에 sales.csv 폴더가 있는지 확인해 주세요."
    )
    st.stop()

except pd.errors.EmptyDataError:
    st.error(
        "CSV 파일 중 내용이 비어 있는 파일이 있어요. "
        "GitHub에서 해당 파일을 확인해 주세요."
    )
    st.stop()

except UnicodeDecodeError:
    st.error(
        "CSV 파일을 cp949 인코딩으로 읽지 못했어요. "
        "파일의 인코딩을 확인해 주세요."
    )
    st.stop()

# ============================================================
# 사이드바 메뉴
# ============================================================
PAGES = {
    "1. 데이터 소개": page_intro,
}

st.sidebar.title(f"{APP_ICON} 메뉴")

menu = st.sidebar.radio(
    "화면 선택",
    list(PAGES.keys())
)

PAGES[menu](data)
