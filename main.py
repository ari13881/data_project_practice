
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

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)


# ============================================================
# 파일 경로 설정
# ============================================================
# main.py가 있는 폴더를 기준으로 경로 설정
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
    """
    sales.csv 폴더 안의 모든 CSV 파일을 읽는다.
    파일마다 인코딩을 개별적으로 확인하고 하나로 합친다.
    """

    # 폴더 안의 모든 CSV 파일 찾기
    csv_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.csv"))
    )

    if not csv_files:
        raise FileNotFoundError(
            "sales.csv 폴더 안에 CSV 파일이 없습니다."
        )

    dataframes = []
    encoding_results = []

    # 시도할 인코딩 목록
    # UTF-8을 먼저 시도하고, 실패하면 CP949와 EUC-KR 시도
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]

    # 각 CSV 파일 개별적으로 읽기
    for file in csv_files:

        df_part = None
        used_encoding = None

        for encoding in encodings:

            try:
                df_part = pd.read_csv(
                    file,
                    encoding=encoding
                )

                used_encoding = encoding
                break

            except UnicodeDecodeError:
                continue

        # 모든 인코딩으로 읽기 실패
        if df_part is None:

            raise ValueError(
                f"파일을 읽을 수 없습니다: "
                f"{os.path.basename(file)}\n"
                f"지원 인코딩: {', '.join(encodings)}"
            )

        # 빈 파일 확인
        if df_part.empty:
            raise ValueError(
                f"데이터가 비어 있는 파일입니다: "
                f"{os.path.basename(file)}"
            )

        dataframes.append(df_part)

        encoding_results.append(
            f"{os.path.basename(file)} : {used_encoding}"
        )

    # 모든 CSV 파일 하나로 합치기
    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df, encoding_results


def quarter_label(code):
    """20262 -> 2026년 2분기"""

    code = int(code)

    return f"{code // 10}년 {code % 10}분기"


def load_record():
    """저장된 JSON 기록(출처)을 불러온다."""

    record = {}

    if os.path.exists(INTRO_JSON):

        try:
            with open(INTRO_JSON, encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                record = loaded

        except (json.JSONDecodeError, OSError):
            record = {}

    if not isinstance(record.get("source"), str):
        record["source"] = ""

    return record


def save_record(record):
    """기록을 JSON 파일로 저장한다."""

    try:
        os.makedirs(
            RECORD_DIR,
            exist_ok=True
        )

        with open(INTRO_JSON, "w", encoding="utf-8") as f:

            json.dump(
                record,
                f,
                ensure_ascii=False,
                indent=2
            )

        return True

    except OSError as e:

        st.error(f"저장하지 못했어요: {e}")

        return False


def build_column_table(df):
    """
    열 이름, 값의 종류, 빈 값 개수 표를 만든다.
    """

    rows = []

    for col in df.columns:

        kind = (
            "숫자"
            if pd.api.types.is_numeric_dtype(df[col])
            else "글자"
        )

        rows.append({
            "열 이름": col,
            "값의 종류": (
                f"{kind} · 서로 다른 값 "
                f"{df[col].nunique():,}종류"
            ),
            "빈 값 개수": int(df[col].isna().sum()),
        })

    return pd.DataFrame(rows)


def big_card(container, label, value):
    """큰 숫자 카드"""

    html = (
        '<div style="padding:18px 12px;border-radius:12px;'
        'background:rgba(128,128,128,0.12);text-align:center;">'
        f'<div style="font-size:0.95rem;opacity:0.75;">'
        f'{label}</div>'
        f'<div style="font-size:2.4rem;font-weight:700;'
        f'line-height:1.3;">{value}</div>'
        '</div>'
    )

    container.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# 1. 데이터 소개 화면
# ============================================================

def page_intro(df):

    # 처음 한 번만 저장된 기록 불러오기
    if "intro_record" not in st.session_state:

        record = load_record()

        st.session_state["intro_record"] = record

        st.session_state["source_text"] = record["source"]

    record = st.session_state["intro_record"]

    st.subheader("1. 데이터 소개")

    st.write(
        "서울시 상권분석서비스의 **자치구별 추정매출** 데이터예요. "
        "한 행은 '한 분기 동안 한 자치구에서 한 업종이 올린 매출'을 뜻하고, "
        "금액과 건수를 요일·시간대·성별·연령대로 나눠 담고 있어요."
    )

    # --------------------------------------------------------
    # 큰 숫자 카드 네 개
    # --------------------------------------------------------

    cols = st.columns(4)

    big_card(
        cols[0],
        "전체 행 수",
        f"{len(df):,}"
    )

    big_card(
        cols[1],
        "열 개수",
        f"{df.shape[1]:,}"
    )

    big_card(
        cols[2],
        "자치구 수",
        f"{df['자치구_코드_명'].nunique():,}"
    )

    big_card(
        cols[3],
        "업종 수",
        f"{df['서비스_업종_코드_명'].nunique():,}"
    )

    # --------------------------------------------------------
    # 데이터 기간
    # --------------------------------------------------------

    first_q = df["기준_년분기_코드"].min()
    last_q = df["기준_년분기_코드"].max()

    st.markdown(
        f"**데이터 기간:** "
        f"{quarter_label(first_q)} ~ "
        f"{quarter_label(last_q)} "
        f"(총 {df['기준_년분기_코드'].nunique()}개 분기)"
    )

    # --------------------------------------------------------
    # 열 설명 표 (열 이름 · 값의 종류 · 빈 값 개수)
    # --------------------------------------------------------

    st.markdown("### 열 설명 표")

    st.dataframe(
        build_column_table(df),
        hide_index=True,
        height=600,
    )

    # --------------------------------------------------------
    # 데이터 처음 다섯 줄
    # --------------------------------------------------------

    st.markdown("### 데이터 처음 다섯 줄")

    st.dataframe(
        df.head(5)
    )

    # --------------------------------------------------------
    # 데이터 출처
    # --------------------------------------------------------

    st.markdown("### 데이터 출처")

    st.text_area(
        "출처를 직접 적어 주세요.",
        key="source_text",
        height=120,
        placeholder=(
            "예: 서울 열린데이터광장, "
            "서울시 상권분석서비스(추정매출-자치구)"
        ),
    )

    if st.button(
        "출처 저장",
        key="save_source"
    ):

        record["source"] = st.session_state["source_text"]

        if save_record(record):

            st.success("출처를 저장했어요.")


# ============================================================
# 앱 실행 부분
# ============================================================

st.title(
    f"{APP_ICON} {APP_TITLE}"
)


# ------------------------------------------------------------
# 데이터 읽기
# ------------------------------------------------------------

try:

    data, encoding_results = load_data()

    st.success(
        f"총 {len(encoding_results)}개의 CSV 파일을 "
        f"성공적으로 불러왔습니다."
    )

    # 파일별 인코딩 확인
    with st.expander("파일별 인코딩 확인하기"):

        for result in encoding_results:

            st.write(result)

    st.caption(
        f"통합 데이터 크기: "
        f"{len(data):,}행 × {data.shape[1]:,}열"
    )


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
        "CSV 파일의 인코딩을 확인할 수 없어요. "
        "원본 파일을 확인해 주세요."
    )

    st.stop()


except ValueError as e:

    st.error(str(e))

    st.stop()


except Exception as e:

    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다: {e}"
    )

    st.stop()


# ============================================================
# 사이드바 메뉴
# ============================================================

PAGES = {
    "1. 데이터 소개": page_intro,
}

st.sidebar.title(
    f"{APP_ICON} 메뉴"
)

menu = st.sidebar.radio(
    "화면 선택",
    list(PAGES.keys())
)

PAGES[menu](data)

