import json
import math
import os
import glob
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
EXPLORE_JSON = os.path.join(RECORD_DIR, "데이터탐색.json")
LABEL_JSON = os.path.join(RECORD_DIR, "라벨만들기.json")
INTERPRET_JSON = os.path.join(RECORD_DIR, "결과해석.json")

# 시간 순서 분할 기준 (예측 대상 분기 코드)
TRAIN_LAST_QUARTER = 20252                      # 학습용: 이 분기 이하
VALID_QUARTERS = [20253, 20254, 20261, 20262]   # 검증용: 네 계절이 한 번씩 들어가게 고른 4개 분기

# 숫자가 아니라 '범주'로 다룰 입력 변수 (나머지는 숫자 변수)
CATEGORY_FEATURES = ["서비스_업종_코드_명", "자치구_코드_명", "계절_번호"]

# 모델 학습 설정
RANDOM_STATE = 42
MODEL_NAMES = ["기준선(Dummy)", "로지스틱 회귀", "랜덤포레스트"]
BALANCED_RANGE = (0.3, 0.7)   # 학습용 성장 비율이 이 범위 안이면 '치우치지 않았다'고 보고 class_weight를 쓰지 않음

# 8단계 '경영적 시사점과 한계' 글상자에 처음부터 들어가 있는 글 (이번 분석 결과를 바탕으로 작성)
DEFAULT_INTERPRET_NOTE = """[경영적 시사점]
1. 직전 분기에 매출이 급등한 업종은 다음 분기에 꺾이기 쉽고, 크게 줄었던 업종은 반등하기 쉬운 경향이 있었다. 직전 증감률은 로지스틱 회귀 계수(−1.79)와 랜덤포레스트 중요도(17.9%) 모두에서 1위였다. 따라서 한 분기의 급등만 보고 확장이나 투자를 결정하기보다 몇 분기에 걸친 추세를 함께 보는 것이 안전하다.
2. 계절의 영향이 크다. 1분기는 성장 확률을 낮추고(계수 −0.72) 2분기는 높인다(+0.69). 분기 매출이 줄었다고 곧바로 사업이 나빠졌다고 보기보다 전년 같은 분기와 비교하는 것이 합리적이다.
3. 업종마다 흐름이 다르다. 의약품, 한식음식점, 일반의원 등은 성장 쪽으로, 애완동물, 고시원, 네일숍 등은 하락 쪽으로 작용했다. 업종을 고를 때는 그 업종 자체의 흐름도 함께 고려해야 한다.
4. 자치구의 중요도는 원래 변수 단위로 가장 낮았다(1.8%). 이 분석에서는 '어느 자치구에 있는가'보다 '어떤 업종이고 최근 흐름이 어떠한가'가 성장 여부와 더 관련이 있었다.

[한계]
- 가장 좋은 모델(랜덤포레스트)도 정확도 약 71%, ROC-AUC 0.79로, 참고 자료로 쓸 수준이지 확정적인 예측은 아니다. 실제로 성장한 업종을 놓친 경우가 잘못 성장으로 예측한 경우보다 많았다.
- 평가가 최근 4개 분기 한 번의 결과여서 다른 기간에서도 같은 성능이 나오는지는 확인하지 못했다.
- 외국어학원, 치과의원, 미곡판매 등은 오분류율이 40%대로 찍는 수준에 가까웠다. 업종당 검증 행이 약 100개라 우연 오차(약 5%p)가 있어서 순위 차이는 신중하게 봐야 한다.
- 계수와 변수 중요도는 모델이 근거로 삼은 변수일 뿐 원인을 뜻하지 않는다. 연령대와 시간대 비중은 합이 1인 구성비여서 하나씩 떼어 해석하기 어렵다.
- 직전 증감률에는 극단값이 있었고, 라벨은 늘었는지 여부만 구분해서 증가 폭은 반영하지 못한다. 금액에 물가가 반영되지 않았다면 물가 상승이 성장으로 보일 수 있다.
- 자치구·업종 단위로 모아 놓은 데이터라 개별 점포의 성과나 경기·정책 같은 외부 요인은 설명하지 못한다."""


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


def save_record(record, path=INTRO_JSON):
    """기록을 JSON 파일로 저장한다."""

    try:
        os.makedirs(
            RECORD_DIR,
            exist_ok=True
        )

        with open(path, "w", encoding="utf-8") as f:

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


def load_note(path):
    """json에 저장된 관찰 메모(note)를 글자로 돌려준다. 없거나 깨져 있으면 빈 글자."""

    if os.path.exists(path):

        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict) and isinstance(loaded.get("note"), str):
                return loaded["note"]

        except (json.JSONDecodeError, OSError):
            pass

    return ""


# ============================================================
# 2. 데이터 탐색 화면
# ============================================================

def page_explore(df):

    # 처음 한 번만 저장된 관찰 메모 불러오기
    if "explore_note" not in st.session_state:

        st.session_state["explore_note"] = load_note(EXPLORE_JSON)

    st.subheader("2. 데이터 탐색")

    st.write(
        "매출이 시간에 따라 어떻게 움직이는지, "
        "업종마다 어떻게 다른지 그래프로 살펴봐요. "
        "(금액은 원 단위 합계를 조 원·억 원으로 바꿔서 보여 줘요.)"
    )

    # 분기 코드를 작은 순서대로 정렬하고, 그래프 축에 쓸 글자 목록을 만든다
    quarter_codes = sorted(df["기준_년분기_코드"].unique())
    quarter_labels = [quarter_label(c) for c in quarter_codes]

    # --------------------------------------------------------
    # 그래프 1) 분기별 전체 매출 합계
    # --------------------------------------------------------

    st.markdown("### 1) 분기별 전체 매출 합계")

    total = (
        df.groupby("기준_년분기_코드", as_index=False)["당월_매출_금액"]
        .sum()
    )
    total["분기"] = total["기준_년분기_코드"].map(quarter_label)
    total["매출 합계(조 원)"] = total["당월_매출_금액"] / 1e12

    fig1 = px.line(
        total,
        x="분기",
        y="매출 합계(조 원)",
        markers=True,
    )
    fig1.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=quarter_labels,
        tickangle=-45,
        title_text="분기",
    )
    fig1.update_yaxes(title_text="매출 합계(조 원)")
    fig1.update_traces(
        hovertemplate="%{x}<br>%{y:,.2f}조 원<extra></extra>"
    )

    st.plotly_chart(fig1)

    # --------------------------------------------------------
    # 그래프 2) 업종을 골라서 보는 분기별 매출 합계
    # --------------------------------------------------------

    st.markdown("### 2) 업종별 분기 매출 합계")

    # 전체 기간 매출이 가장 큰 업종을 처음에 선택된 상태로 둔다
    industry_total = (
        df.groupby("서비스_업종_코드_명")["당월_매출_금액"]
        .sum()
        .sort_values(ascending=False)
    )
    industries = sorted(industry_total.index)

    selected = st.selectbox(
        "업종 선택",
        industries,
        index=industries.index(industry_total.index[0]),
        key="explore_industry",
    )

    by_quarter = (
        df[df["서비스_업종_코드_명"] == selected]
        .groupby("기준_년분기_코드", as_index=False)["당월_매출_금액"]
        .sum()
    )
    by_quarter["분기"] = by_quarter["기준_년분기_코드"].map(quarter_label)
    by_quarter["매출 합계(억 원)"] = by_quarter["당월_매출_금액"] / 1e8

    fig2 = px.line(
        by_quarter,
        x="분기",
        y="매출 합계(억 원)",
        markers=True,
        title=f"{selected}의 분기별 매출 합계",
    )
    fig2.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=quarter_labels,
        tickangle=-45,
        title_text="분기",
    )
    fig2.update_yaxes(title_text="매출 합계(억 원)")
    fig2.update_traces(
        hovertemplate="%{x}<br>%{y:,.0f}억 원<extra></extra>"
    )

    st.plotly_chart(fig2)

    # --------------------------------------------------------
    # 그래프 3) 가장 최근 분기의 매출 상위 10개 업종
    # --------------------------------------------------------

    latest = df["기준_년분기_코드"].max()

    st.markdown(f"### 3) {quarter_label(latest)} 매출 상위 10개 업종")

    top10 = (
        df[df["기준_년분기_코드"] == latest]
        .groupby("서비스_업종_코드_명", as_index=False)["당월_매출_금액"]
        .sum()
        .nlargest(10, "당월_매출_금액")
    )
    top10["매출 합계(억 원)"] = top10["당월_매출_금액"] / 1e8

    fig3 = px.bar(
        top10,
        x="매출 합계(억 원)",
        y="서비스_업종_코드_명",
        orientation="h",
    )
    # 매출이 큰 업종이 맨 위에 오도록 정렬
    fig3.update_yaxes(
        categoryorder="total ascending",
        title_text="업종",
    )
    fig3.update_xaxes(title_text="매출 합계(억 원)")
    fig3.update_traces(
        texttemplate="%{x:,.0f}",
        textposition="outside",
        hovertemplate="%{y}<br>%{x:,.0f}억 원<extra></extra>",
    )

    st.plotly_chart(fig3)

    # --------------------------------------------------------
    # 내가 관찰한 점
    # --------------------------------------------------------

    st.markdown("### 내가 관찰한 점")

    st.text_area(
        "그래프를 보고 알게 된 점을 적어 주세요.",
        key="explore_note",
        height=150,
        placeholder="예: 매년 4분기에 매출이 높아지는 모양이 보인다.",
    )

    if st.button(
        "관찰 저장",
        key="save_explore_note"
    ):

        if save_record(
            {"note": st.session_state["explore_note"]},
            EXPLORE_JSON,
        ):

            st.success("관찰한 점을 저장했어요.")


# ============================================================
# 3. 라벨 만들기 (성장여부)
# ============================================================

@st.cache_data
def make_label_data(df):
    """
    '직전 분기 대비 당월_매출_금액이 늘었는가'를 나타내는 성장여부(1/0) 열을 만든다.

    일부 (자치구, 업종) 조합은 중간 분기가 비어 있어서 shift를 쓰면
    '한 줄 위'가 직전 분기가 아닐 수 있다. 그래서 분기 코드를 연속 번호로
    바꾼 뒤, 연속 번호가 정확히 1 작은 행을 merge로 붙여서 직전 분기 매출을 가져온다.

    돌려주는 값: (라벨 붙은 데이터, 통계 딕셔너리)
    """

    keys = ["자치구_코드", "서비스_업종_코드"]

    work = df.copy()

    # 분기 코드를 연속 번호로 바꾸기: 연도 × 4 + 분기 (20262 -> 2026 × 4 + 2)
    work["분기번호"] = (
        (work["기준_년분기_코드"] // 10) * 4
        + (work["기준_년분기_코드"] % 10)
    )

    # (분기, 자치구, 업종)이 겹치는 행이 있으면 merge할 때 행이 늘어나므로 미리 제거
    dup_count = int(work.duplicated(subset=keys + ["분기번호"]).sum())

    if dup_count > 0:
        work = work.drop_duplicates(
            subset=keys + ["분기번호"],
            keep="first"
        )

    total_rows = len(work)

    # 직전 분기 매출 표 만들기: 분기번호에 1을 더해서
    # '다음 분기 행'과 짝이 맞도록 옮겨 둔다
    prev = work[keys + ["분기번호", "당월_매출_금액"]].copy()
    prev["분기번호"] = prev["분기번호"] + 1
    prev = prev.rename(
        columns={"당월_매출_금액": "직전분기_매출_금액"}
    )

    # merge: 같은 자치구·업종이면서 분기번호가 정확히 1 작은 행을 붙인다
    merged = work.merge(
        prev,
        on=keys + ["분기번호"],
        how="left"
    )

    has_prev = merged["직전분기_매출_금액"].notna()

    # 제외할 행: 직전 분기 행이 없는 행
    excluded_total = int((~has_prev).sum())
    first_quarter_no = work["분기번호"].min()
    excluded_first = int(
        ((~has_prev) & (merged["분기번호"] == first_quarter_no)).sum()
    )

    labeled = merged[has_prev].copy()

    # 당월 매출이 직전 분기보다 크면 1(성장), 아니면 0(하락 또는 같음)
    labeled["성장여부"] = (
        labeled["당월_매출_금액"] > labeled["직전분기_매출_금액"]
    ).astype(int)

    labeled = labeled.reset_index(drop=True)

    stats = {
        "total_rows": total_rows,
        "excluded_total": excluded_total,
        "excluded_first": excluded_first,
        "excluded_gap": excluded_total - excluded_first,
        "remaining_rows": len(labeled),
        "dup_removed": dup_count,
    }

    return labeled, stats


def page_label(df):

    # 처음 한 번만 저장된 해석 메모 불러오기
    if "label_note" not in st.session_state:

        st.session_state["label_note"] = load_note(LABEL_JSON)

    st.subheader("3. 라벨 만들기")

    st.write(
        "맞히고 싶은 정답(라벨)은 **'직전 분기보다 매출이 늘었는가'** 예요. "
        "당월 매출이 직전 분기 매출보다 크면 **1(성장)**, "
        "아니면 **0(하락 또는 같음)** 으로 표시해요."
    )

    st.caption(
        "일부 (자치구, 업종)은 중간 분기가 비어 있어서, 바로 윗줄을 직전 분기로 쓰지 않고 "
        "분기 번호(연도 × 4 + 분기)가 정확히 1 작은 행을 찾아서 연결했어요."
    )

    labeled, stats = make_label_data(df)

    if stats["dup_removed"] > 0:

        st.warning(
            f"(분기, 자치구, 업종)이 겹치는 행 {stats['dup_removed']:,}개를 "
            "먼저 제거했어요. 같은 CSV가 두 번 들어 있지 않은지 확인해 보세요."
        )

    # --------------------------------------------------------
    # 제외한 행 수와 남은 행 수
    # --------------------------------------------------------

    st.markdown("### 제외한 행 수와 남은 행 수")

    cols = st.columns(3)

    big_card(cols[0], "전체 행 수", f"{stats['total_rows']:,}")
    big_card(cols[1], "제외한 행 수", f"{stats['excluded_total']:,}")
    big_card(cols[2], "남은 행 수", f"{stats['remaining_rows']:,}")

    st.caption(
        f"제외한 {stats['excluded_total']:,}행 중 "
        f"가장 첫 분기라서 비교할 직전 분기가 없는 행이 {stats['excluded_first']:,}개, "
        f"중간 분기가 비어 있어서 직전 분기를 찾지 못한 행이 {stats['excluded_gap']:,}개예요."
    )

    # --------------------------------------------------------
    # 라벨을 붙인 데이터 처음 다섯 줄
    # --------------------------------------------------------

    st.markdown("### 라벨을 붙인 데이터 처음 다섯 줄")

    st.caption("보기 쉽도록 핵심 열만 보여 줘요. (실제 데이터에는 나머지 열도 모두 들어 있어요.)")

    show_cols = [
        "기준_년분기_코드",
        "자치구_코드_명",
        "서비스_업종_코드_명",
        "직전분기_매출_금액",
        "당월_매출_금액",
        "성장여부",
    ]

    st.dataframe(
        labeled[show_cols].head(5),
        hide_index=True,
    )

    # --------------------------------------------------------
    # 전체 성장 비율
    # --------------------------------------------------------

    st.markdown("### 전체 성장 비율")

    growth_count = int((labeled["성장여부"] == 1).sum())
    fall_count = int((labeled["성장여부"] == 0).sum())
    overall_ratio = labeled["성장여부"].mean()

    cols = st.columns(3)

    big_card(cols[0], "성장(1) 비율", f"{overall_ratio * 100:.1f}%")
    big_card(cols[1], "성장(1) 행 수", f"{growth_count:,}")
    big_card(cols[2], "하락·같음(0) 행 수", f"{fall_count:,}")

    # --------------------------------------------------------
    # 예측 대상 분기별 성장 비율 막대 그래프
    # --------------------------------------------------------

    st.markdown("### 예측 대상 분기별 성장 비율")

    by_quarter = (
        labeled.groupby("기준_년분기_코드", as_index=False)["성장여부"]
        .mean()
    )
    by_quarter["분기"] = by_quarter["기준_년분기_코드"].map(quarter_label)
    by_quarter["성장 비율(%)"] = by_quarter["성장여부"] * 100
    by_quarter["계절"] = (by_quarter["기준_년분기_코드"] % 10).astype(str) + "분기"

    quarter_labels = [
        quarter_label(c)
        for c in sorted(by_quarter["기준_년분기_코드"].unique())
    ]

    fig = px.bar(
        by_quarter,
        x="분기",
        y="성장 비율(%)",
        color="계절",
        category_orders={"계절": ["1분기", "2분기", "3분기", "4분기"]},
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=quarter_labels,
        tickangle=-45,
        title_text="예측 대상 분기",
    )
    fig.update_yaxes(
        range=[0, 100],
        title_text="성장 비율(%)",
    )
    fig.update_traces(
        texttemplate="%{y:.0f}",
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    )
    # 전체 평균선
    fig.add_hline(
        y=overall_ratio * 100,
        line_dash="dash",
        annotation_text=f"전체 평균 {overall_ratio * 100:.1f}%",
    )

    st.plotly_chart(fig)

    st.caption("막대 색은 예측 대상 분기가 1~4분기 중 어느 분기인지를 뜻해요.")

    # --------------------------------------------------------
    # 내 해석
    # --------------------------------------------------------

    st.markdown("### 내 해석")

    st.text_area(
        "그래프를 보고 알게 된 점과 생각을 적어 주세요.",
        key="label_note",
        height=150,
        placeholder="예: 분기에 따라 성장 비율이 크게 달라진다.",
    )

    if st.button(
        "해석 저장",
        key="save_label_note"
    ):

        if save_record(
            {"note": st.session_state["label_note"]},
            LABEL_JSON,
        ):

            st.success("해석을 저장했어요.")


# ============================================================
# 4. 입력 변수 만들기
# ============================================================

def safe_ratio(num, den):
    """num ÷ den. 분모가 0이면 0으로 처리한다."""

    return (num / den.where(den != 0)).fillna(0.0)


def safe_log1p(value):
    """math.log1p. 0 이하의 값은 로그를 구할 수 없으므로 0으로 처리한다."""

    return math.log1p(value) if value > 0 else 0.0


@st.cache_data
def make_feature_data(df):
    """
    성장여부를 예측할 입력 변수를 만든다.

    [데이터 누수 방지 규칙]
    입력 변수는 모두 '직전 분기'(예측 대상 분기보다 1분기 앞) 또는
    '전전 분기'(2분기 앞) 행에서만 가져온다.
    예측 대상 분기(당월)의 금액·건수는 입력에 절대 넣지 않는다.

    돌려주는 값: 딕셔너리 (data, feature_cols, info, stats)
    """

    # 3단계에서 만든 라벨 데이터 (예측 대상 분기 한 줄이 한 행)
    labeled, label_stats = make_label_data(df)

    keys = ["자치구_코드", "서비스_업종_코드"]

    # 3단계와 똑같은 방식으로 분기 번호를 만들고 겹치는 행을 제거
    work = df.copy()
    work["분기번호"] = (
        (work["기준_년분기_코드"] // 10) * 4
        + (work["기준_년분기_코드"] % 10)
    )
    work = work.drop_duplicates(
        subset=keys + ["분기번호"],
        keep="first"
    )

    # 입력 변수를 만드는 데 필요한 열 찾기
    age_cols = [
        c for c in df.columns
        if c.startswith("연령대_") and c.endswith("_매출_금액")
    ]
    time_cols = [
        c for c in df.columns
        if c.startswith("시간대_") and c.endswith("_매출_금액")
    ]
    need_cols = (
        [
            "당월_매출_금액", "당월_매출_건수",
            "주중_매출_금액", "주말_매출_금액",
            "남성_매출_금액", "여성_매출_금액",
        ]
        + age_cols
        + time_cols
    )

    missing = [c for c in need_cols if c not in df.columns]

    if missing or len(age_cols) != 6 or len(time_cols) != 6:
        raise ValueError(
            "입력 변수를 만드는 데 필요한 열을 찾지 못했어요. "
            f"없는 열: {missing}, 연령대 열 {len(age_cols)}개, 시간대 열 {len(time_cols)}개"
        )

    # 예측 대상 행의 기본 정보 (여기에는 당월 금액을 가져오지 않는다)
    # 직전분기_매출_금액은 3단계에서 구한 값이고, 아래 점검용으로만 쓴다
    base = labeled[
        ["기준_년분기_코드", "분기번호"] + keys
        + ["자치구_코드_명", "서비스_업종_코드_명", "직전분기_매출_금액", "성장여부"]
    ].copy()

    # ----- 직전 분기 행 붙이기 (분기번호가 정확히 1 작은 행) -----
    p1 = work[keys + ["분기번호"] + need_cols].copy()
    p1["직전분기번호"] = p1["분기번호"]          # 원래 분기 번호를 점검용으로 보관
    p1["분기번호"] = p1["분기번호"] + 1          # 다음 분기 행과 짝이 맞도록 옮김
    p1 = p1.rename(columns={c: f"p1_{c}" for c in need_cols})

    merged = base.merge(p1, on=keys + ["분기번호"], how="left")

    # ----- 전전 분기 행 붙이기 (분기번호가 정확히 2 작은 행, 매출만 필요) -----
    p2 = work[keys + ["분기번호", "당월_매출_금액"]].copy()
    p2["전전분기번호"] = p2["분기번호"]
    p2["분기번호"] = p2["분기번호"] + 2
    p2 = p2.rename(columns={"당월_매출_금액": "p2_당월_매출_금액"})

    merged = merged.merge(p2, on=keys + ["분기번호"], how="left")

    # 전전 분기가 없는 행은 제외 (직전 분기 증감률을 구할 수 없음)
    has_p2 = merged["p2_당월_매출_금액"].notna()
    excluded_no_p2 = int((~has_p2).sum())
    data = merged[has_p2].reset_index(drop=True)

    # ----- 입력 변수 만들기 -----
    feat = pd.DataFrame(index=data.index)
    info = []

    def add(name, series, description):
        feat[name] = series
        info.append({"변수 이름": name, "설명": description})

    add(
        "직전_매출_로그",
        data["p1_당월_매출_금액"].map(safe_log1p),
        "직전 분기 매출 금액의 로그값 (math.log1p, 0 이하는 0)",
    )
    add(
        "직전_객단가",
        safe_ratio(data["p1_당월_매출_금액"], data["p1_당월_매출_건수"]),
        "직전 분기 매출 금액 ÷ 직전 분기 매출 건수 (거래 1건당 매출)",
    )
    add(
        "직전_주말_매출_비중",
        safe_ratio(
            data["p1_주말_매출_금액"],
            data["p1_주중_매출_금액"] + data["p1_주말_매출_금액"],
        ),
        "직전 분기 주말 매출 ÷ (주중 + 주말 매출)",
    )
    add(
        "직전_여성_매출_비중",
        safe_ratio(
            data["p1_여성_매출_금액"],
            data["p1_남성_매출_금액"] + data["p1_여성_매출_금액"],
        ),
        "직전 분기 여성 매출 ÷ (남성 + 여성 매출)",
    )

    age_total = data[[f"p1_{c}" for c in age_cols]].sum(axis=1)

    for c in age_cols:
        group = c.replace("연령대_", "").replace("_매출_금액", "")
        add(
            f"직전_연령대_{group}_매출_비중",
            safe_ratio(data[f"p1_{c}"], age_total),
            f"직전 분기 연령대 {group} 매출 ÷ 연령대 6개 매출 합계",
        )

    time_total = data[[f"p1_{c}" for c in time_cols]].sum(axis=1)

    for c in time_cols:
        group = c.replace("시간대_", "").replace("_매출_금액", "")
        add(
            f"직전_시간대_{group}_매출_비중",
            safe_ratio(data[f"p1_{c}"], time_total),
            f"직전 분기 시간대 {group}시 매출 ÷ 시간대 6개 매출 합계",
        )

    add(
        "직전_증감률",
        (
            data["p1_당월_매출_금액"]
            / data["p2_당월_매출_금액"].where(data["p2_당월_매출_금액"] != 0)
            - 1
        ).fillna(0.0),
        "직전 분기 매출 ÷ 전전 분기 매출 − 1 (전전 분기 매출이 0이면 0)",
    )
    add(
        "계절_번호",
        data["기준_년분기_코드"] % 10,
        "예측 대상 분기의 분기 번호(1~4). 미리 알 수 있는 값이라 누수가 아님",
    )
    add(
        "서비스_업종_코드_명",
        data["서비스_업종_코드_명"],
        "업종 이름 (글자 그대로, 인코딩은 다음 단계에서)",
    )
    add(
        "자치구_코드_명",
        data["자치구_코드_명"],
        "자치구 이름 (글자 그대로, 인코딩은 다음 단계에서)",
    )

    feature_cols = list(feat.columns)

    out = pd.concat(
        [
            data[["기준_년분기_코드", "분기번호", "자치구_코드", "서비스_업종_코드", "성장여부"]],
            feat,
        ],
        axis=1,
    )

    stats = {
        "label_rows": len(merged),
        "excluded_no_p2": excluded_no_p2,
        "final_rows": len(out),
        # 점검용: 입력 정보를 가져온 행이 정말 1분기 앞, 2분기 앞인지
        "prev_gap_ok": bool(((data["분기번호"] - data["직전분기번호"]) == 1).all()),
        "prev2_gap_ok": bool(((data["분기번호"] - data["전전분기번호"]) == 2).all()),
        # 점검용: 3단계의 직전 분기 매출과 4단계가 쓴 직전 분기 매출이 같은지
        "prev_match_ok": bool(
            (data["직전분기_매출_금액"] == data["p1_당월_매출_금액"]).all()
        ),
    }

    return {
        "data": out,
        "feature_cols": feature_cols,
        "info": pd.DataFrame(info),
        "stats": stats,
    }


def page_features(df):

    st.subheader("4. 입력 변수")

    st.write(
        "성장여부를 맞히는 데 쓸 **입력 변수(힌트)** 를 만들어요."
    )

    st.info(
        "가장 중요한 규칙: 입력 변수는 모두 **직전 분기 또는 그 이전 정보**로만 만들어요. "
        "예측 대상 분기(당월)의 금액·건수가 입력에 들어가면 정답을 미리 알려 주는 "
        "**데이터 누수**가 생겨서 평가 점수가 가짜로 높아져요."
    )

    try:
        result = make_feature_data(df)
    except ValueError as e:
        st.error(str(e))
        return

    out = result["data"]
    feature_cols = result["feature_cols"]
    info = result["info"]
    stats = result["stats"]

    # --------------------------------------------------------
    # 입력 변수 목록 표
    # --------------------------------------------------------

    st.markdown(f"### 입력 변수 목록 ({len(feature_cols)}개)")

    st.dataframe(
        info,
        hide_index=True,
        height=(len(info) + 1) * 35 + 3,
    )

    # --------------------------------------------------------
    # 행 수
    # --------------------------------------------------------

    st.markdown("### 전전 분기가 없어서 제외한 행")

    cols = st.columns(3)

    big_card(cols[0], "3단계에서 남은 행", f"{stats['label_rows']:,}")
    big_card(cols[1], "제외한 행 수", f"{stats['excluded_no_p2']:,}")
    big_card(cols[2], "최종 행 수", f"{stats['final_rows']:,}")

    st.caption(
        "직전 분기 증감률(직전 ÷ 전전 − 1)을 구하려면 전전 분기 매출이 있어야 해서, "
        "전전 분기 행이 없는 행을 제외했어요."
    )

    # --------------------------------------------------------
    # 만든 데이터 처음 다섯 줄
    # --------------------------------------------------------

    st.markdown("### 만든 데이터 처음 다섯 줄")

    st.caption("예측 대상 분기, 정답(성장여부), 입력 변수를 보여 줘요.")

    st.dataframe(
        out[["기준_년분기_코드", "성장여부"] + feature_cols].head(5),
        hide_index=True,
    )

    # --------------------------------------------------------
    # 성장여부(0/1)별 입력 변수 평균 비교
    # --------------------------------------------------------

    st.markdown("### 성장여부(0/1)별 입력 변수 평균 비교")

    numeric_features = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(out[c])
    ]

    mean_table = out.groupby("성장여부")[numeric_features].mean().T
    mean_table.columns = ["하락·같음(0) 평균", "성장(1) 평균"]
    mean_table["차이(성장 − 하락)"] = (
        mean_table["성장(1) 평균"] - mean_table["하락·같음(0) 평균"]
    )
    mean_table = mean_table.round(4).reset_index(names="변수 이름")

    st.dataframe(
        mean_table,
        hide_index=True,
        height=(len(mean_table) + 1) * 35 + 3,
    )

    st.caption(
        "변수마다 단위가 달라서 '차이'의 크기를 변수끼리 바로 비교할 수는 없어요. "
        "두 그룹의 평균이 눈에 띄게 다른 변수가 구분에 도움이 될 수 있다는 힌트로만 봐 주세요."
    )

    # --------------------------------------------------------
    # 누수 점검
    # --------------------------------------------------------

    st.markdown("### 누수 점검")

    # 원본 데이터의 금액·건수 열은 모두 '그 행의 분기(= 예측 대상 분기)' 값이다
    current_cols = [
        c for c in df.columns
        if c.endswith("_금액") or c.endswith("_건수")
    ]

    same_name = [c for c in feature_cols if c in current_cols]
    has_current_word = [c for c in feature_cols if "당월" in c]
    label_in_features = [c for c in feature_cols if c == "성장여부"]

    checks = [
        (
            "입력 변수 중 원본의 금액·건수 열(당월 값)과 이름이 같은 변수가 없다",
            len(same_name) == 0,
            f"{len(same_name)}개",
        ),
        (
            "입력 변수 이름에 '당월'이 들어간 것이 없다",
            len(has_current_word) == 0,
            f"{len(has_current_word)}개",
        ),
        (
            "정답(성장여부)이 입력 변수에 들어 있지 않다",
            len(label_in_features) == 0,
            f"{len(label_in_features)}개",
        ),
        (
            "입력 정보를 가져온 행이 예측 대상 분기보다 정확히 1분기(직전)·2분기(전전) 앞이다",
            stats["prev_gap_ok"] and stats["prev2_gap_ok"],
            "모든 행 확인",
        ),
        (
            "3단계에서 구한 직전 분기 매출과 4단계가 쓴 직전 분기 매출이 모든 행에서 같다",
            stats["prev_match_ok"],
            "모든 행 확인",
        ),
    ]

    check_table = pd.DataFrame(
        {
            "점검 내용": [c[0] for c in checks],
            "결과": ["✅ 통과" if c[1] else "❌ 문제" for c in checks],
            "상세": [c[2] for c in checks],
        }
    )

    st.dataframe(check_table, hide_index=True)

    if all(c[1] for c in checks):

        st.success(
            "누수 점검 통과: 입력 변수 목록에 예측 대상 분기(당월)의 값이 하나도 없어요."
        )

    else:

        st.error(
            "누수 점검에서 문제가 발견됐어요. 위 표의 ❌ 항목을 확인해 주세요."
        )


# ============================================================
# 5. 데이터 나누기와 전처리
# ============================================================

@st.cache_data
def split_train_valid(df):
    """
    4단계 데이터를 시간 순서로 학습용/검증용으로 나눈다. (무작위로 섞지 않는다)
    - 학습용: 예측 대상 분기가 TRAIN_LAST_QUARTER(20252) 이하인 행
    - 검증용: 예측 대상 분기가 VALID_QUARTERS(20253, 20254, 20261, 20262)인 행

    돌려주는 값: 딕셔너리 (X_train, X_valid, y_train, y_valid, 열 목록, 요약 표 등)
    """

    result = make_feature_data(df)
    data = result["data"]
    feature_cols = result["feature_cols"]

    categorical_cols = [c for c in CATEGORY_FEATURES if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    quarter = data["기준_년분기_코드"]

    # 시간 순서 기준으로 행 고르기
    is_train = quarter <= TRAIN_LAST_QUARTER
    is_valid = quarter.isin(VALID_QUARTERS)

    X = data[feature_cols].copy()

    # 계절 번호는 크기의 의미가 없는 범주이므로 글자('1분기' 등)로 바꿔서 범주로 다룬다
    if "계절_번호" in X.columns:
        X["계절_번호"] = X["계절_번호"].astype(str) + "분기"

    X_train = X[is_train].reset_index(drop=True)
    X_valid = X[is_valid].reset_index(drop=True)
    y_train = data.loc[is_train, "성장여부"].reset_index(drop=True)
    y_valid = data.loc[is_valid, "성장여부"].reset_index(drop=True)
    q_train = quarter[is_train].reset_index(drop=True)
    q_valid = quarter[is_valid].reset_index(drop=True)

    # 요약 표 (학습용/검증용 행 수와 성장 비율)
    summary_rows = []

    for name, q, y in [("학습용", q_train, y_train), ("검증용", q_valid, y_valid)]:

        summary_rows.append(
            {
                "구분": name,
                "예측 대상 분기": f"{quarter_label(q.min())} ~ {quarter_label(q.max())}",
                "행 수": len(y),
                "성장(1) 행 수": int(y.sum()),
                "성장 비율(%)": float(y.mean() * 100),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # 검증용 분기별 행 수와 성장 비율
    valid_by_quarter = (
        pd.DataFrame({"분기코드": q_valid, "성장여부": y_valid})
        .groupby("분기코드", as_index=False)["성장여부"]
        .agg(["count", "mean"])
    )
    valid_by_quarter.columns = ["분기코드", "행 수", "성장 비율(%)"]
    valid_by_quarter["성장 비율(%)"] = valid_by_quarter["성장 비율(%)"] * 100
    valid_by_quarter.insert(
        0, "예측 대상 분기", valid_by_quarter["분기코드"].map(quarter_label)
    )
    valid_by_quarter = valid_by_quarter.drop(columns="분기코드")

    # 점검 결과
    checks = {
        "time_order_ok": bool(q_train.max() < q_valid.min()),
        "no_overlap_ok": int((is_train & is_valid).sum()) == 0,
        "all_used_ok": len(X_train) + len(X_valid) == len(data),
        "valid_quarters_ok": sorted(q_valid.unique()) == sorted(VALID_QUARTERS),
    }

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "q_train": q_train,
        "q_valid": q_valid,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "summary": summary,
        "valid_by_quarter": valid_by_quarter,
        "checks": checks,
    }


def build_preprocessor(numeric_cols, categorical_cols):
    """
    전처리기(ColumnTransformer)를 만든다. (아직 fit하지 않은 상태)
    - 숫자 변수: StandardScaler (평균 0, 표준편차 1로 맞춤)
    - 범주 변수(업종, 자치구, 계절 번호): OneHotEncoder
      (검증용에만 있는 새 범주가 나와도 오류 대신 0으로 처리: handle_unknown="ignore")

    6단계에서는 이 함수로 만든 전처리기를 모델과 Pipeline으로 연결해서 쓴다.
    """

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


@st.cache_data
def preprocess_split(df):
    """
    시간 순서로 나눈 데이터에 전처리를 적용한다.

    [데이터 누수 방지 규칙]
    전처리기는 반드시 '학습용 데이터로만' fit한다.
    검증용 데이터에는 transform만 적용하고, 절대 fit하지 않는다.
    (검증용 정보로 평균·표준편차·범주 목록을 정하면 미래 정보가 새어 들어간다.)
    """

    split = split_train_valid(df)

    pre = build_preprocessor(split["numeric_cols"], split["categorical_cols"])

    # ★ 학습용 데이터로만 fit (평균, 표준편차, 범주 목록을 학습용에서만 배운다)
    X_train_t = pre.fit_transform(split["X_train"])

    # ★ 검증용에는 transform만 적용 (fit 금지)
    X_valid_t = pre.transform(split["X_valid"])

    return {
        **split,
        "preprocessor": pre,
        "X_train_t": X_train_t,
        "X_valid_t": X_valid_t,
        "feature_names": list(pre.get_feature_names_out()),
    }


def page_split(df):

    st.subheader("5. 데이터 나누기")

    st.write(
        "모델을 학습시킬 **학습용**과, 학습에 쓰지 않은 데이터로 성능을 재는 **검증용**으로 나눠요. "
        "시간이 있는 데이터라서 무작위로 섞지 않고 **시간 순서**로 나눠요. "
        "과거로 배우고 이후 분기를 맞히는, 실제 예측과 같은 상황을 만들기 위해서예요."
    )

    st.info(
        f"학습용: 예측 대상 분기가 {quarter_label(TRAIN_LAST_QUARTER)} 이하인 행\n\n"
        f"검증용: 예측 대상 분기가 "
        f"{', '.join(quarter_label(q) for q in VALID_QUARTERS)}인 행 "
        "(1~4분기가 한 번씩 들어가게 했어요)"
    )

    try:
        prep = preprocess_split(df)
    except ValueError as e:
        st.error(str(e))
        return

    summary = prep["summary"]

    # --------------------------------------------------------
    # 학습용·검증용 행 수와 성장 비율 (표)
    # --------------------------------------------------------

    st.markdown("### 학습용·검증용 행 수와 성장 비율")

    st.dataframe(
        summary.round({"성장 비율(%)": 1}),
        hide_index=True,
    )

    # --------------------------------------------------------
    # 막대 그래프 두 개
    # --------------------------------------------------------

    cols = st.columns(2)

    fig_count = px.bar(
        summary,
        x="구분",
        y="행 수",
        color="구분",
        title="행 수",
    )
    fig_count.update_traces(
        texttemplate="%{y:,}",
        textposition="outside",
    )
    fig_count.update_layout(showlegend=False)

    fig_ratio = px.bar(
        summary,
        x="구분",
        y="성장 비율(%)",
        color="구분",
        title="성장(1) 비율",
    )
    fig_ratio.update_yaxes(range=[0, 100])
    fig_ratio.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
    )
    fig_ratio.update_layout(showlegend=False)

    with cols[0]:
        st.plotly_chart(fig_count)

    with cols[1]:
        st.plotly_chart(fig_ratio)

    st.markdown("#### 검증용 분기별 행 수와 성장 비율")

    st.dataframe(
        prep["valid_by_quarter"].round({"성장 비율(%)": 1}),
        hide_index=True,
    )

    st.caption(
        "검증용의 성장 비율은 분기마다 크게 달라요(계절 효과). "
        "네 계절이 모두 들어 있어서 한 계절에만 맞춘 모델이 높은 점수를 받기 어려워요."
    )

    # --------------------------------------------------------
    # 시간 순서 점검
    # --------------------------------------------------------

    st.markdown("### 분할 점검")

    scaler = prep["preprocessor"].named_transformers_["num"]
    checks = prep["checks"]

    check_rows = [
        (
            "학습용의 마지막 분기가 검증용의 첫 분기보다 앞선다 (시간 순서)",
            checks["time_order_ok"],
        ),
        (
            "학습용과 검증용에 겹치는 행이 없다",
            checks["no_overlap_ok"],
        ),
        (
            "4단계의 모든 행이 학습용 또는 검증용에 들어간다",
            checks["all_used_ok"],
        ),
        (
            "검증용에 지정한 4개 분기가 모두 들어 있다",
            checks["valid_quarters_ok"],
        ),
        (
            "전처리기가 학습용 행 수만큼으로만 fit되었다 (검증용 행은 fit에 쓰이지 않음)",
            int(scaler.n_samples_seen_) == len(prep["X_train"]),
        ),
    ]

    st.dataframe(
        pd.DataFrame(
            {
                "점검 내용": [c[0] for c in check_rows],
                "결과": ["✅ 통과" if c[1] else "❌ 문제" for c in check_rows],
            }
        ),
        hide_index=True,
    )

    if all(c[1] for c in check_rows):
        st.success("분할 점검 통과: 시간 순서대로 나뉘었고, 전처리기는 학습용으로만 fit됐어요.")
    else:
        st.error("분할 점검에서 문제가 발견됐어요. 위 표의 ❌ 항목을 확인해 주세요.")

    # --------------------------------------------------------
    # 전처리 결과
    # --------------------------------------------------------

    st.markdown("### 전처리 결과")

    st.write(
        "숫자 변수는 **StandardScaler**(평균 0, 표준편차 1로 맞춤), "
        "업종·자치구·계절 번호는 **OneHotEncoder**(범주마다 0/1 열 만들기)로 바꿔요."
    )

    cols = st.columns(4)

    big_card(cols[0], "숫자 변수", f"{len(prep['numeric_cols'])}개")
    big_card(cols[1], "범주 변수", f"{len(prep['categorical_cols'])}개")
    big_card(cols[2], "변환 후 열 수", f"{prep['X_train_t'].shape[1]:,}")
    big_card(
        cols[3],
        "변환 후 모양 (학습/검증)",
        f"{prep['X_train_t'].shape[0]:,} / {prep['X_valid_t'].shape[0]:,}",
    )

    # ----- 스케일링 확인: 학습용은 평균 0·표준편차 1, 검증용은 그렇지 않은 것이 정상 -----
    st.markdown("#### 숫자 변수 스케일링 확인")

    train_scaled = pd.DataFrame(
        scaler.transform(prep["X_train"][prep["numeric_cols"]]),
        columns=prep["numeric_cols"],
    )
    valid_scaled = pd.DataFrame(
        scaler.transform(prep["X_valid"][prep["numeric_cols"]]),
        columns=prep["numeric_cols"],
    )

    scale_table = pd.DataFrame(
        {
            "변수 이름": prep["numeric_cols"],
            "학습용 평균": train_scaled.mean().values,
            "학습용 표준편차": train_scaled.std(ddof=0).values,
            "검증용 평균": valid_scaled.mean().values,
            "검증용 표준편차": valid_scaled.std(ddof=0).values,
        }
    ).round(3)

    st.dataframe(
        scale_table,
        hide_index=True,
        height=(len(scale_table) + 1) * 35 + 3,
    )

    st.caption(
        "학습용은 평균 0, 표준편차 1이 되도록 맞췄어요. "
        "검증용은 학습용의 평균·표준편차를 그대로 적용했기 때문에 0, 1에서 벗어나는 게 정상이에요. "
        "(검증용으로 따로 맞췄다면 검증용도 0, 1이 되었을 거예요.)"
    )

    # ----- 범주 변수: 학습용에 없던 범주 -----
    st.markdown("#### 범주 변수: 검증용에만 있는 범주")

    cat_rows = []

    for c in prep["categorical_cols"]:

        train_set = set(prep["X_train"][c].unique())
        valid_set = set(prep["X_valid"][c].unique())

        cat_rows.append(
            {
                "변수 이름": c,
                "학습용 범주 수": len(train_set),
                "검증용 범주 수": len(valid_set),
                "학습용에 없던 범주 수": len(valid_set - train_set),
            }
        )

    st.dataframe(
        pd.DataFrame(cat_rows),
        hide_index=True,
    )

    st.caption(
        "학습용에 없던 범주가 검증용에 나오면 OneHotEncoder(handle_unknown=\"ignore\")가 "
        "오류 대신 모든 열을 0으로 처리해요."
    )


# ============================================================
# 6. 모델 학습
# ============================================================

def build_models(numeric_cols, categorical_cols, class_weight=None):
    """
    5단계 전처리와 모델을 Pipeline으로 연결한 모델 세 개를 만든다. (아직 학습 전)

    Pipeline을 fit하면 전처리기도 '학습용 데이터로만' fit되고,
    검증용을 예측할 때는 transform만 적용되므로 데이터 누수가 생기지 않는다.
    """

    def make_pipeline(model):
        return Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )

    return {
        "기준선(Dummy)": make_pipeline(
            DummyClassifier(
                strategy="most_frequent",
                random_state=RANDOM_STATE,
            )
        ),
        "로지스틱 회귀": make_pipeline(
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight=class_weight,
            )
        ),
        "랜덤포레스트": make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                class_weight=class_weight,
            )
        ),
    }


def train_all_models(split, class_weight=None):
    """
    모델 세 개를 학습용 데이터로 학습시키고,
    검증용 예측값(클래스와 성장 확률)과 학습 시간을 돌려준다.
    """

    models = build_models(
        split["numeric_cols"],
        split["categorical_cols"],
        class_weight,
    )

    # 검증용 예측값을 모아 둘 표: 예측 대상 분기, 정답, 모델별 예측 클래스와 성장 확률
    predictions = pd.DataFrame(
        {
            "기준_년분기_코드": split["q_valid"].values,
            "정답": split["y_valid"].values,
        }
    )

    time_rows = []

    for name, model in models.items():

        # 학습 (학습용 데이터만 사용) + 걸린 시간 재기
        start = time.perf_counter()
        model.fit(split["X_train"], split["y_train"])
        elapsed = time.perf_counter() - start

        # 검증용 예측 (전처리는 transform만 적용됨)
        predictions[f"{name}_예측"] = model.predict(split["X_valid"])

        # 성장(1) 확률: predict_proba에서 클래스 1에 해당하는 열
        class_index = list(model.classes_).index(1)
        predictions[f"{name}_성장확률"] = model.predict_proba(
            split["X_valid"]
        )[:, class_index]

        time_rows.append(
            {
                "모델": name,
                "학습 완료": "✅ 학습 완료",
                "학습 시간(초)": round(elapsed, 2),
            }
        )

    return models, predictions, pd.DataFrame(time_rows)


def page_train(df):

    st.subheader("6. 모델 학습")

    st.write(
        "5단계 전처리와 모델을 **Pipeline**으로 이어서 모델 세 개를 학습시켜요. "
        "세 모델은 모두 학습용 데이터로만 학습하고, `random_state`는 42로 고정했어요."
    )

    st.markdown(
        "- **기준선(Dummy)**: 항상 학습용에서 더 많았던 쪽으로 찍는 모델이에요. 다른 모델이 이보다 나아야 의미가 있어요.\n"
        "- **로지스틱 회귀**: 입력 변수의 가중합으로 성장 확률을 계산하는 기본 분류 모델이에요.\n"
        "- **랜덤포레스트**: 결정트리 200개의 의견을 모아 정하는 모델이에요."
    )

    try:
        split = split_train_valid(df)
    except ValueError as e:
        st.error(str(e))
        return

    # ----- 성장 비율이 치우쳤는지 확인해서 class_weight 사용 여부 결정 -----
    train_ratio = float(split["y_train"].mean())
    balanced = BALANCED_RANGE[0] <= train_ratio <= BALANCED_RANGE[1]
    class_weight = None if balanced else "balanced"

    if balanced:

        st.info(
            f"학습용 성장 비율이 {train_ratio * 100:.1f}%로 치우치지 않아서 "
            "class_weight는 쓰지 않아요."
        )

    else:

        st.warning(
            f"학습용 성장 비율이 {train_ratio * 100:.1f}%로 한쪽에 치우쳐 있어서 "
            "로지스틱 회귀와 랜덤포레스트에 class_weight='balanced'를 적용해요."
        )

    cols = st.columns(3)

    big_card(cols[0], "학습용 행 수", f"{len(split['X_train']):,}")
    big_card(cols[1], "검증용 행 수", f"{len(split['X_valid']):,}")
    big_card(cols[2], "입력 변수", f"{len(split['numeric_cols']) + len(split['categorical_cols'])}개")

    # ----- 학습 버튼: 눌렀을 때만 학습한다 -----
    st.markdown("### 학습하기")

    if st.button("모델 학습하기", key="train_models"):

        try:

            with st.spinner("모델을 학습하는 중이에요. 랜덤포레스트는 조금 걸릴 수 있어요..."):

                models, predictions, train_times = train_all_models(
                    split,
                    class_weight,
                )

            # 다음 단계(모델 평가)에서 쓸 수 있도록 session_state에 저장
            st.session_state["trained_models"] = models           # 학습된 Pipeline 3개 (딕셔너리)
            st.session_state["valid_predictions"] = predictions   # 검증용 예측 클래스·성장 확률 표
            st.session_state["train_results"] = train_times       # 학습 완료 여부와 학습 시간 표

            st.success("모델 3개의 학습이 끝났어요.")

        except Exception as e:

            st.error(f"학습 중 오류가 발생했어요: {e}")

    # ----- 학습 완료 여부와 학습 시간 표 -----
    st.markdown("### 학습 완료 여부와 학습 시간")

    if "train_results" in st.session_state:

        results = st.session_state["train_results"]

    else:

        results = pd.DataFrame(
            {
                "모델": MODEL_NAMES,
                "학습 완료": "⏳ 아직 학습 전",
                "학습 시간(초)": None,
            }
        )

    st.dataframe(
        results,
        hide_index=True,
    )

    # ----- 저장된 검증용 예측값 미리보기 -----
    if "valid_predictions" in st.session_state:

        predictions = st.session_state["valid_predictions"]

        st.markdown("### 저장된 검증용 예측값 (처음 다섯 줄)")

        st.caption(
            f"검증용 {len(predictions):,}행에 대해 모델별 예측 클래스(0/1)와 성장(1) 확률을 저장했어요. "
            "다음 단계에서 이 값으로 성능을 평가해요."
        )

        st.dataframe(
            predictions.head(5),
            hide_index=True,
        )

    else:

        st.caption("아직 학습하지 않았어요. 위의 '모델 학습하기' 버튼을 눌러 주세요.")


# ============================================================
# 7. 모델 평가
# ============================================================

BASELINE_NAME = "기준선(Dummy)"


def evaluate_models(predictions):
    """
    6단계에서 저장한 검증용 예측값으로 모델별 평가 지표와 ROC 곡선 데이터를 만든다.

    돌려주는 값: (지표 표, ROC 곡선용 표, 모델 이름 목록)
    """

    y_true = predictions["정답"]

    model_names = [
        n for n in MODEL_NAMES
        if f"{n}_예측" in predictions.columns
    ]

    metric_rows = []
    roc_tables = []

    for name in model_names:

        y_pred = predictions[f"{name}_예측"]
        y_prob = predictions[f"{name}_성장확률"]

        auc = roc_auc_score(y_true, y_prob)

        # zero_division=0: 성장(1)을 한 번도 예측하지 않는 모델(Dummy)은 정밀도 등을 0으로 처리
        metric_rows.append(
            {
                "모델": name,
                "정확도": accuracy_score(y_true, y_pred),
                "정밀도": precision_score(y_true, y_pred, zero_division=0),
                "재현율": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "ROC-AUC": auc,
            }
        )

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        roc_tables.append(
            pd.DataFrame(
                {
                    "모델": f"{name} (AUC {auc:.3f})",
                    "거짓 양성 비율(FPR)": fpr,
                    "참 양성 비율(TPR)": tpr,
                }
            )
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(roc_tables, ignore_index=True),
        model_names,
    )


def accuracy_by_quarter(predictions, model_names):
    """검증용 분기별로 모델마다 정확도를 계산한다."""

    rows = []

    for code, part in predictions.groupby("기준_년분기_코드"):

        for name in model_names:

            rows.append(
                {
                    "분기코드": code,
                    "분기": quarter_label(code),
                    "모델": name,
                    "정확도": accuracy_score(part["정답"], part[f"{name}_예측"]),
                }
            )

    return pd.DataFrame(rows)


def summarize_vs_baseline(metrics):
    """기준선(Dummy)보다 F1과 ROC-AUC가 높은 모델이 무엇인지 문장으로 요약한다."""

    base = metrics[metrics["모델"] == BASELINE_NAME].iloc[0]
    others = metrics[metrics["모델"] != BASELINE_NAME]

    lines = []
    beat_both = []

    for _, r in others.iterrows():

        f1_up = r["F1"] > base["F1"]
        auc_up = r["ROC-AUC"] > base["ROC-AUC"]

        detail = (
            f"F1 {r['F1']:.3f} (기준선 {base['F1']:.3f}), "
            f"ROC-AUC {r['ROC-AUC']:.3f} (기준선 {base['ROC-AUC']:.3f})"
        )

        if f1_up and auc_up:
            lines.append(f"**{r['모델']}**: F1과 ROC-AUC가 모두 기준선보다 높아요. → {detail}")
            beat_both.append(r["모델"])
        elif f1_up:
            lines.append(f"**{r['모델']}**: F1만 기준선보다 높아요. → {detail}")
        elif auc_up:
            lines.append(f"**{r['모델']}**: ROC-AUC만 기준선보다 높아요. → {detail}")
        else:
            lines.append(f"**{r['모델']}**: F1과 ROC-AUC 모두 기준선보다 높지 않아요. → {detail}")

    best_f1 = metrics.loc[metrics["F1"].idxmax(), "모델"]
    best_auc = metrics.loc[metrics["ROC-AUC"].idxmax(), "모델"]

    conclusion = (
        f"F1이 가장 높은 모델은 **{best_f1}**, "
        f"ROC-AUC가 가장 높은 모델은 **{best_auc}**예요."
    )

    return lines, conclusion, beat_both


def page_eval(df):

    st.subheader("7. 모델 평가")

    # 6단계에서 학습한 결과가 없으면 안내만 보여 주고 끝낸다
    if "valid_predictions" not in st.session_state:

        st.warning(
            "아직 학습된 모델이 없어요. 먼저 왼쪽 메뉴에서 '6. 모델 학습'으로 가서 "
            "'모델 학습하기' 버튼을 눌러 주세요."
        )

        return

    predictions = st.session_state["valid_predictions"]

    metrics, roc_table, model_names = evaluate_models(predictions)

    st.write(
        f"검증용 {len(predictions):,}행(학습에 쓰지 않은 최근 4개 분기)에 대한 "
        "세 모델의 예측값으로 성능을 평가해요."
    )

    # --------------------------------------------------------
    # 1) 모델별 평가 지표 비교표
    # --------------------------------------------------------

    st.markdown("### 1) 모델별 평가 지표 비교")

    st.dataframe(
        metrics.round(3),
        hide_index=True,
    )

    st.caption(
        "정확도: 전체 중 맞힌 비율 / 정밀도: 성장이라고 예측한 것 중 실제 성장 비율 / "
        "재현율: 실제 성장 중 찾아낸 비율 / F1: 정밀도와 재현율의 조화평균 / "
        "ROC-AUC: 확률로 성장과 하락을 얼마나 잘 가르는지 (0.5는 찍는 수준, 1에 가까울수록 좋음). "
        "기준선은 성장(1)을 한 번도 예측하지 않아서 정밀도·재현율·F1이 0이에요."
    )

    # --------------------------------------------------------
    # 2) 기준선과의 비교 요약 (자동 문장)
    # --------------------------------------------------------

    st.markdown("### 2) 기준선(Dummy)과 비교한 요약")

    if BASELINE_NAME not in model_names:

        st.info("기준선 모델의 예측값이 없어서 비교 요약을 만들 수 없어요.")

    else:

        lines, conclusion, beat_both = summarize_vs_baseline(metrics)

        for line in lines:
            st.markdown(f"- {line}")

        if beat_both:
            st.success(
                f"{', '.join(beat_both)}: 기준선보다 F1과 ROC-AUC가 모두 높아요. " + conclusion
            )
        else:
            st.warning("F1과 ROC-AUC가 모두 기준선보다 높은 모델이 없어요. " + conclusion)

    # --------------------------------------------------------
    # 3) 혼동행렬 (모델 선택)
    # --------------------------------------------------------

    st.markdown("### 3) 혼동행렬")

    selected = st.selectbox(
        "혼동행렬을 볼 모델",
        model_names,
        key="eval_model",
    )

    cm = confusion_matrix(
        predictions["정답"],
        predictions[f"{selected}_예측"],
        labels=[0, 1],
    )

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=["예측 0 (하락·같음)", "예측 1 (성장)"],
        y=["실제 0 (하락·같음)", "실제 1 (성장)"],
        color_continuous_scale="Blues",
        title=f"{selected} 혼동행렬",
    )
    fig_cm.update_layout(coloraxis_showscale=False)
    fig_cm.update_xaxes(title_text="모델의 예측", side="bottom")
    fig_cm.update_yaxes(title_text="실제 값")

    st.plotly_chart(fig_cm)

    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    st.caption(
        f"하락·같음을 맞힌 수 {tn:,} / 하락인데 성장으로 잘못 예측한 수 {fp:,} / "
        f"성장인데 하락으로 잘못 예측한 수 {fn:,} / 성장을 맞힌 수 {tp:,}"
    )

    # --------------------------------------------------------
    # 4) ROC 곡선
    # --------------------------------------------------------

    st.markdown("### 4) ROC 곡선")

    fig_roc = px.line(
        roc_table,
        x="거짓 양성 비율(FPR)",
        y="참 양성 비율(TPR)",
        color="모델",
    )
    # 대각선 기준선 (찍는 수준, AUC 0.5)
    fig_roc.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray"),
    )
    fig_roc.update_xaxes(range=[0, 1])
    fig_roc.update_yaxes(range=[0, 1])

    st.plotly_chart(fig_roc)

    st.caption("점선 대각선은 아무렇게나 찍는 수준(AUC 0.5)이에요. 곡선이 왼쪽 위로 휘어질수록 좋아요.")

    # --------------------------------------------------------
    # 5) 검증용 분기별 정확도 (계절별 안정성)
    # --------------------------------------------------------

    st.markdown("### 5) 검증용 분기별 정확도")

    by_quarter = accuracy_by_quarter(predictions, model_names)

    quarter_order = [
        quarter_label(c) for c in sorted(by_quarter["분기코드"].unique())
    ]

    fig_q = px.bar(
        by_quarter,
        x="분기",
        y="정확도",
        color="모델",
        barmode="group",
        category_orders={"분기": quarter_order, "모델": model_names},
    )
    fig_q.update_yaxes(range=[0, 1])
    fig_q.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
    )

    st.plotly_chart(fig_q)

    # 분기별 정확도 표와 '최고 − 최저' 차이 (작을수록 계절에 덜 흔들림)
    pivot = by_quarter.pivot(index="모델", columns="분기", values="정확도")
    pivot = pivot[quarter_order].loc[model_names]
    pivot["최고 − 최저"] = pivot.max(axis=1) - pivot.min(axis=1)

    st.dataframe(
        pivot.round(3).reset_index(),
        hide_index=True,
    )

    st.caption(
        "'최고 − 최저'가 작을수록 분기(계절)가 바뀌어도 성능이 비슷하다는 뜻이에요. "
        "검증용 분기마다 성장 비율이 다르다는 점도 함께 봐 주세요."
    )


# ============================================================
# 8. 결과 해석
# ============================================================

MIN_GROUP_ROWS = 30   # 업종·자치구별 오분류율을 계산할 때 필요한 최소 검증 행 수


def clean_feature_name(name):
    """전처리기가 만든 열 이름을 읽기 쉽게 바꾼다. (num__직전_증감률 -> 직전_증감률)"""

    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("서비스_업종_코드_명_", "업종: ")
    name = name.replace("자치구_코드_명_", "자치구: ")
    name = name.replace("계절_번호_", "계절: ")

    return name


def original_variable_name(name, categorical_cols):
    """원핫 인코딩으로 쪼개진 열 이름을 원래 변수 이름으로 되돌린다."""

    if name.startswith("num__"):
        return name[len("num__"):]

    raw = name[len("cat__"):] if name.startswith("cat__") else name

    for col in categorical_cols:
        if raw.startswith(col + "_"):
            return col

    return raw


def pick_best_model(metrics):
    """기준선을 뺀 모델 중 F1이 가장 높은 모델을 고른다 (같으면 ROC-AUC가 높은 쪽)."""

    candidates = metrics[metrics["모델"] != BASELINE_NAME]

    if candidates.empty:
        candidates = metrics

    best = candidates.sort_values(["F1", "ROC-AUC"], ascending=False).iloc[0]

    return best["모델"], best


def group_error_table(frame, group_col, min_rows=MIN_GROUP_ROWS):
    """
    그룹(업종 또는 자치구)별로 틀린 정도를 계산한다.
    frame에는 '정답', '예측', 그리고 group_col 열이 있어야 한다.
    검증 행이 min_rows보다 적은 그룹은 제외한다.
    """

    work = frame.copy()
    work["틀림"] = (work["정답"] != work["예측"]).astype(int)
    work["성장 놓침"] = ((work["정답"] == 1) & (work["예측"] == 0)).astype(int)
    work["잘못 성장 예측"] = ((work["정답"] == 0) & (work["예측"] == 1)).astype(int)

    grouped = work.groupby(group_col).agg(
        **{
            "검증 행 수": ("틀림", "size"),
            "틀린 수": ("틀림", "sum"),
            "성장 놓침": ("성장 놓침", "sum"),
            "잘못 성장 예측": ("잘못 성장 예측", "sum"),
            "실제 성장 비율(%)": ("정답", "mean"),
        }
    )

    grouped = grouped[grouped["검증 행 수"] >= min_rows].copy()
    grouped["오분류율(%)"] = grouped["틀린 수"] / grouped["검증 행 수"] * 100
    grouped["실제 성장 비율(%)"] = grouped["실제 성장 비율(%)"] * 100

    return grouped.reset_index()


def page_interpret(df):

    st.subheader("8. 결과 해석")

    # 6단계에서 학습한 결과가 없으면 안내만 보여 주고 끝낸다
    if (
        "trained_models" not in st.session_state
        or "valid_predictions" not in st.session_state
    ):

        st.warning(
            "아직 학습된 모델이 없어요. 먼저 왼쪽 메뉴에서 '6. 모델 학습'으로 가서 "
            "'모델 학습하기' 버튼을 눌러 주세요."
        )

        return

    # 처음 한 번만: 직접 저장해 둔 글이 있으면 그것을, 없으면 미리 써 둔 글을 글상자에 넣는다
    if "interpret_note" not in st.session_state:

        saved_note = load_note(INTERPRET_JSON)

        st.session_state["interpret_note"] = (
            saved_note if saved_note.strip() else DEFAULT_INTERPRET_NOTE
        )

    models = st.session_state["trained_models"]
    predictions = st.session_state["valid_predictions"]

    try:
        split = split_train_valid(df)
    except ValueError as e:
        st.error(str(e))
        return

    # 예측값이 지금의 검증용 데이터와 같은 순서인지 확인 (업종·자치구를 붙여야 해서)
    aligned = (
        len(predictions) == len(split["X_valid"])
        and (predictions["정답"].values == split["y_valid"].values).all()
        and (predictions["기준_년분기_코드"].values == split["q_valid"].values).all()
    )

    if not aligned:

        st.error(
            "저장된 예측값과 현재 검증용 데이터가 맞지 않아요. 6단계에서 다시 학습해 주세요."
        )

        return

    metrics, _, model_names = evaluate_models(predictions)

    st.write(
        "모델이 **무엇을 근거로 예측했는지**, 그리고 **어디에서 특히 틀렸는지**를 살펴봐요."
    )

    # --------------------------------------------------------
    # 1) 로지스틱 회귀 계수
    # --------------------------------------------------------

    st.markdown("### 1) 로지스틱 회귀 계수: 성장에 플러스·마이너스로 작용한 변수")

    if "로지스틱 회귀" not in models:

        st.info("로지스틱 회귀 모델이 없어서 계수를 볼 수 없어요.")

    else:

        lr_pipeline = models["로지스틱 회귀"]
        lr_names = lr_pipeline.named_steps["preprocess"].get_feature_names_out()
        lr_coef = lr_pipeline.named_steps["model"].coef_[0]

        coef_table = pd.DataFrame(
            {
                "변수": [clean_feature_name(n) for n in lr_names],
                "계수": lr_coef,
            }
        )

        top_plus = coef_table.nlargest(10, "계수")
        top_minus = coef_table.nsmallest(10, "계수")

        fig_plus = px.bar(
            top_plus,
            x="계수",
            y="변수",
            orientation="h",
            title="성장에 플러스로 작용한 변수 상위 10개",
        )
        # 계수가 큰 변수가 맨 위에 오도록 정렬
        fig_plus.update_yaxes(categoryorder="total ascending", title_text="")
        fig_plus.update_traces(texttemplate="%{x:.2f}", textposition="outside")

        fig_minus = px.bar(
            top_minus,
            x="계수",
            y="변수",
            orientation="h",
            title="성장에 마이너스로 작용한 변수 상위 10개",
        )
        # 계수가 가장 작은(가장 마이너스인) 변수가 맨 위에 오도록 정렬
        fig_minus.update_yaxes(categoryorder="total descending", title_text="")
        fig_minus.update_traces(texttemplate="%{x:.2f}", textposition="outside")

        st.plotly_chart(fig_plus)
        st.plotly_chart(fig_minus)

        st.caption(
            "계수가 +이면 그 변수가 클수록(또는 해당 범주일수록) 성장(1) 확률이 올라가고, −이면 내려가요. "
            "숫자 변수는 표준화돼 있어서 '1 표준편차 증가'당 효과이고, 업종·자치구·계절은 '해당 범주인 경우'의 효과예요. "
            "연령대·시간대 비중처럼 서로 합이 1인 변수들은 계수가 함께 움직이므로 하나씩 떼어 해석하는 데는 조심해야 해요."
        )

    # --------------------------------------------------------
    # 2) 랜덤포레스트 변수 중요도
    # --------------------------------------------------------

    st.markdown("### 2) 랜덤포레스트 변수 중요도")

    if "랜덤포레스트" not in models:

        st.info("랜덤포레스트 모델이 없어서 변수 중요도를 볼 수 없어요.")

    else:

        rf_pipeline = models["랜덤포레스트"]
        rf_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
        rf_importance = rf_pipeline.named_steps["model"].feature_importances_

        importance_table = pd.DataFrame(
            {
                "열 이름": list(rf_names),
                "변수": [clean_feature_name(n) for n in rf_names],
                "중요도": rf_importance,
            }
        )

        top_importance = importance_table.nlargest(10, "중요도")

        fig_imp = px.bar(
            top_importance,
            x="중요도",
            y="변수",
            orientation="h",
            title="변수 중요도 상위 10개 (원핫 인코딩 열 기준)",
        )
        fig_imp.update_yaxes(categoryorder="total ascending", title_text="")
        fig_imp.update_traces(texttemplate="%{x:.3f}", textposition="outside")

        st.plotly_chart(fig_imp)

        # 원핫으로 쪼개진 열을 원래 변수 단위로 합쳐서 보기
        categorical_cols = split["categorical_cols"]

        importance_table["원래 변수"] = [
            original_variable_name(n, categorical_cols)
            for n in importance_table["열 이름"]
        ]

        by_variable = (
            importance_table.groupby("원래 변수", as_index=False)["중요도"]
            .sum()
            .sort_values("중요도", ascending=False)
        )
        by_variable["중요도 합계(%)"] = by_variable["중요도"] * 100
        by_variable = by_variable[["원래 변수", "중요도 합계(%)"]].round(2)

        st.markdown("#### 원래 변수 단위로 합친 중요도")

        st.dataframe(
            by_variable,
            hide_index=True,
            height=(len(by_variable) + 1) * 35 + 3,
        )

        st.caption(
            "업종·자치구·계절은 범주마다 열이 쪼개져 있어서 한 열씩 보면 중요도가 작아 보여요. "
            "원래 변수 단위로 합치면 변수 전체의 기여도를 비교할 수 있어요. "
            "다만 이 중요도는 트리가 값을 나눌 때 쓴 횟수와 효과를 센 것이라, "
            "값의 종류가 많은 연속형 변수가 크게 나오는 경향이 있어요."
        )

    # --------------------------------------------------------
    # 가장 성능이 좋은 모델 고르기 (3, 4번에서 사용)
    # --------------------------------------------------------

    best_name, best_row = pick_best_model(metrics)

    valid_info = split["X_valid"][["서비스_업종_코드_명", "자치구_코드_명"]].reset_index(drop=True)
    valid_info["정답"] = predictions["정답"].values

    # --------------------------------------------------------
    # 3) 업종별 오분류율
    # --------------------------------------------------------

    st.markdown("### 3) 업종별 오분류율: 가장 많이 틀린 업종 5개")

    st.info(
        f"성능이 가장 좋은 모델은 **{best_name}**이에요 "
        f"(F1 {best_row['F1']:.3f}, ROC-AUC {best_row['ROC-AUC']:.3f}). "
        f"검증 행이 {MIN_GROUP_ROWS}개 미만인 업종은 제외했고, 이 모델의 오분류율 기준으로 정렬했어요."
    )

    compare_models = [n for n in model_names if n != BASELINE_NAME]

    rates = {}

    for name in compare_models:

        frame = valid_info.copy()
        frame["예측"] = predictions[f"{name}_예측"].values
        rates[name] = group_error_table(frame, "서비스_업종_코드_명")

    best_frame = valid_info.copy()
    best_frame["예측"] = predictions[f"{best_name}_예측"].values

    industry_best = group_error_table(best_frame, "서비스_업종_코드_명")

    industry_top5 = industry_best.sort_values(
        ["오분류율(%)", "검증 행 수"], ascending=[False, False]
    ).head(5)

    industry_view = industry_top5[
        ["서비스_업종_코드_명", "검증 행 수", "실제 성장 비율(%)"]
    ].copy()

    for name in compare_models:

        lookup = rates[name].set_index("서비스_업종_코드_명")["오분류율(%)"]
        industry_view[f"{name} 오분류율(%)"] = industry_view["서비스_업종_코드_명"].map(lookup)

    industry_view = industry_view.rename(columns={"서비스_업종_코드_명": "업종"})

    st.dataframe(
        industry_view.round(1),
        hide_index=True,
    )

    overall_error = float((predictions["정답"] != predictions[f"{best_name}_예측"]).mean() * 100)

    st.caption(
        f"{best_name}의 전체 오분류율은 {overall_error:.1f}%예요. "
        "업종마다 검증 행이 100개 안팎이라 오분류율의 우연 오차(표준오차)가 약 5%p예요. "
        "63개 업종 중에서 가장 나쁜 5개를 고른 것이라, 순위 차이가 작으면 우연일 수 있어요."
    )

    # --------------------------------------------------------
    # 4) 가장 성능이 좋은 모델의 약점 (업종·자치구)
    # --------------------------------------------------------

    st.markdown(f"### 4) {best_name}이(가) 특히 약한 업종·자치구")

    district_best = group_error_table(best_frame, "자치구_코드_명")

    district_top5 = district_best.sort_values(
        ["오분류율(%)", "검증 행 수"], ascending=[False, False]
    ).head(5)

    def weak_rows(table, kind, name_col):

        out = table[
            [name_col, "검증 행 수", "틀린 수", "오분류율(%)", "성장 놓침", "잘못 성장 예측", "실제 성장 비율(%)"]
        ].copy()
        out.insert(0, "구분", kind)
        out = out.rename(columns={name_col: "이름"})
        out["평균 대비(%p)"] = out["오분류율(%)"] - overall_error
        out["주된 오류"] = [
            "성장 업종을 놓침" if m > w else "하락인데 성장으로 예측"
            for m, w in zip(out["성장 놓침"], out["잘못 성장 예측"])
        ]

        return out

    weak_table = pd.concat(
        [
            weak_rows(industry_top5, "업종", "서비스_업종_코드_명"),
            weak_rows(district_top5, "자치구", "자치구_코드_명"),
        ],
        ignore_index=True,
    )

    weak_table = weak_table[
        [
            "구분", "이름", "검증 행 수", "틀린 수", "오분류율(%)", "평균 대비(%p)",
            "성장 놓침", "잘못 성장 예측", "주된 오류", "실제 성장 비율(%)",
        ]
    ]

    st.dataframe(
        weak_table.round(1),
        hide_index=True,
    )

    st.caption(
        "성장 놓침: 실제로는 성장했는데 하락으로 예측한 수 / 잘못 성장 예측: 실제로는 하락인데 성장으로 예측한 수. "
        "평균 대비는 모델 전체 오분류율과의 차이예요. 자치구는 검증 행이 약 240개라 우연 오차(표준오차)가 약 3%p로 업종보다 작아요."
    )

    # --------------------------------------------------------
    # 5) 경영적 시사점과 한계
    # --------------------------------------------------------

    st.markdown("### 경영적 시사점과 한계")

    st.caption(
        "이번 분석 결과를 바탕으로 미리 써 둔 글이에요. 자유롭게 고쳐 쓰고 저장할 수 있어요. "
        "(수치는 이번 학습 결과 기준이라, 데이터나 설정을 바꿨다면 확인해서 고쳐 주세요.)"
    )

    st.text_area(
        "경영적 시사점과 한계",
        key="interpret_note",
        height=520,
        label_visibility="collapsed",
    )

    if st.button(
        "시사점과 한계 저장",
        key="save_interpret_note"
    ):

        if save_record(
            {"note": st.session_state["interpret_note"]},
            INTERPRET_JSON,
        ):

            st.success("저장했어요.")


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
    "2. 데이터 탐색": page_explore,
    "3. 라벨 만들기": page_label,
    "4. 입력 변수": page_features,
    "5. 데이터 나누기": page_split,
    "6. 모델 학습": page_train,
    "7. 모델 평가": page_eval,
    "8. 결과 해석": page_interpret,
}

st.sidebar.title(
    f"{APP_ICON} 메뉴"
)

menu = st.sidebar.radio(
    "화면 선택",
    list(PAGES.keys())
)

PAGES[menu](data)
