import json
import math
import os
import glob

import pandas as pd
import plotly.express as px
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
EXPLORE_JSON = os.path.join(RECORD_DIR, "데이터탐색.json")
LABEL_JSON = os.path.join(RECORD_DIR, "라벨만들기.json")


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
}

st.sidebar.title(
    f"{APP_ICON} 메뉴"
)

menu = st.sidebar.radio(
    "화면 선택",
    list(PAGES.keys())
)

PAGES[menu](data)
