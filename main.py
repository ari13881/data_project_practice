import json
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
}

st.sidebar.title(
    f"{APP_ICON} 메뉴"
)

menu = st.sidebar.radio(
    "화면 선택",
    list(PAGES.keys())
)

PAGES[menu](data)
