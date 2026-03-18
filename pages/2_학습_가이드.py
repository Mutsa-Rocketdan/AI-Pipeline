"""학습 가이드 페이지: 주차별 요약, 핵심 개념, 복습 포인트, 개념 맵."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.guide_generator import (
    generate_weekly_guide,
    generate_daily_guide,
    get_available_weeks,
    build_concept_map_mermaid,
)
from src.preprocessing import get_available_dates, load_curriculum
from src.embeddings import vectorstore_exists

st.set_page_config(page_title="학습 가이드", layout="wide")


def render_weekly_guide(guide: dict):
    """주차별 학습 가이드 렌더링."""
    st.subheader(f"{guide.get('week', '')}주차 학습 가이드")

    st.markdown("### 주차 요약")
    st.markdown(guide.get("weekly_summary", "요약 정보가 없습니다."))

    daily_summaries = guide.get("daily_summaries", [])
    if daily_summaries:
        st.markdown("### 일별 요약")
        for ds in daily_summaries:
            with st.expander(ds.get("date", "")):
                st.markdown(ds.get("summary", ""))

    key_concepts = guide.get("key_concepts", [])
    if key_concepts:
        st.markdown("### 핵심 개념")
        for concept in key_concepts:
            term = concept.get("term", "")
            definition = concept.get("definition", "")
            st.markdown(f"- **{term}**: {definition}")

    review_points = guide.get("review_points", [])
    if review_points:
        st.markdown("### 복습 포인트")
        for i, point in enumerate(review_points, 1):
            st.markdown(f"{i}. {point}")

    concept_relations = guide.get("concept_relations", [])
    if concept_relations:
        st.markdown("### 개념 관계 맵")
        mermaid_code = build_concept_map_mermaid(concept_relations)
        if mermaid_code:
            st.code(mermaid_code, language="mermaid")

        with st.expander("관계 목록 보기"):
            for rel in concept_relations:
                st.markdown(
                    f"- **{rel.get('from', '')}** → **{rel.get('to', '')}**: "
                    f"{rel.get('relation', '')}"
                )


def render_daily_guide(guide: dict):
    """일별 학습 가이드 렌더링."""
    st.subheader(f"{guide.get('date', '')} 학습 가이드")

    st.markdown("### 강의 요약")
    st.markdown(guide.get("summary", "요약 정보가 없습니다."))

    key_concepts = guide.get("key_concepts", [])
    if key_concepts:
        st.markdown("### 핵심 개념")
        for concept in key_concepts:
            term = concept.get("term", "")
            definition = concept.get("definition", "")
            st.markdown(f"- **{term}**: {definition}")

    review_points = guide.get("review_points", [])
    if review_points:
        st.markdown("### 복습 포인트")
        for i, point in enumerate(review_points, 1):
            st.markdown(f"{i}. {point}")


def main():
    st.title("학습 가이드")
    st.markdown("강의 내용을 기반으로 자동 생성된 학습 가이드입니다.")

    if not vectorstore_exists():
        st.warning("벡터DB가 구축되지 않았습니다. 메인 페이지에서 먼저 구축하세요.")
        return

    st.sidebar.header("가이드 설정")

    view_mode = st.sidebar.radio("보기 모드", ["주차별 가이드", "날짜별 가이드"])

    if view_mode == "주차별 가이드":
        weeks = get_available_weeks()
        if not weeks:
            st.info("사용 가능한 주차 데이터가 없습니다.")
            return

        selected_week = st.sidebar.selectbox(
            "주차 선택",
            options=weeks,
            format_func=lambda x: f"{x}주차",
        )

        if st.sidebar.button("가이드 생성", type="primary", use_container_width=True):
            with st.spinner(f"{selected_week}주차 학습 가이드를 생성하는 중..."):
                try:
                    guide = generate_weekly_guide(selected_week)
                    st.session_state.current_guide = guide
                    st.session_state.guide_mode = "weekly"
                except Exception as e:
                    st.error(f"가이드 생성 중 오류: {e}")
                    return

        if "current_guide" in st.session_state and st.session_state.get("guide_mode") == "weekly":
            render_weekly_guide(st.session_state.current_guide)
        else:
            st.info("왼쪽 사이드바에서 주차를 선택하고 '가이드 생성'을 클릭하세요.")

    else:
        dates = get_available_dates()
        if not dates:
            st.info("사용 가능한 강의 데이터가 없습니다.")
            return

        base_dir = Path(__file__).parent.parent
        import yaml
        with open(base_dir / "config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        csv_path = base_dir / config["paths"]["curriculum_csv"]
        curriculum_df = load_curriculum(csv_path)

        date_labels = {}
        for d in dates:
            rows = curriculum_df[curriculum_df["date"] == d]
            if not rows.empty:
                content = rows.iloc[0].get("content", "")
                date_labels[d] = f"{d} ({content})"
            else:
                date_labels[d] = d

        selected_date = st.sidebar.selectbox(
            "날짜 선택",
            options=dates,
            format_func=lambda x: date_labels.get(x, x),
        )

        if st.sidebar.button("가이드 생성", type="primary", use_container_width=True):
            with st.spinner(f"{selected_date} 학습 가이드를 생성하는 중..."):
                try:
                    guide = generate_daily_guide(selected_date)
                    st.session_state.current_guide = guide
                    st.session_state.guide_mode = "daily"
                except Exception as e:
                    st.error(f"가이드 생성 중 오류: {e}")
                    return

        if "current_guide" in st.session_state and st.session_state.get("guide_mode") == "daily":
            render_daily_guide(st.session_state.current_guide)
        else:
            st.info("왼쪽 사이드바에서 날짜를 선택하고 '가이드 생성'을 클릭하세요.")


if __name__ == "__main__":
    main()
