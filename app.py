"""AI 복습 퀴즈 & 학습 가이드 생성기 - Streamlit 메인 앱."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import get_available_dates, load_curriculum
from src.embeddings import vectorstore_exists, build_vectorstore, EmbeddingQuotaError

st.set_page_config(
    page_title="AI 복습 퀴즈 & 학습 가이드",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .feature-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def init_vectorstore():
    """벡터스토어 초기화 상태 확인 및 빌드."""
    if "vectorstore_built" not in st.session_state:
        st.session_state.vectorstore_built = vectorstore_exists()


def main():
    init_vectorstore()

    st.markdown('<p class="main-header">AI 복습 퀴즈 & 학습 가이드 생성기</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">강의 내용 기반 맞춤형 퀴즈와 핵심 요약 학습 가이드를 자동 생성합니다</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    if not st.session_state.vectorstore_built:
        st.warning("벡터 데이터베이스가 아직 구축되지 않았습니다. 아래 버튼을 클릭하여 초기 설정을 진행하세요.")
        if st.button("벡터DB 구축하기", type="primary", use_container_width=True):
            with st.spinner("강의 스크립트를 분석하고 벡터DB를 구축하는 중... (최초 1회만 필요합니다)"):
                try:
                    build_vectorstore(force_rebuild=True)
                    st.session_state.vectorstore_built = True
                    st.success("벡터DB 구축이 완료되었습니다!")
                    st.rerun()
                except EmbeddingQuotaError as e:
                    st.error(str(e))
                    st.info(
                        "**해결 방법:** 프로젝트의 `config.yaml`을 열어 "
                        "`embedding_backend` 값을 `\"local\"`로 변경한 뒤, 페이지를 새로고침하고 다시 '벡터DB 구축하기'를 눌러주세요. "
                        "로컬 모델은 API 비용/할당량 없이 동작합니다."
                    )
                except Exception as e:
                    st.error(f"벡터DB 구축 중 오류가 발생했습니다: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        dates = get_available_dates()
        st.metric("강의 일수", f"{len(dates)}일")
    with col2:
        base_dir = Path(__file__).parent
        config_path = base_dir / "config.yaml"
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        csv_path = base_dir / config["paths"]["curriculum_csv"]
        df = load_curriculum(csv_path)
        weeks = df["week"].nunique()
        st.metric("수업 주차", f"{weeks}주")
    with col3:
        subjects = df["subject"].nunique()
        st.metric("과목 수", f"{subjects}개")

    st.divider()

    st.subheader("주요 기능")
    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.markdown("### 퀴즈 풀기")
        st.markdown("""
        - 객관식 / 주관식 / 빈칸 채우기 / 코드
        - 난이도 선택 (쉬움 / 보통 / 어려움)
        - 즉시 정답 확인 및 해설 제공
        """)
        st.page_link("pages/1_퀴즈_풀기.py", label="퀴즈 풀러 가기 →", use_container_width=True)

    with feat_col2:
        st.markdown("### 학습 가이드")
        st.markdown("""
        - 주차별 핵심 요약
        - 핵심 개념 및 정의
        - 개념 관계 맵 시각화
        """)
        st.page_link("pages/2_학습_가이드.py", label="학습 가이드 보기 →", use_container_width=True)

    with feat_col3:
        st.markdown("### 학습 분석")
        st.markdown("""
        - 퀴즈 결과 통계
        - 취약 영역 분석
        - 맞춤형 학습 추천
        """)
        st.page_link("pages/3_학습_분석.py", label="학습 분석 보기 →", use_container_width=True)

    st.divider()

    st.subheader("강의 일정")
    curriculum_df = df[["week", "date", "session", "subject", "content", "learning_goal"]].copy()
    curriculum_df.columns = ["주차", "날짜", "세션", "과목", "내용", "학습 목표"]
    st.dataframe(curriculum_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
