"""학습 분석 페이지: 퀴즈 결과 통계, 취약 영역 분석, 학습 추천."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback import SessionStats

st.set_page_config(page_title="학습 분석", layout="wide")

QUIZ_TYPE_NAMES = {
    "multiple_choice": "객관식",
    "short_answer": "주관식",
    "fill_blank": "빈칸 채우기",
    "code": "코드",
}

DIFFICULTY_NAMES = {
    "easy": "쉬움",
    "medium": "보통",
    "hard": "어려움",
}


def main():
    st.title("학습 분석")
    st.markdown("퀴즈 풀이 결과를 분석하여 취약 영역과 학습 추천을 제공합니다.")

    session: SessionStats = st.session_state.get("quiz_session", SessionStats())

    if session.total == 0:
        st.info("아직 퀴즈를 풀지 않았습니다. '퀴즈 풀기' 페이지에서 퀴즈를 풀어보세요.")
        st.page_link("pages/1_퀴즈_풀기.py", label="퀴즈 풀러 가기 →")
        return

    st.subheader("전체 성적 요약")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 문항 수", f"{session.total}문항")
    with col2:
        st.metric("정답 수", f"{session.correct_count}문항")
    with col3:
        st.metric("정답률", f"{session.accuracy:.1f}%")
    with col4:
        st.metric("점수", f"{session.score}점")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("유형별 정답률")
        type_stats = session.accuracy_by_type()
        if type_stats:
            type_data = []
            for qtype, stats in type_stats.items():
                type_data.append({
                    "유형": QUIZ_TYPE_NAMES.get(qtype, qtype),
                    "총 문항": stats["total"],
                    "정답": stats["correct"],
                    "정답률(%)": round(stats["accuracy"], 1),
                })
            df_type = pd.DataFrame(type_data)
            st.dataframe(df_type, use_container_width=True, hide_index=True)

            chart_df = df_type.set_index("유형")[["정답률(%)"]]
            st.bar_chart(chart_df)
        else:
            st.info("유형별 데이터가 없습니다.")

    with col_right:
        st.subheader("난이도별 정답률")
        diff_stats = session.accuracy_by_difficulty()
        if diff_stats:
            diff_data = []
            for diff, stats in diff_stats.items():
                diff_data.append({
                    "난이도": DIFFICULTY_NAMES.get(diff, diff),
                    "총 문항": stats["total"],
                    "정답": stats["correct"],
                    "정답률(%)": round(stats["accuracy"], 1),
                })
            df_diff = pd.DataFrame(diff_data)
            st.dataframe(df_diff, use_container_width=True, hide_index=True)

            chart_df = df_diff.set_index("난이도")[["정답률(%)"]]
            st.bar_chart(chart_df)
        else:
            st.info("난이도별 데이터가 없습니다.")

    st.divider()

    st.subheader("강의별 정답률")
    date_stats = session.accuracy_by_date()
    if date_stats:
        date_data = []
        for date_key, stats in sorted(date_stats.items()):
            date_data.append({
                "날짜": date_key,
                "총 문항": stats["total"],
                "정답": stats["correct"],
                "정답률(%)": round(stats["accuracy"], 1),
            })
        df_date = pd.DataFrame(date_data)
        st.dataframe(df_date, use_container_width=True, hide_index=True)

        chart_df = df_date.set_index("날짜")[["정답률(%)"]]
        st.bar_chart(chart_df)

    st.divider()

    st.subheader("취약 영역")
    weak_areas = session.get_weak_areas()
    if weak_areas:
        for area in weak_areas:
            st.warning(
                f"**{area['date']}** 강의: 정답률 {area['accuracy']:.0f}% "
                f"({area['correct']}/{area['total']}문항)"
            )
    else:
        st.success("취약 영역이 없습니다! 잘하고 있습니다.")

    st.divider()

    st.subheader("학습 추천")
    recommendations = session.get_recommendations()
    for rec in recommendations:
        st.markdown(f"- {rec}")

    st.divider()

    st.subheader("오답 노트")
    wrong_questions = session.get_wrong_questions()
    if wrong_questions:
        for i, wq in enumerate(wrong_questions, 1):
            with st.expander(f"오답 {i}: Q{wq.quiz_id} ({wq.source_date})"):
                st.markdown(f"**유형:** {QUIZ_TYPE_NAMES.get(wq.quiz_type, wq.quiz_type)}")
                st.markdown(f"**난이도:** {DIFFICULTY_NAMES.get(wq.difficulty, wq.difficulty)}")
                st.markdown(f"**내 답변:** {wq.user_answer}")
                st.markdown(f"**정답:** {wq.correct_answer}")
                if wq.explanation:
                    st.markdown(f"**해설:** {wq.explanation}")
    else:
        st.success("오답이 없습니다!")

    st.divider()
    if st.button("학습 기록 초기화", type="secondary"):
        st.session_state.quiz_session = SessionStats()
        st.rerun()


if __name__ == "__main__":
    main()
