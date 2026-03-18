"""퀴즈 풀기 페이지: 문항 표시, 답변 입력, 정답/해설 피드백."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import get_available_dates, load_curriculum
from src.quiz_generator import (
    generate_quiz,
    generate_mixed_quiz,
    get_available_quiz_types,
    get_available_difficulties,
)
from src.feedback import check_answer, QuizResult, SessionStats
from src.embeddings import vectorstore_exists

st.set_page_config(page_title="퀴즈 풀기", layout="wide")

QUIZ_TYPE_NAMES = {
    "multiple_choice": "객관식",
    "short_answer": "주관식",
    "fill_blank": "빈칸 채우기",
    "code": "코드",
}


def init_session_state():
    if "quiz_session" not in st.session_state:
        st.session_state.quiz_session = SessionStats()
    if "current_quizzes" not in st.session_state:
        st.session_state.current_quizzes = []
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}


def render_quiz_settings() -> dict | None:
    """사이드바에 퀴즈 설정 렌더링."""
    st.sidebar.header("퀴즈 설정")

    if not vectorstore_exists():
        st.sidebar.error("벡터DB가 구축되지 않았습니다. 메인 페이지에서 먼저 구축하세요.")
        return None

    dates = get_available_dates()
    if not dates:
        st.sidebar.error("강의 스크립트가 없습니다.")
        return None

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
            subject = rows.iloc[0].get("subject", "")
            content = rows.iloc[0].get("content", "")
            date_labels[d] = f"{d} ({subject} - {content})"
        else:
            date_labels[d] = d

    selected_date = st.sidebar.selectbox(
        "강의 날짜 선택",
        options=dates,
        format_func=lambda x: date_labels.get(x, x),
    )

    quiz_types = get_available_quiz_types()
    selected_type = st.sidebar.selectbox(
        "퀴즈 유형",
        options=[t["id"] for t in quiz_types],
        format_func=lambda x: next((t["name"] for t in quiz_types if t["id"] == x), x),
    )

    difficulties = get_available_difficulties()
    selected_difficulty = st.sidebar.selectbox(
        "난이도",
        options=[d["id"] for d in difficulties],
        format_func=lambda x: next(
            (f"{d['name']} - {d['description']}" for d in difficulties if d["id"] == x), x
        ),
        index=1,
    )

    count = st.sidebar.slider("문항 수", min_value=3, max_value=15, value=5)

    return {
        "date": selected_date,
        "quiz_type": selected_type,
        "difficulty": selected_difficulty,
        "count": count,
    }


def render_quiz_question(quiz: dict, index: int):
    """개별 퀴즈 문항 렌더링."""
    q_type = quiz.get("type", "multiple_choice")
    type_name = QUIZ_TYPE_NAMES.get(q_type, q_type)
    difficulty = quiz.get("difficulty", "medium")

    st.markdown(f"**Q{index + 1}.** `{type_name}` | 난이도: `{difficulty}`")
    st.markdown(quiz.get("question", ""))

    answer_key = f"answer_{index}"

    if q_type == "multiple_choice" and quiz.get("options"):
        options = quiz["options"]
        user_answer = st.radio(
            f"Q{index + 1} 답변 선택",
            options=options,
            key=answer_key,
            label_visibility="collapsed",
        )
        st.session_state.user_answers[index] = user_answer
    elif q_type == "code" and quiz.get("options"):
        options = quiz["options"]
        user_answer = st.radio(
            f"Q{index + 1} 답변 선택",
            options=options,
            key=answer_key,
            label_visibility="collapsed",
        )
        st.session_state.user_answers[index] = user_answer
    else:
        user_answer = st.text_input(
            f"Q{index + 1} 답변 입력",
            key=answer_key,
            label_visibility="collapsed",
            placeholder="답변을 입력하세요...",
        )
        st.session_state.user_answers[index] = user_answer


def render_result(quiz: dict, index: int, user_answer: str):
    """퀴즈 결과 렌더링."""
    correct_answer = quiz.get("answer", "")
    q_type = quiz.get("type", "multiple_choice")
    is_correct = check_answer(user_answer, correct_answer, q_type)

    type_name = QUIZ_TYPE_NAMES.get(q_type, q_type)

    if is_correct:
        st.success(f"**Q{index + 1}. 정답** `{type_name}`")
    else:
        st.error(f"**Q{index + 1}. 오답** `{type_name}`")

    st.markdown(quiz.get("question", ""))
    st.markdown(f"**내 답변:** {user_answer}")
    st.markdown(f"**정답:** {correct_answer}")

    with st.expander("해설 보기"):
        st.markdown(quiz.get("explanation", "해설이 제공되지 않았습니다."))

    return is_correct


def main():
    init_session_state()

    st.title("퀴즈 풀기")
    st.markdown("강의 내용을 기반으로 생성된 퀴즈를 풀어보세요.")

    settings = render_quiz_settings()
    if settings is None:
        return

    st.divider()

    if st.sidebar.button("퀴즈 생성", type="primary", use_container_width=True):
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}

        with st.spinner("퀴즈를 생성하고 있습니다..."):
            try:
                if settings["quiz_type"] == "mixed":
                    quizzes = generate_mixed_quiz(
                        date=settings["date"],
                        difficulty=settings["difficulty"],
                        count=settings["count"],
                    )
                else:
                    quizzes = generate_quiz(
                        date=settings["date"],
                        quiz_type=settings["quiz_type"],
                        difficulty=settings["difficulty"],
                        count=settings["count"],
                    )
                st.session_state.current_quizzes = quizzes
                if not quizzes:
                    st.warning("퀴즈를 생성하지 못했습니다. 다시 시도해 주세요.")
            except Exception as e:
                st.error(f"퀴즈 생성 중 오류가 발생했습니다: {e}")
                st.session_state.current_quizzes = []

    quizzes = st.session_state.current_quizzes

    if not quizzes:
        st.info("왼쪽 사이드바에서 설정을 선택하고 '퀴즈 생성' 버튼을 클릭하세요.")
        return

    if not st.session_state.quiz_submitted:
        st.subheader(f"퀴즈 ({len(quizzes)}문항)")

        for i, quiz in enumerate(quizzes):
            with st.container():
                render_quiz_question(quiz, i)
                st.divider()

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("제출하기", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()
    else:
        st.subheader("퀴즈 결과")

        correct_count = 0
        for i, quiz in enumerate(quizzes):
            user_answer = st.session_state.user_answers.get(i, "")
            is_correct = render_result(quiz, i, user_answer)
            if is_correct:
                correct_count += 1

            result = QuizResult(
                quiz_id=quiz.get("id", i + 1),
                quiz_type=quiz.get("type", "multiple_choice"),
                topic=quiz.get("topic", settings["date"]),
                difficulty=quiz.get("difficulty", "medium"),
                source_date=quiz.get("source_date", settings["date"]),
                user_answer=user_answer,
                correct_answer=quiz.get("answer", ""),
                is_correct=is_correct,
                explanation=quiz.get("explanation", ""),
            )
            if not any(
                r.quiz_id == result.quiz_id and r.source_date == result.source_date
                for r in st.session_state.quiz_session.results
            ):
                st.session_state.quiz_session.results.append(result)

            st.divider()

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("정답 수", f"{correct_count} / {len(quizzes)}")
        with res_col2:
            accuracy = correct_count / len(quizzes) * 100 if quizzes else 0
            st.metric("정답률", f"{accuracy:.0f}%")
        with res_col3:
            score = round(correct_count / len(quizzes) * 100) if quizzes else 0
            st.metric("점수", f"{score}점")

        if st.button("새 퀴즈 풀기", use_container_width=True):
            st.session_state.current_quizzes = []
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = {}
            st.rerun()


if __name__ == "__main__":
    main()
