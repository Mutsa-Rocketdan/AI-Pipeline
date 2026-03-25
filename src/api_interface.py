"""Backend-API가 호출할 수 있는 통합 인터페이스.

강의는 업로드 시 ``ingest_lecture_upload``로 전역 FAISS 벡터DB에 반영되고,
개념·퀴즈·가이드 생성 시 ``lecture_id``로 RAG 검색하여 컨텍스트를 구성합니다.
인제스트되지 않았거나 RAG 결과가 비면 ``Lecture.content``를 직접 청킹한 폴백을 사용합니다.
"""

from __future__ import annotations

import json
from typing import Any

from .common import get_openai_client, load_config
from .preprocessing import chunk_text
from .prompts import SYSTEM_PROMPT, DIFFICULTY_GUIDE, QUIZ_TEMPLATES
from .feedback import check_answer, SessionStats, QuizResult


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _build_context(content: str, max_chunks: int = 15) -> str:
    """원문을 청킹해서 LLM 컨텍스트 문자열로 합침 (RAG 폴백)."""
    config = load_config()
    chunk_size = config.get("preprocessing", {}).get("chunk_size", 600)
    overlap = config.get("preprocessing", {}).get("chunk_overlap", 100)
    chunks = chunk_text(content, max_tokens=chunk_size, overlap_tokens=overlap)
    return "\n\n".join(chunks[:max_chunks])


def _context_from_rag_or_content(
    lecture_id: str | None,
    content: str,
    *,
    search_query: str,
    top_k: int = 15,
) -> str:
    """``lecture_id``가 있으면 RAG 검색으로 컨텍스트를 만들고, 없거나 실패 시 content 청킹."""
    if not lecture_id or not lecture_id.strip():
        return _build_context(content)
    try:
        from .rag import search

        rows = search(query=search_query, lecture_id=lecture_id.strip(), top_k=top_k)
        if not rows:
            return _build_context(content)
        parts = []
        for r in rows:
            subj = r.get("subject", "")
            parts.append(f"[{subj}]\n{r.get('text', '')}")
        return "\n\n---\n\n".join(parts)
    except Exception:
        return _build_context(content)


def _guide_context_from_rag_or_content(lecture_id: str | None, content: str) -> str:
    """가이드용: 해당 강의의 전체 청크를 우선 합치고, 없으면 폴백."""
    if not lecture_id or not lecture_id.strip():
        return _build_context(content)
    try:
        from .rag import get_all_chunks_for_lecture_id

        chunks = get_all_chunks_for_lecture_id(lecture_id.strip())
        if not chunks:
            return _build_context(content)
        texts = [c.get("text", "") for c in chunks[:25]]
        joined = "\n\n".join(t for t in texts if t)
        return joined if joined.strip() else _build_context(content)
    except Exception:
        return _build_context(content)


def _call_llm(prompt: str, *, system: str = SYSTEM_PROMPT, temperature: float | None = None, max_tokens: int | None = None) -> str:
    config = load_config()
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=config["openai"]["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature or config["openai"]["temperature"],
        max_tokens=max_tokens or config["openai"]["max_tokens"],
    )
    return resp.choices[0].message.content or ""


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return []


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# 1. 개념 추출 (Backend Concept 스키마)
# ---------------------------------------------------------------------------

_CONCEPT_PROMPT = """아래 강의 내용을 분석하여 핵심 개념(지식)을 추출하세요.

[강의 내용]
{context}

[출력 형식]
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
[
  {{
    "concept_name": "핵심 개념/용어 이름",
    "description": "해당 개념에 대한 간결한 설명 (1~2문장)"
  }}
]
```

규칙:
- 강의에서 실제로 다룬 내용만 추출하세요.
- 8~15개 사이의 핵심 개념을 추출하세요.
- 중요도가 높은 순서로 정렬하세요.
- 한국어로 작성하세요."""


def generate_concepts(content: str, lecture_id: str | None = None) -> list[dict[str, Any]]:
    """강의 content에서 개념을 추출합니다. ``lecture_id``가 있으면 RAG로 검색된 청크를 우선 사용합니다.

    Returns:
        list[{"concept_name": str, "description": str, "mastery_score": float}]
        Backend의 models.Concept 스키마에 대응합니다.
    """
    context = _context_from_rag_or_content(
        lecture_id,
        content,
        search_query="강의 핵심 개념 용어 정의 학습 목표",
        top_k=15,
    )
    prompt = _CONCEPT_PROMPT.format(context=context)
    raw = _call_llm(prompt, temperature=0.3)
    concepts = _parse_json_array(raw)

    results = []
    for c in concepts:
        results.append({
            "concept_name": c.get("concept_name", c.get("term", "")),
            "description": c.get("description", c.get("definition", "")),
            "mastery_score": 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# 2. 퀴즈 생성 (Backend QuizQuestion 스키마)
# ---------------------------------------------------------------------------

def generate_quiz_questions(
    content: str,
    quiz_type: str = "multiple_choice",
    difficulty: str = "medium",
    count: int = 5,
    lecture_id: str | None = None,
) -> list[dict[str, Any]]:
    """강의 content 기반 퀴즈 문항을 생성합니다. ``lecture_id``가 있으면 RAG 컨텍스트를 사용합니다.

    Returns:
        list[{"question_text": str, "options": list|None,
              "correct_answer": str, "explanation": str}]
        Backend의 models.QuizQuestion 스키마에 대응합니다.
    """
    context = _context_from_rag_or_content(
        lecture_id,
        content,
        search_query="강의 중요 내용 핵심 키워드 문제 출제",
        top_k=15,
    )
    difficulty_guide = DIFFICULTY_GUIDE.get(difficulty, DIFFICULTY_GUIDE["medium"])
    template = QUIZ_TEMPLATES.get(quiz_type, QUIZ_TEMPLATES["multiple_choice"])
    prompt = template.format(
        context=context,
        learning_goal="",
        difficulty=difficulty,
        difficulty_guide=difficulty_guide,
        count=count,
    )

    raw = _call_llm(prompt)
    parsed = _parse_json_array(raw)

    results = []
    for q in parsed:
        question_text = q.get("question", q.get("question_text", ""))
        options = q.get("options")
        correct_answer = q.get("answer", q.get("correct_answer", ""))
        explanation = q.get("explanation", "")

        if options and correct_answer not in options:
            options[-1] = correct_answer

        results.append({
            "question_text": question_text,
            "options": options if options else [],
            "correct_answer": correct_answer,
            "explanation": explanation,
        })
    return results


# ---------------------------------------------------------------------------
# 3. 학습 가이드 생성 (Backend Guide 스키마)
# ---------------------------------------------------------------------------

_GUIDE_PROMPT = """아래 강의 내용을 분석하여 학습 가이드를 생성하세요.

[강의 내용]
{context}

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
```json
{{
  "summary": "강의 전체 핵심 내용 요약 (3~5문단)",
  "key_summaries": [
    "핵심 요약 포인트 1",
    "핵심 요약 포인트 2"
  ],
  "review_checklist": [
    "스스로 점검해볼 복습 항목 1",
    "스스로 점검해볼 복습 항목 2"
  ],
  "concept_map": {{
    "nodes": ["개념A", "개념B", "개념C"],
    "edges": [
      {{"from": "개념A", "to": "개념B"}}
    ]
  }}
}}
```

규칙:
- 핵심 요약(key_summaries)은 5~10개 항목으로 작성하세요.
- 복습 체크리스트(review_checklist)는 학습자가 자기 점검할 수 있는 질문 형태로 5~8개 작성하세요.
- 개념 맵의 nodes는 주요 개념 5~10개, edges는 개념 간 관계를 표현하세요.
- 한국어로 작성하세요."""


def generate_study_guide(content: str, lecture_id: str | None = None) -> dict[str, Any]:
    """강의 content 기반 학습 가이드를 생성합니다. ``lecture_id``가 있으면 인제스트된 전체 청크를 우선 사용합니다.

    Returns:
        {"summary": str, "key_summaries": list[str],
         "review_checklist": list[str], "concept_map": dict}
        Backend의 models.Guide 스키마에 대응합니다.
    """
    context = _guide_context_from_rag_or_content(lecture_id, content)
    prompt = _GUIDE_PROMPT.format(context=context)
    raw = _call_llm(prompt, temperature=0.5)
    parsed = _parse_json_object(raw)

    return {
        "summary": parsed.get("summary", "가이드 생성에 실패했습니다."),
        "key_summaries": parsed.get("key_summaries", []),
        "review_checklist": parsed.get("review_checklist", []),
        "concept_map": parsed.get("concept_map", {"nodes": [], "edges": []}),
    }


# ---------------------------------------------------------------------------
# 4. 퀴즈 채점 및 피드백 (Backend QuizResult 스키마)
# ---------------------------------------------------------------------------

def evaluate_quiz(
    questions: list[dict],
    user_answers: list[str],
) -> dict[str, Any]:
    """퀴즈 채점 후 점수·피드백을 반환합니다.

    Args:
        questions: DB에서 가져온 QuizQuestion 목록
                   (question_text, options, correct_answer, explanation 포함)
        user_answers: 사용자가 제출한 답변 리스트 (questions와 같은 순서)

    Returns:
        {"score": int, "ai_feedback": str}
        Backend의 models.QuizResult에 저장 가능한 형태입니다.
    """
    stats = SessionStats()
    for i, q in enumerate(questions):
        ua = user_answers[i] if i < len(user_answers) else ""
        ca = q.get("correct_answer", "")
        qtype = "multiple_choice" if q.get("options") else "short_answer"
        is_correct = check_answer(ua, ca, quiz_type=qtype)

        stats.results.append(QuizResult(
            quiz_id=i + 1,
            quiz_type=qtype,
            topic="",
            difficulty="",
            source_date="",
            user_answer=ua,
            correct_answer=ca,
            is_correct=is_correct,
            explanation=q.get("explanation", ""),
        ))

    recommendations = stats.get_recommendations()

    return {
        "score": stats.score,
        "ai_feedback": json.dumps(recommendations, ensure_ascii=False),
    }
