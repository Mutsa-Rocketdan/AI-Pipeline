"""퀴즈 자동 생성 엔진: 4가지 유형 + 난이도 조절 + 해설 생성."""

from __future__ import annotations

import hashlib
import json

from .prompts import SYSTEM_PROMPT, get_quiz_prompt
from .rag import get_context_for_generation, get_all_chunks_for_date
from .common import (
    get_openai_client,
    load_config,
    load_json,
    quiz_cache_path,
    save_json,
)

def _cache_key(date: str, quiz_type: str, difficulty: str, count: int) -> str:
    raw = f"{date}_{quiz_type}_{difficulty}_{count}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(cache_key: str) -> list[dict] | None:
    cache_path = quiz_cache_path(cache_key)
    return load_json(cache_path)


def _save_cache(cache_key: str, data: list[dict]) -> None:
    cache_path = quiz_cache_path(cache_key)
    save_json(cache_path, data)


def _parse_llm_response(response_text: str) -> list[dict]:
    """LLM 응답에서 JSON 배열을 파싱."""
    text = response_text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return []


def generate_quiz(
    date: str,
    quiz_type: str = "multiple_choice",
    difficulty: str = "medium",
    count: int = 5,
    use_cache: bool = True,
) -> list[dict]:
    """특정 날짜의 강의 내용 기반 퀴즈 생성.

    Args:
        date: 강의 날짜 (YYYY-MM-DD)
        quiz_type: 퀴즈 유형 (multiple_choice, short_answer, fill_blank, code)
        difficulty: 난이도 (easy, medium, hard)
        count: 생성할 문항 수
        use_cache: 캐시 사용 여부

    Returns:
        퀴즈 문항 리스트
    """
    key = _cache_key(date, quiz_type, difficulty, count)
    if use_cache:
        cached = _load_cache(key)
        if cached:
            return cached

    config = load_config()

    date_chunks = get_all_chunks_for_date(date)
    if date_chunks:
        learning_goal = date_chunks[0].get("learning_goal", "")
        context_texts = [c.get("text", "") for c in date_chunks]
        context = "\n\n".join(context_texts[:15])
    else:
        context = get_context_for_generation(topic=f"{date} 강의 내용", date=date)
        learning_goal = ""

    prompt = get_quiz_prompt(
        quiz_type=quiz_type,
        context=context,
        learning_goal=learning_goal,
        difficulty=difficulty,
        count=count,
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=config["openai"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config["openai"]["temperature"],
        max_tokens=config["openai"]["max_tokens"],
    )

    response_text = response.choices[0].message.content or ""
    quizzes = _parse_llm_response(response_text)

    for i, quiz in enumerate(quizzes):
        quiz["id"] = i + 1
        quiz["source_date"] = date
        if "type" not in quiz:
            quiz["type"] = quiz_type
        if "difficulty" not in quiz:
            quiz["difficulty"] = difficulty

    _save_cache(key, quizzes)
    return quizzes


def generate_mixed_quiz(
    date: str,
    difficulty: str = "medium",
    count: int = 10,
    use_cache: bool = True,
) -> list[dict]:
    """여러 유형이 혼합된 퀴즈 세트 생성."""
    type_distribution = {
        "multiple_choice": max(1, count // 3),
        "short_answer": max(1, count // 4),
        "fill_blank": max(1, count // 4),
        "code": max(1, count - count // 3 - count // 4 - count // 4),
    }

    remaining = count - sum(type_distribution.values())
    if remaining > 0:
        type_distribution["multiple_choice"] += remaining

    all_quizzes = []
    for quiz_type, type_count in type_distribution.items():
        if type_count <= 0:
            continue
        quizzes = generate_quiz(
            date=date,
            quiz_type=quiz_type,
            difficulty=difficulty,
            count=type_count,
            use_cache=use_cache,
        )
        all_quizzes.extend(quizzes)

    for i, quiz in enumerate(all_quizzes):
        quiz["id"] = i + 1

    return all_quizzes


def get_available_quiz_types() -> list[dict]:
    """사용 가능한 퀴즈 유형 목록."""
    return [
        {"id": "multiple_choice", "name": "객관식 (5지선다)"},
        {"id": "short_answer", "name": "주관식 (단답형)"},
        {"id": "fill_blank", "name": "빈칸 채우기"},
        {"id": "code", "name": "코드 실행"},
        {"id": "mixed", "name": "혼합 출제"},
    ]


def get_available_difficulties() -> list[dict]:
    """사용 가능한 난이도 목록."""
    return [
        {"id": "easy", "name": "쉬움", "description": "기본 정의/용어 확인"},
        {"id": "medium", "name": "보통", "description": "개념 비교/적용"},
        {"id": "hard", "name": "어려움", "description": "복합 개념/코드 분석"},
    ]
