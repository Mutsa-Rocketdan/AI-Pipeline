"""학습 가이드 생성 모듈: 주차별 요약, 핵심 개념, 복습 포인트, 개념 맵."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .prompts import SYSTEM_PROMPT, get_study_guide_prompt, get_daily_guide_prompt
from .rag import get_all_chunks_for_date, get_all_chunks_for_week
from .preprocessing import load_curriculum
from .common import (
    get_openai_client,
    guide_cache_path,
    load_config,
    load_json,
    save_json,
)

def _load_cache(prefix: str, key: str) -> dict | None:
    return load_json(guide_cache_path(prefix, key))


def _save_cache(prefix: str, key: str, data: dict) -> None:
    save_json(guide_cache_path(prefix, key), data)


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def _get_curriculum_info(week: int) -> dict:
    """커리큘럼에서 주차 정보 추출."""
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / config["paths"]["curriculum_csv"]
    df = load_curriculum(csv_path)
    week_rows = df[df["week"] == week]

    if week_rows.empty:
        return {"subject": "", "learning_goal": "", "dates": []}

    subjects = week_rows["subject"].unique().tolist()
    goals = week_rows["learning_goal"].unique().tolist()
    dates = sorted(week_rows["date"].unique().tolist())

    return {
        "subject": ", ".join(subjects),
        "learning_goal": " / ".join(goals),
        "dates": dates,
    }


def generate_daily_guide(date: str, use_cache: bool = True) -> dict:
    """특정 날짜의 강의 학습 가이드 생성."""
    cache_key = f"daily_{date}"
    if use_cache:
        cached = _load_cache("guide", cache_key)
        if cached:
            return cached

    config = load_config()
    chunks = get_all_chunks_for_date(date)

    if not chunks:
        return {"date": date, "summary": "해당 날짜의 강의 데이터가 없습니다.", "key_concepts": [], "review_points": []}

    context = "\n\n".join(c.get("text", "") for c in chunks[:15])
    subject = chunks[0].get("subject", "")
    content = chunks[0].get("content", "")
    learning_goal = chunks[0].get("learning_goal", "")

    prompt = get_daily_guide_prompt(
        context=context, date=date,
        subject=subject, content=content, learning_goal=learning_goal,
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=config["openai"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=config["guide"]["summary_max_tokens"],
    )

    result = _parse_json_response(response.choices[0].message.content or "")
    if not result:
        result = {"date": date, "summary": "가이드 생성에 실패했습니다.", "key_concepts": [], "review_points": []}

    _save_cache("guide", cache_key, result)
    return result


def generate_weekly_guide(week: int, use_cache: bool = True) -> dict:
    """주차별 종합 학습 가이드 생성."""
    cache_key = f"weekly_{week}"
    if use_cache:
        cached = _load_cache("guide", cache_key)
        if cached:
            return cached

    config = load_config()
    curriculum_info = _get_curriculum_info(week)
    chunks = get_all_chunks_for_week(week)

    if not chunks:
        return {
            "week": week,
            "weekly_summary": "해당 주차의 강의 데이터가 없습니다.",
            "daily_summaries": [],
            "key_concepts": [],
            "review_points": [],
            "concept_relations": [],
        }

    context_texts = [c.get("text", "") for c in chunks]
    max_chunks = 20
    if len(context_texts) > max_chunks:
        step = len(context_texts) // max_chunks
        context_texts = context_texts[::step][:max_chunks]
    context = "\n\n".join(context_texts)

    dates_str = ", ".join(curriculum_info["dates"]) if curriculum_info["dates"] else "정보 없음"

    prompt = get_study_guide_prompt(
        context=context,
        week=week,
        dates=dates_str,
        subject=curriculum_info["subject"],
        learning_goal=curriculum_info["learning_goal"],
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=config["openai"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=config["openai"]["max_tokens"],
    )

    result = _parse_json_response(response.choices[0].message.content or "")
    if not result:
        result = {
            "week": week,
            "weekly_summary": "가이드 생성에 실패했습니다.",
            "daily_summaries": [],
            "key_concepts": [],
            "review_points": [],
            "concept_relations": [],
        }
    result["week"] = week

    _save_cache("guide", cache_key, result)
    return result


def get_available_weeks() -> list[int]:
    """사용 가능한 주차 목록."""
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / config["paths"]["curriculum_csv"]
    df = load_curriculum(csv_path)
    return sorted(df["week"].unique().tolist())


def build_concept_map_mermaid(concept_relations: list[dict]) -> str:
    """개념 관계를 Mermaid 다이어그램 코드로 변환."""
    if not concept_relations:
        return ""

    lines = ["graph TD"]
    node_ids = {}
    counter = 0

    for rel in concept_relations:
        from_name = rel.get("from", "")
        to_name = rel.get("to", "")
        relation = rel.get("relation", "")

        if from_name not in node_ids:
            node_ids[from_name] = f"N{counter}"
            counter += 1
        if to_name not in node_ids:
            node_ids[to_name] = f"N{counter}"
            counter += 1

        from_id = node_ids[from_name]
        to_id = node_ids[to_name]
        safe_from = from_name.replace('"', "'")
        safe_to = to_name.replace('"', "'")
        safe_rel = relation.replace('"', "'")

        lines.append(f'    {from_id}["{safe_from}"] -->|"{safe_rel}"| {to_id}["{safe_to}"]')

    return "\n".join(lines)
