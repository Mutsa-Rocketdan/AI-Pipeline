"""RAG 검색 파이프라인: 메타데이터 필터링 + FAISS 유사도 검색."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .embeddings import (
    create_embeddings,
    load_vectorstore,
    build_vectorstore,
    vectorstore_exists,
)
from .common import load_config


def _get_vectorstore() -> tuple[faiss.IndexFlatIP, list[dict]]:
    if vectorstore_exists():
        return load_vectorstore()
    return build_vectorstore()


def filter_metadata(
    metadata_list: list[dict],
    date: str | None = None,
    week: int | None = None,
    subject: str | None = None,
    lecture_id: str | None = None,
) -> list[int]:
    """메타데이터 기준으로 필터링하여 해당 인덱스 반환."""
    indices = []
    for i, meta in enumerate(metadata_list):
        if lecture_id and meta.get("lecture_id") != lecture_id:
            continue
        if date and meta.get("date") != date:
            continue
        if week is not None and meta.get("week") != week:
            continue
        if subject and subject.lower() not in meta.get("subject", "").lower():
            continue
        indices.append(i)
    return indices


def search(
    query: str,
    date: str | None = None,
    week: int | None = None,
    subject: str | None = None,
    lecture_id: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """RAG 검색: 메타데이터 필터링 후 유사도 기반 top-k 청크 반환.

    ``lecture_id``가 지정되면 해당 강의로 인제스트된 청크만 검색합니다.
    일치하는 청크가 없으면 빈 리스트를 반환합니다(전역 DB를 열어두지 않음).
    """
    config = load_config()
    if top_k is None:
        top_k = config["vectorstore"]["top_k"]

    index, metadata_list = _get_vectorstore()

    filtered_indices = filter_metadata(
        metadata_list,
        date=date,
        week=week,
        subject=subject,
        lecture_id=lecture_id,
    )

    strict = bool(lecture_id or date or week is not None or subject)
    if not filtered_indices:
        if strict:
            return []
        filtered_indices = list(range(len(metadata_list)))

    query_embedding = create_embeddings([query])
    faiss.normalize_L2(query_embedding)

    n_total = index.ntotal
    search_k = min(n_total, max(top_k * 5, 50))
    scores, indices = index.search(query_embedding, search_k)

    filtered_set = set(filtered_indices)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if int(idx) in filtered_set:
            result = metadata_list[int(idx)].copy()
            result["score"] = float(score)
            results.append(result)
            if len(results) >= top_k:
                break

    return results


def get_context_for_generation(
    topic: str,
    date: str | None = None,
    week: int | None = None,
    lecture_id: str | None = None,
    top_k: int = 8,
) -> str:
    """퀴즈/가이드 생성에 사용할 컨텍스트 텍스트 조합."""
    results = search(query=topic, date=date, week=week, lecture_id=lecture_id, top_k=top_k)
    context_parts = []
    for r in results:
        context_parts.append(f"[{r.get('date', '')} | {r.get('subject', '')}]\n{r.get('text', '')}")
    return "\n\n---\n\n".join(context_parts)


def get_all_chunks_for_date(date: str) -> list[dict]:
    """특정 날짜의 모든 청크를 순서대로 반환 (학습 가이드용)."""
    _, metadata_list = _get_vectorstore()
    return [m for m in metadata_list if m.get("date") == date]


def get_all_chunks_for_week(week: int) -> list[dict]:
    """특정 주차의 모든 청크를 날짜순으로 반환."""
    _, metadata_list = _get_vectorstore()
    chunks = [m for m in metadata_list if m.get("week") == week]
    chunks.sort(key=lambda x: x.get("date", ""))
    return chunks


def get_all_chunks_for_lecture_id(lecture_id: str) -> list[dict]:
    """Backend ``lecture_id``로 인제스트된 모든 청크를 반환 (학습 가이드 등 전체 맥락용)."""
    _, metadata_list = _get_vectorstore()
    return [m for m in metadata_list if m.get("lecture_id") == lecture_id]
