"""
새 강의 스크립트 인제스트(전처리→청킹→임베딩→기존 벡터DB 증분 업데이트) 엔트리포인트.

Streamlit 없이 동작하며, Backend-API 백그라운드 작업에서 호출하기 위한 형태로 구성합니다.
"""

from __future__ import annotations

from pathlib import Path

from .common import load_config
from .embeddings import upsert_vectorstore_from_chunks
from .preprocessing import chunks_from_uploaded_lecture_text, load_curriculum, process_script_file


def ingest_lecture_script(
    script_file: str | Path,
    *,
    curriculum_csv: str | Path | None = None,
    force_rebuild: bool = False,
) -> dict:
    """
    Args:
        script_file: STT 원문 텍스트 파일(.txt) 경로
        curriculum_csv: 커리큘럼 CSV 경로 (미지정 시 config.yaml의 paths 사용)
        force_rebuild: 벡터DB 차원 불일치 등 안전 모드 필요 시 전체 재구축

    Returns:
        dict: ingest 처리 결과(상태/추가된 청크 수/스킵 수 등)
    """
    base_dir = Path(__file__).resolve().parent.parent
    cfg = load_config()

    script_path = Path(script_file)
    if not script_path.exists():
        raise FileNotFoundError(f"강의 스크립트 파일을 찾을 수 없습니다: {script_path}")

    if curriculum_csv is None:
        curriculum_path = base_dir / cfg["paths"]["curriculum_csv"]
    else:
        curriculum_path = Path(curriculum_csv)

    curriculum_df = load_curriculum(curriculum_path)

    # process_script_file 내부에서 date_str(파일명 규칙) 기반 메타데이터가 생성됩니다.
    config_dict = cfg
    chunks = process_script_file(script_path, curriculum_df, config=config_dict)

    result = upsert_vectorstore_from_chunks(chunks, force_rebuild=force_rebuild)
    return {
        "script_file": str(script_path),
        "total_new_chunks": len(chunks),
        **result,
    }


def ingest_lecture_upload(
    content: str,
    *,
    lecture_id: str,
    week: int | None = None,
    subject: str | None = None,
    instructor: str | None = None,
    session: str | None = None,
    date_str: str | None = None,
    title: str | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Backend에서 업로드한 강의 본문을 전역 벡터DB에 증분 반영합니다.

    메타데이터에 ``lecture_id``가 들어가므로 이후 RAG 검색 시 해당 강의만 대상으로 할 수 있습니다.

    Args:
        content: 강의 전체 텍스트 (STT 또는 평문)
        lecture_id: Backend ``Lecture.id`` (문자열 UUID)
        week, subject, instructor, session, date_str, title: 선택 메타데이터
        force_rebuild: 벡터DB 전체 재구축이 필요할 때 True

    Returns:
        upsert 결과 dict에 ``lecture_id`` 키를 추가한 형태
    """
    cfg = load_config()
    config_dict = cfg

    chunks = chunks_from_uploaded_lecture_text(
        content,
        lecture_id=lecture_id,
        week=week if week is not None else 0,
        subject=subject or "",
        instructor=instructor or "",
        session=session or "",
        date_str=date_str or "",
        title=title or "",
        learning_goal="",
        config=config_dict,
    )

    if not chunks:
        return {
            "lecture_id": lecture_id,
            "total_new_chunks": 0,
            "status": "empty",
            "added_chunks": 0,
            "skipped_chunks": 0,
            "embedding_backend": cfg.get("embedding_backend", "openai"),
        }

    result = upsert_vectorstore_from_chunks(chunks, force_rebuild=force_rebuild)
    return {
        "lecture_id": lecture_id,
        "total_new_chunks": len(chunks),
        **result,
    }

