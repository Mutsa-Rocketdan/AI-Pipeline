"""
새 강의 스크립트 인제스트(전처리→청킹→임베딩→기존 벡터DB 증분 업데이트) 엔트리포인트.

Streamlit 없이 동작하며, Backend-API 백그라운드 작업에서 호출하기 위한 형태로 구성합니다.
"""

from __future__ import annotations

from pathlib import Path

from .common import load_config
from .embeddings import upsert_vectorstore_from_chunks
from .preprocessing import load_curriculum, process_script_file


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

