"""임베딩 생성 및 FAISS 벡터DB 구축/관리 모듈."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import faiss
import numpy as np
from openai import APIError

from .preprocessing import Chunk, process_all_scripts
from .common import get_openai_client, load_config

# FAISS C++ 라이브러리는 Windows에서 유니코드 경로를 지원하지 않음.
# 프로젝트 경로에 한글이 있으면 ASCII만 있는 임시 디렉터리에 저장.
FAISS_LOCATION_FILE = "faiss_location.txt"


class EmbeddingQuotaError(Exception):
    """OpenAI API 할당량 초과 시 사용."""
    pass


def _path_has_non_ascii(p: Path) -> bool:
    """경로에 ASCII가 아닌 문자가 있으면 True (FAISS가 열지 못함)."""
    try:
        str(p).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _get_vectorstore_paths() -> tuple[Path, Path, Path]:
    """
    FAISS 인덱스/메타데이터에 쓸 실제 경로 반환.
    프로젝트 경로에 한글이 있으면 ASCII만 있는 임시 디렉터리 사용.
    Returns:
        (index_path, metadata_path, location_file_path)
        location_file_path: 프로젝트 내 위치 파일 (temp 사용 시 그 경로를 기록).
    """
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    rel_index = config["vectorstore"]["index_path"]
    rel_metadata = config["vectorstore"]["metadata_path"]
    store_dir = base_dir / Path(rel_index).parent
    location_file = store_dir / FAISS_LOCATION_FILE

    if not _path_has_non_ascii(base_dir):
        index_path = base_dir / rel_index
        metadata_path = base_dir / rel_metadata
        return index_path, metadata_path, location_file

    # 한글 등 비-ASCII 경로: 임시 디렉터리 사용 (FAISS는 ASCII 경로만 가능)
    key = hashlib.md5(str(base_dir).encode("utf-8")).hexdigest()
    temp_base = Path(tempfile.gettempdir()) / "create_quiz_guide_faiss" / key
    temp_base.mkdir(parents=True, exist_ok=True)
    index_path = temp_base / "faiss_index"
    metadata_path = temp_base / "metadata.json"
    return index_path, metadata_path, location_file


def _resolve_vectorstore_paths() -> tuple[Path, Path]:
    """
    저장된 벡터스토어의 실제 index_path, metadata_path 반환.
    이전에 한글 경로 때문에 temp에 저장했다면 location 파일에서 읽음.
    """
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent
    rel_index = config["vectorstore"]["index_path"]
    rel_metadata = config["vectorstore"]["metadata_path"]
    store_dir = base_dir / Path(rel_index).parent
    location_file = store_dir / FAISS_LOCATION_FILE

    if location_file.exists():
        try:
            temp_base = Path(location_file.read_text(encoding="utf-8").strip())
            return temp_base / "faiss_index", temp_base / "metadata.json"
        except Exception:
            pass

    return base_dir / rel_index, base_dir / rel_metadata


def create_embeddings_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """OpenAI API로 텍스트 리스트의 임베딩 벡터를 생성."""
    client = get_openai_client()
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def create_embeddings_local(texts: list[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """로컬 sentence-transformers 모델로 임베딩 생성 (API 불필요, 한글 지원)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 20)
    return np.array(embeddings, dtype=np.float32)


def create_embeddings(texts: list[str], model: str | None = None) -> np.ndarray:
    """config 기준으로 OpenAI 또는 로컬 임베딩 생성. (RAG 검색 시 쿼리 임베딩용 호환)."""
    config = load_config()
    backend = config.get("embedding_backend", "openai")
    if backend == "local":
        model_name = config.get("local_embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return create_embeddings_local(texts, model_name=model_name)
    openai_model = model or config["openai"]["embedding_model"]
    return create_embeddings_openai(texts, model=openai_model)


def build_vectorstore(chunks: list[Chunk] | None = None, force_rebuild: bool = False) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """FAISS 인덱스 빌드 및 저장. 이미 존재하면 로드."""
    index_path, metadata_path, location_file = _get_vectorstore_paths()

    if not force_rebuild and index_path.exists() and metadata_path.exists():
        return load_vectorstore()

    if chunks is None:
        chunks = process_all_scripts()

    texts = [c.text for c in chunks]
    metadata_list = [c.to_dict() for c in chunks]

    config = load_config()
    backend = config.get("embedding_backend", "openai")

    if backend == "local":
        model_name = config.get("local_embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = create_embeddings_local(texts, model_name=model_name)
    else:
        try:
            embedding_model = config["openai"]["embedding_model"]
            embeddings = create_embeddings_openai(texts, model=embedding_model)
        except APIError as e:
            err_body = getattr(e, "body", None) or {}
            err_info = err_body.get("error", {}) if isinstance(err_body, dict) else {}
            is_quota = (
                getattr(e, "status_code", None) == 429
                or err_info.get("code") == "insufficient_quota"
                or "quota" in str(err_info.get("message", "")).lower()
            )
            if is_quota:
                raise EmbeddingQuotaError(
                    "OpenAI API 할당량을 초과했습니다. "
                    "config.yaml에서 embedding_backend를 \"local\"로 바꾼 뒤 다시 벡터DB를 구축하세요. "
                    "(로컬 모델은 API 비용 없이 동작합니다.)"
                ) from e
            raise

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    # 한글 경로로 인해 temp에 저장한 경우, 프로젝트 내 location 파일에 기록 (로드 시 사용)
    if _path_has_non_ascii(Path(__file__).resolve().parent.parent):
        location_file.parent.mkdir(parents=True, exist_ok=True)
        location_file.write_text(str(index_path.parent), encoding="utf-8")

    return index, metadata_list


def load_vectorstore() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """저장된 FAISS 인덱스와 메타데이터 로드."""
    index_path, metadata_path = _resolve_vectorstore_paths()

    index = faiss.read_index(str(index_path))

    with open(metadata_path, encoding="utf-8") as f:
        metadata_list = json.load(f)

    return index, metadata_list


def vectorstore_exists() -> bool:
    """벡터스토어가 이미 빌드되어 있는지 확인."""
    index_path, metadata_path = _resolve_vectorstore_paths()
    return index_path.exists() and metadata_path.exists()
