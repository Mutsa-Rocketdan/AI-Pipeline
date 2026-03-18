"""프로젝트 공통 유틸리티 모듈.

- config.yaml 로딩
- .env 로딩 후 OpenAI 클라이언트 생성
- generated_dir 기준 JSON 캐시 경로 생성/저장/로드
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


def get_base_dir() -> Path:
    """AI-Pipeline 루트(= config.yaml, .env가 있는 위치) 반환."""
    # src/common.py -> AI-Pipeline/src -> AI-Pipeline
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """config.yaml을 로드해 dict로 반환합니다."""
    config_path = get_base_dir() / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_openai_client() -> OpenAI:
    """프로젝트 .env를 로드한 뒤 OpenAI 클라이언트를 생성합니다."""
    from dotenv import load_dotenv

    load_dotenv(get_base_dir() / ".env")
    return OpenAI()


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def generated_dir() -> Path:
    """generated_dir 경로를 보장(디렉터리 생성)하고 Path를 반환합니다."""
    cfg = load_config()
    p = get_base_dir() / cfg["paths"]["generated_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def quiz_cache_path(cache_key: str) -> Path:
    """quiz 캐시 파일 경로 반환.

    기존 구현과 동일하게 `quiz_{cache_key}.json` 포맷을 사용합니다.
    """
    return generated_dir() / f"quiz_{cache_key}.json"


def guide_cache_path(prefix: str, key_text: str) -> Path:
    """guide 캐시 파일 경로 반환.

    기존 구현과 동일하게 `"{prefix}_{md5(key_text)}.json"` 포맷을 사용합니다.
    """
    return generated_dir() / f"{prefix}_{md5_hex(key_text)}.json"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

