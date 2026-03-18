"""STT 강의 스크립트 전처리 및 청킹 모듈."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tiktoken

from .common import load_config


@dataclass
class ChunkMetadata:
    date: str
    week: int
    subject: str
    content: str
    learning_goal: str
    session: str = ""
    instructor: str = ""


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata
    chunk_id: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "date": self.metadata.date,
            "week": self.metadata.week,
            "subject": self.metadata.subject,
            "content": self.metadata.content,
            "learning_goal": self.metadata.learning_goal,
            "session": self.metadata.session,
            "instructor": self.metadata.instructor,
        }


_TIMESTAMP_PATTERN = re.compile(r"<\d{2}:\d{2}:\d{2}>\s*")
_SPEAKER_PATTERN = re.compile(r"^[a-f0-9]{8}:\s*", re.MULTILINE)
_TIMESTAMP_EXTRACT = re.compile(r"<(\d{2}):(\d{2}):(\d{2})>")


def _time_to_seconds(h: str, m: str, s: str) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_curriculum(csv_path: str | Path) -> pd.DataFrame:
    """커리큘럼 CSV 로드 및 날짜별 메타데이터 구성."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["date"] = df["date"].astype(str).str.strip()
    return df


def get_metadata_for_date(curriculum_df: pd.DataFrame, date_str: str) -> ChunkMetadata:
    """특정 날짜에 해당하는 커리큘럼 메타데이터 반환."""
    rows = curriculum_df[curriculum_df["date"] == date_str]
    if rows.empty:
        return ChunkMetadata(
            date=date_str, week=0,
            subject="Unknown", content="Unknown", learning_goal="Unknown",
        )
    first = rows.iloc[0]
    sessions = rows["session"].unique().tolist() if "session" in rows.columns else []
    return ChunkMetadata(
        date=date_str,
        week=int(first.get("week", 0)),
        subject=str(first.get("subject", "")),
        content=str(first.get("content", "")),
        learning_goal=str(first.get("learning_goal", "")),
        session=", ".join(sessions),
        instructor=str(first.get("instructor", "")),
    )


def parse_stt_lines(raw_text: str) -> list[dict]:
    """STT 원본 텍스트를 (timestamp_seconds, clean_text) 리스트로 파싱."""
    lines = raw_text.strip().split("\n")
    parsed = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ts_match = _TIMESTAMP_EXTRACT.search(line)
        ts_seconds = 0
        if ts_match:
            ts_seconds = _time_to_seconds(ts_match.group(1), ts_match.group(2), ts_match.group(3))
        clean = _TIMESTAMP_PATTERN.sub("", line)
        clean = _SPEAKER_PATTERN.sub("", clean).strip()
        if clean:
            parsed.append({"time": ts_seconds, "text": clean})
    return parsed


def segment_by_gap(parsed_lines: list[dict], gap_seconds: int = 30) -> list[str]:
    """시간 간격 기준으로 연속 발화를 세그먼트로 묶기."""
    if not parsed_lines:
        return []

    segments: list[list[str]] = [[]]
    prev_time = parsed_lines[0]["time"]

    for item in parsed_lines:
        if item["time"] - prev_time > gap_seconds and segments[-1]:
            segments.append([])
        segments[-1].append(item["text"])
        prev_time = item["time"]

    return [" ".join(seg) for seg in segments if seg]


def chunk_text(text: str, max_tokens: int = 600, overlap_tokens: int = 100) -> list[str]:
    """텍스트를 토큰 기준 오버랩 청킹."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += max_tokens - overlap_tokens

    return chunks


def process_script_file(
    file_path: str | Path,
    curriculum_df: pd.DataFrame,
    config: dict | None = None,
) -> list[Chunk]:
    """단일 STT 스크립트 파일을 전처리하여 Chunk 리스트 반환."""
    if config is None:
        config = load_config()

    file_path = Path(file_path)
    date_str = file_path.stem.split("_")[0]  # e.g. "2026-02-02"

    with open(file_path, encoding="utf-8") as f:
        raw_text = f.read()

    metadata = get_metadata_for_date(curriculum_df, date_str)
    parsed = parse_stt_lines(raw_text)
    segments = segment_by_gap(
        parsed,
        gap_seconds=config.get("preprocessing", {}).get("segment_gap_seconds", 30),
    )

    chunk_size = config.get("preprocessing", {}).get("chunk_size", 600)
    chunk_overlap = config.get("preprocessing", {}).get("chunk_overlap", 100)

    chunks: list[Chunk] = []
    for segment in segments:
        text_chunks = chunk_text(segment, max_tokens=chunk_size, overlap_tokens=chunk_overlap)
        for tc in text_chunks:
            chunks.append(Chunk(text=tc, metadata=metadata))

    return chunks


def process_all_scripts(
    scripts_dir: str | Path | None = None,
    curriculum_path: str | Path | None = None,
) -> list[Chunk]:
    """모든 STT 스크립트 파일을 처리하여 전체 Chunk 리스트 반환."""
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent

    if scripts_dir is None:
        scripts_dir = base_dir / config["paths"]["scripts_dir"]
    if curriculum_path is None:
        curriculum_path = base_dir / config["paths"]["curriculum_csv"]

    scripts_dir = Path(scripts_dir)
    curriculum_df = load_curriculum(curriculum_path)

    all_chunks: list[Chunk] = []
    script_files = sorted(scripts_dir.glob("*.txt"))

    for fpath in script_files:
        file_chunks = process_script_file(fpath, curriculum_df, config)
        all_chunks.extend(file_chunks)

    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_id = idx

    return all_chunks


def get_available_dates(scripts_dir: str | Path | None = None) -> list[str]:
    """사용 가능한 강의 날짜 목록 반환."""
    config = load_config()
    base_dir = Path(__file__).resolve().parent.parent

    if scripts_dir is None:
        scripts_dir = base_dir / config["paths"]["scripts_dir"]

    scripts_dir = Path(scripts_dir)
    dates = []
    for fpath in sorted(scripts_dir.glob("*.txt")):
        date_str = fpath.stem.split("_")[0]
        dates.append(date_str)
    return dates
