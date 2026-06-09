"""File I/O utilities and constants for video/audio input processing."""

from __future__ import annotations

import fnmatch
import glob as _glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pycut.transcript_store import load_segments_with_meta
from pycut.utils import Segment

SUPPORTED_OUTPUT_FORMATS = ("ass", "srt", "fcpxml", "video", "txt", "json")
DEFAULT_OUTPUT_FORMATS = ("srt",)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS


def _parse_output_formats(raw_value: str) -> List[str]:
    """Parse comma-separated output formats and validate strictly."""
    if raw_value is None:
        return list(DEFAULT_OUTPUT_FORMATS)

    values = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("output format list is empty")

    unique_values: List[str] = []
    for value in values:
        if value not in SUPPORTED_OUTPUT_FORMATS:
            supported = ", ".join(SUPPORTED_OUTPUT_FORMATS)
            raise ValueError(f"unsupported format '{value}', supported formats: {supported}")
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _normalize_output_formats(value: Optional[Iterable[str]]) -> List[str]:
    """Normalize output formats from iterable or comma-separated string."""
    if value is None:
        return list(DEFAULT_OUTPUT_FORMATS)
    if isinstance(value, str):
        return _parse_output_formats(value)
    return _parse_output_formats(",".join(str(v) for v in value))


def _is_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def _expand_video_inputs(raw_inputs: Iterable[str]) -> List[str]:
    """Expand CLI inputs to concrete video files (supports dir, glob, and multi-file)."""
    resolved: List[str] = []
    for raw in raw_inputs:
        candidate = Path(raw).expanduser()
        matches: List[Path] = []
        if candidate.is_dir():
            matches = sorted(
                (p for p in candidate.rglob("*") if _is_media_file(p)),
                key=lambda p: str(p).lower(),
            )
        elif candidate.exists() and _is_media_file(candidate):
            matches = [candidate]
        elif any(ch in raw for ch in "*?[]"):
            glob_pattern = str(candidate)
            matches = sorted(
                (Path(p) for p in _glob.glob(glob_pattern)),
                key=lambda p: str(p).lower(),
            )
            if not matches:
                parent = candidate.parent if str(candidate.parent) else Path(".")
                if parent.exists() and parent.is_dir():
                    name_pattern = candidate.name.lower()
                    matches = sorted(
                        (
                            p for p in parent.iterdir()
                            if fnmatch.fnmatch(p.name.lower(), name_pattern)
                        ),
                        key=lambda p: p.name.lower(),
                    )
            matches = [p for p in matches if _is_media_file(p)]
        resolved.extend(str(p.resolve()) for p in matches)

    deduped: List[str] = []
    seen: set = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _load_segments_from_transcript_json(transcript_path: str) -> Tuple[List[Segment], dict]:
    """Load segments and metadata from transcript JSON."""
    return load_segments_with_meta(transcript_path)
