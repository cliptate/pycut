"""Transcript JSON persistence for media jobs."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from pycut.utils import Segment


@dataclass
class TranscriptMetadata:
    """Metadata stored next to transcript segments."""

    title: str = ""
    subtitle: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
        }


@dataclass
class TranscriptDocument:
    """Loaded transcript segments and metadata."""

    segments: List[Segment]
    metadata: TranscriptMetadata = field(default_factory=TranscriptMetadata)


def _segment_to_dict(segment: Segment) -> dict:
    return {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": segment.words or [],
    }


def load_transcript(path: str | Path) -> TranscriptDocument:
    """Load transcript JSON, accepting both current and legacy formats."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_segments = payload
        metadata = TranscriptMetadata()
    elif isinstance(payload, dict):
        raw_segments = payload.get("segments", []) or []
        metadata = TranscriptMetadata(
            title=str(payload.get("title", "") or ""),
            subtitle=str(payload.get("subtitle", "") or ""),
        )
    else:
        print(f"⚠️  Unexpected transcript JSON root type {type(payload).__name__}, treating as empty")
        raw_segments = []
        metadata = TranscriptMetadata()

    segments: List[Segment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            text = str(item.get("text", ""))
            words = list(item.get("words", []) or [])
            segments.append(Segment(start=start, end=end, text=text, words=words))
        except (TypeError, ValueError):
            continue

    return TranscriptDocument(segments=segments, metadata=metadata)


def save_transcript(
    path: str | Path,
    segments: List[Segment],
    metadata: TranscriptMetadata | None = None,
) -> str:
    """Write transcript JSON in the current object format."""
    resolved_metadata = metadata or TranscriptMetadata()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": resolved_metadata.title,
        "subtitle": resolved_metadata.subtitle,
        "segments": [_segment_to_dict(segment) for segment in segments],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(output)


class TranscriptStore:
    """Filesystem adapter for per-media transcript persistence."""

    def __init__(self, output_dir: str | Path, media_stem: str):
        self.output_dir = Path(output_dir)
        self.media_stem = media_stem
        self.path = self.output_dir / f"{media_stem}_transcript.json"

    def load_existing(self) -> TranscriptDocument | None:
        if not self.path.exists():
            return None
        return load_transcript(self.path)

    def load_provided(self, transcript_path: str | Path) -> TranscriptDocument:
        document = load_transcript(transcript_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        resolved_src = os.path.realpath(transcript_path)
        resolved_dst = os.path.realpath(self.path) if self.path.exists() else None
        if resolved_src != resolved_dst:
            shutil.copy2(transcript_path, self.path)
        return document

    def save(
        self,
        segments: List[Segment],
        metadata: TranscriptMetadata | None = None,
    ) -> str:
        return save_transcript(self.path, segments, metadata=metadata)


def load_segments_with_meta(transcript_path: str) -> Tuple[List[Segment], dict]:
    """Compatibility helper returning the old ``(segments, metadata dict)`` tuple."""
    document = load_transcript(transcript_path)
    return document.segments, document.metadata.to_dict()
