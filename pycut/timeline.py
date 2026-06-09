"""Transcript timeline preparation for subtitle and media exports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, List, Tuple

from pycut.utils import Segment


@dataclass
class TimelineCue:
    """One export-ready transcript cue."""

    start: float
    end: float
    text: str
    words: List[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptTimeline:
    """Prepared transcript timeline consumed by output adapters."""

    cues: List[TimelineCue]
    title: str = ""
    subtitle: str = ""

    @property
    def start(self) -> float:
        return self.cues[0].start if self.cues else 0.0

    @property
    def end(self) -> float:
        return self.cues[-1].end if self.cues else 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def text_lines(self) -> List[str]:
        return [cue.text.strip() for cue in self.cues if cue.text.strip()]


def split_transcript_segments(
    segments: Iterable[Segment],
    max_duration: float,
) -> List[List[Segment]]:
    """Split transcript segments into chunks with a maximum source duration."""
    segment_list = list(segments)
    if not segment_list:
        return []

    chunks: List[List[Segment]] = []
    current: List[Segment] = []
    chunk_start = None

    for seg in segment_list:
        if not current:
            current = [seg]
            chunk_start = seg.start
            continue

        if chunk_start is not None and seg.end - chunk_start <= max_duration:
            current.append(seg)
        else:
            chunks.append(current)
            current = [seg]
            chunk_start = seg.start

    if current:
        chunks.append(current)

    return chunks


def split_timeline_cues(
    cues: Iterable[TimelineCue],
    max_duration: float,
) -> List[List[TimelineCue]]:
    """Split timeline cues into chunks with a maximum source duration."""
    cue_list = list(cues)
    if not cue_list:
        return []

    chunks: List[List[TimelineCue]] = []
    current: List[TimelineCue] = []
    chunk_start = None

    for cue in cue_list:
        if not current:
            current = [cue]
            chunk_start = cue.start
            continue

        if chunk_start is not None and cue.end - chunk_start <= max_duration:
            current.append(cue)
        else:
            chunks.append(current)
            current = [cue]
            chunk_start = cue.start

    if current:
        chunks.append(current)

    return chunks


def resolve_overlaps(
    segments: Iterable[Segment],
    margin_left: float = 0.0,
    margin_right: float = 0.0,
) -> List[Segment]:
    """Apply margins and resolve adjacent overlaps using midpoint splits."""
    segment_list = list(segments)
    if not segment_list:
        return []

    shifted = [
        replace(seg, start=max(0.0, seg.start + margin_left), end=max(0.0, seg.end + margin_right))
        for seg in segment_list
    ]

    resolved: List[Segment] = []
    for index, seg in enumerate(shifted):
        start = seg.start
        end = seg.end
        if resolved:
            prev = resolved[-1]
            if start < prev.end:
                mid = (prev.end + start) / 2
                resolved[-1] = replace(prev, end=mid)
                start = mid
        if index + 1 < len(shifted):
            next_seg = shifted[index + 1]
            if end > next_seg.start:
                end = (end + next_seg.start) / 2
        resolved.append(replace(seg, start=start, end=end))
    return resolved


def prepare_timeline(
    segments: Iterable[Segment],
    *,
    title: str = "",
    subtitle: str = "",
    filter_empty_segments: bool = True,
    margin_left: float = 0.0,
    margin_right: float = 0.0,
) -> TranscriptTimeline:
    """Return an export-ready transcript timeline."""
    segment_list = list(segments)
    if filter_empty_segments:
        segment_list = [seg for seg in segment_list if str(getattr(seg, "text", "") or "").strip()]

    resolved = resolve_overlaps(segment_list, margin_left=margin_left, margin_right=margin_right)
    cues = [
        TimelineCue(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=list(seg.words or []),
        )
        for seg in resolved
    ]
    return TranscriptTimeline(cues=cues, title=title, subtitle=subtitle)


def prepare_export_timeline(
    timeline: TranscriptTimeline,
    max_duration: float,
) -> Tuple[TranscriptTimeline, List[List[TimelineCue]]]:
    """Return the export-ready timeline and cue chunks without leaving the timeline interface."""
    chunks = split_timeline_cues(timeline.cues, max_duration)
    cues = [cue for chunk in chunks for cue in chunk]
    return TranscriptTimeline(
        cues=[
            TimelineCue(
                start=cue.start,
                end=cue.end,
                text=cue.text,
                words=list(cue.words or []),
            )
            for cue in cues
        ],
        title=timeline.title,
        subtitle=timeline.subtitle,
    ), chunks


def timeline_to_segments(timeline: TranscriptTimeline) -> List[Segment]:
    """Convert a transcript timeline back to segments for legacy helpers."""
    return [
        Segment(start=cue.start, end=cue.end, text=cue.text, words=list(cue.words or []))
        for cue in timeline.cues
    ]
