"""Media job options and output path decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

from pycut.transcript_store import TranscriptStore
from pycut.video_io import DEFAULT_OUTPUT_FORMATS, _normalize_output_formats


@dataclass
class MediaJob:
    """One local media processing job."""

    video_path: str
    output_dir: str
    translate: bool = False
    source_lang: str = "en"
    target_lang: str = "en"
    orientation: str = "landscape"
    subtitle_position: str = "original-top"
    first_subtitle_delay: float = 1.0
    original_subtitle_color: str = "#FFFFFF"
    translation_subtitle_color: str = "#FFA500"
    filter_empty_segments: bool = True
    margin_left: float = -0.15
    margin_right: float = 0.15
    output_formats: Optional[Iterable[str]] = None
    export_fcpxml: bool = False
    fcpxml_frame_rate: float = 25.0
    fcpxml_speed: float = 1.0
    transcript_json_path: Optional[str] = None

    @property
    def video_name(self) -> str:
        return Path(self.video_path).stem

    def selected_formats(self) -> Set[str]:
        if self.output_formats is None:
            formats = {"fcpxml"} if self.export_fcpxml else set(DEFAULT_OUTPUT_FORMATS)
        else:
            formats = set(_normalize_output_formats(self.output_formats))
            if self.export_fcpxml:
                formats.add("fcpxml")
        return formats

    def transcript_store(self) -> TranscriptStore:
        return TranscriptStore(self.output_dir, self.video_name)

    def output_path(self, suffix: str) -> str:
        return str(Path(self.output_dir) / f"{self.video_name}{suffix}")
