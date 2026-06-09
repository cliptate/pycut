"""ASS/SRT subtitle generation utilities."""

from __future__ import annotations

from typing import Callable, List, Optional

import pycut.config as config
from pycut.timeline import TranscriptTimeline
from pycut.utils import Segment, hex_color_to_ass


def extract_transcription_for_range(
    segments: List[Segment],
    start_time: float,
    end_time: float,
) -> str:
    """Extract transcription text for a specific time range."""
    texts = []
    for seg in segments:
        if seg.end > start_time and seg.start < end_time:
            texts.append(seg.text)
    return " ".join(texts)


def generate_ass_subtitle(
    timeline: TranscriptTimeline,
    output_path: str,
    translate: bool = False,
    source_lang: str = "zh",
    target_lang: str = "en",
    orientation: str = "landscape",
    subtitle_position: str = "original-top",
    first_subtitle_delay: float = 1.0,
    translate_fn: Optional[Callable[[List[str], str, str], List[str]]] = None,
    original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
    translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
) -> str:
    """Generate an ASS subtitle file with multi-layer support.

    Styles:
    - Title: Full highlight duration (top-center, cyan, bold)
    - Subtitle: Full highlight duration (top-center, white, bold)
    - FirstLine: Per-segment timing (white, larger font, bottom-center)
    - SecondLine: Per-segment timing (orange, smaller font, bottom-center)

    Args:
        translate_fn: Optional callable ``(texts, source_lang, target_lang) -> List[str]``
                      used to translate subtitle text.  Pass ``None`` to skip translation.
        first_subtitle_delay: Delay in seconds for the first screen subtitle.
    """
    print("📝 Generating ASS subtitle file...")
    print(f"  Translation: {'Enabled' if translate else 'Disabled'}")
    print(f"  Subtitle position: {subtitle_position}")
    if first_subtitle_delay > 0:
        print(f"  First subtitle delay: {first_subtitle_delay}s")

    original_ass_color = f"{hex_color_to_ass(original_subtitle_color)}&"
    translation_ass_color = f"{hex_color_to_ass(translation_subtitle_color)}&"

    if orientation == "portrait":
        ass_header = """[Script Info]
Title: Generated Subtitle
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Arial Unicode MS,140.0,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3.0,2,8,20,20,250,0
Style: Subtitle,Arial Unicode MS,100.0,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1.5,8,20,20,250,0
Style: OriginalTop,Arial Unicode MS,60.0,{original_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,520,0
Style: OriginalBottom,Arial Unicode MS,40.0,{original_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,460,0
Style: TranslationTop,Arial Unicode MS,60.0,{translation_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,520,0
Style: TranslationBottom,Arial Unicode MS,40.0,{translation_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,460,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            original_ass_color=original_ass_color,
            translation_ass_color=translation_ass_color,
        )
    else:
        ass_header = """[Script Info]
Title: Generated Subtitle
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Arial Unicode MS,100.0,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3.0,2,2,20,20,100,0
Style: Subtitle,Arial Unicode MS,70.0,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1.5,2,20,20,100,0
Style: OriginalTop,Arial Unicode MS,50.0,{original_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,240,0
Style: OriginalBottom,Arial Unicode MS,35.0,{original_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,180,0
Style: TranslationTop,Arial Unicode MS,50.0,{translation_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,240,0
Style: TranslationBottom,Arial Unicode MS,35.0,{translation_ass_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1.0,1,2,20,20,180,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            original_ass_color=original_ass_color,
            translation_ass_color=translation_ass_color,
        )

    def format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    events: List[str] = []
    cue_texts = [cue.text for cue in timeline.cues]
    translated_segments: List[str] = []
    if translate and cue_texts and translate_fn is not None:
        translated_segments = translate_fn(cue_texts, source_lang, target_lang)
        if len(translated_segments) != len(cue_texts):
            translated_segments = cue_texts

    total_duration = sum(cue.duration for cue in timeline.cues)
    if total_duration > 0 and (timeline.title or timeline.subtitle):
        timeline_end_fmt = format_time(total_duration)
        events.append(f"Dialogue: 0,00:00.00,{timeline_end_fmt},Title,,0,0,0,,{timeline.title}")
        events.append(f"Dialogue: 0,00:00.00,{timeline_end_fmt},Subtitle,,0,0,0,,{timeline.subtitle}")

    cumulative_time = 0.0
    for cue_index, cue in enumerate(timeline.cues):
        cue_start_time = cumulative_time
        if cue_index == 0 and first_subtitle_delay > 0:
            cue_start_time = max(cue_start_time, first_subtitle_delay)

        cue_end_time = cumulative_time + cue.duration
        seg_start = format_time(cue_start_time)
        seg_end = format_time(cue_end_time)

        if translate and translated_segments:
            translated_text = translated_segments[cue_index]
            if subtitle_position == "original-top":
                events.append(
                    f"Dialogue: 0,{seg_start},{seg_end},OriginalTop,,0,0,0,,{cue.text}"
                )
                events.append(
                    f"Dialogue: 0,{seg_start},{seg_end},TranslationBottom,,0,0,0,,{translated_text}"
                )
            else:
                events.append(
                    f"Dialogue: 0,{seg_start},{seg_end},TranslationTop,,0,0,0,,{translated_text}"
                )
                events.append(
                    f"Dialogue: 0,{seg_start},{seg_end},OriginalBottom,,0,0,0,,{cue.text}"
                )
        else:
            events.append(
                f"Dialogue: 0,{seg_start},{seg_end},OriginalTop,,0,0,0,,{cue.text}"
            )

        cumulative_time = cue_end_time

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        f.write("\n".join(events))

    print(f"✅ ASS subtitle saved to {output_path}")
    return output_path
