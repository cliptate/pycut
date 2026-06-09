"""FCPXML generation for Final Cut Pro and DaVinci Resolve."""

from __future__ import annotations

import datetime
import math
import subprocess
from html import escape
from pathlib import Path
from typing import Callable, List, Optional

import pycut.config as config
from pycut.timeline import TimelineCue, TranscriptTimeline
from pycut.utils import (
    get_audio_duration as _get_audio_duration,
    hex_color_to_fcpxml,
)


def get_video_info(
    video_path: str,
    get_duration_fn: Callable[[str], float] = _get_audio_duration,
) -> tuple[int, int, float]:
    """Return ``(width, height, duration)`` for the given video file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split("x"))
    duration = get_duration_fn(video_path)
    return width, height, duration


def build_fcpxml_timemap(
    start_f: int,
    timeline_dur_f: int,
    source_dur_f: int,
    fps_int: int,
) -> str:
    """Return an FCPXML ``<timeMap>`` element for constant-speed retiming.

    Maps clip local time [start_f, start_f+timeline_dur_f] to source time
    [start_f, start_f+source_dur_f], producing a speed of
    source_dur_f / timeline_dur_f relative to normal speed.
    """
    t0 = start_f
    t1 = start_f + timeline_dur_f
    v1 = start_f + source_dur_f
    return (
        f"              <timeMap>\n"
        f'                <timept time="{t0}/{fps_int}s" value="{t0}/{fps_int}s" interp="linear"/>\n'
        f'                <timept time="{t1}/{fps_int}s" value="{v1}/{fps_int}s" interp="linear"/>\n'
        f"              </timeMap>"
    )


def _build_fcpxml_title_for_cue(
    cue: TimelineCue,
    translation: str,
    offset_frames: int,
    duration_frames: int,
    fps_int: int,
    style_id: int,
    orientation: str,
    original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
    translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
) -> str:
    """Return an FCPXML ``<title>`` element string for one transcript cue."""

    def xml_attr(value: str) -> str:
        return escape(value, quote=True)

    def xml_text(value: str) -> str:
        return escape(value, quote=False)

    font_size = 60 if orientation == "landscape" else 48
    trans_font_size = 38 if orientation == "landscape" else 25
    vertical_pos = -33 if orientation == "landscape" else -13
    original_color = hex_color_to_fcpxml(original_subtitle_color)
    translation_color = hex_color_to_fcpxml(translation_subtitle_color)
    name_attr = (cue.text[:50] if cue.text else f"s{style_id}") or f"s{style_id}"
    lines = [
        f'              <title ref="r3" name="{xml_attr(name_attr)}" lane="1"'
        f' offset="{offset_frames}/{fps_int}s"'
        f' duration="{duration_frames}/{fps_int}s">',
        "                <text>",
        f'                  <text-style ref="ts{style_id}">{xml_text(cue.text)}</text-style>',
    ]
    if translation:
        lines += [
            "                  <text-style>&#xA;</text-style>",
            f'                  <text-style ref="ts{style_id}_t">{xml_text(translation)}</text-style>',
        ]
    lines += [
        "                </text>",
        f'                <text-style-def id="ts{style_id}">',
        f'                  <text-style font="Arial Unicode MS" fontSize="{font_size:g}"'
        f' fontFace="Regular" fontColor="{original_color}" bold="1" italic="0"'
        f' strokeColor="0 0 0 1" strokeWidth="-1"'
        f' shadowColor="0 0 0 0.5" shadowOffset="2 315" alignment="center"/>',
        "                </text-style-def>",
    ]
    if translation:
        lines += [
            f'                <text-style-def id="ts{style_id}_t">',
            f'                  <text-style font="Arial Unicode MS" fontSize="{trans_font_size}"'
            f' fontFace="Regular" fontColor="{translation_color}" bold="0" italic="0"'
            f' strokeColor="0 0 0 1" strokeWidth="-1"'
            f' shadowColor="0 0 0 0.5" shadowOffset="2 315" alignment="center"/>',
            "                </text-style-def>",
        ]
    lines += [
        f'                <adjust-transform position="0 {vertical_pos}"/>',
        "              </title>",
    ]
    return "\n".join(lines)


def generate_fcpxml(
    video_path: str,
    timeline: TranscriptTimeline,
    output_path: str,
    frame_rate: float = 25.0,
    speed: float = 1.0,
    translate: bool = False,
    source_lang: str = "zh",
    target_lang: str = "en",
    orientation: str = "landscape",
    translate_fn: Optional[Callable[[List[str], str, str], List[str]]] = None,
    original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
    translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
) -> str:
    """Generate an FCPXML file for Final Cut Pro or DaVinci Resolve.

    The FCPXML timeline is built from prepared transcript cues.
    """
    print(
        f"📋 Generating FCPXML: {output_path}, orientation: {orientation}, "
        f"frame_rate: {frame_rate}, speed: {speed}"
    )
    if speed <= 0:
        raise ValueError("FCPXML speed must be greater than 0")
    fps_int = int(frame_rate)
    timeline_speed = float(speed)

    def s2f(seconds: float) -> int:
        return int(math.ceil(round(seconds * frame_rate, 9)))

    def s2f_start(seconds: float) -> int:
        return max(0, int(math.floor(round(seconds * frame_rate, 9))))

    def s2f_end(seconds: float) -> int:
        return max(0, int(math.ceil(round(seconds * frame_rate, 9))))

    def s2f_timeline(seconds: float) -> int:
        return int(math.ceil(round(seconds * frame_rate / timeline_speed, 9)))

    def ft(n: int) -> str:
        return f"{n}/{fps_int}s"

    width, height = (1920, 1080) if orientation == "landscape" else (1080, 1920)
    active = [cue for cue in timeline.cues if str(cue.text or "").strip()]
    video_duration = timeline.end if timeline.cues else 0.0

    video_url = Path(video_path).resolve().as_uri()
    video_name = Path(video_path).stem
    project_name = video_name
    export_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    video_src_dur_f = s2f(video_duration)
    video_dur_f = s2f_timeline(video_duration)

    if translate and active and translate_fn is not None:
        print("🌍 Translating segments for FCPXML...")
        raw = translate_fn([s.text for s in active], source_lang, target_lang)
        trans_list: List[str] = raw if len(raw) == len(active) else [""] * len(active)
    else:
        trans_list = [""] * len(active)

    total_f = 0
    for cue in active:
        dur_f_src = s2f_end(cue.end) - s2f_start(cue.start)
        if dur_f_src <= 0:
            continue
        total_f += max(1, int(math.ceil(dur_f_src / timeline_speed)))
    if total_f <= 0:
        total_f = video_dur_f

    buf: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        '<fcpxml version="1.11">',
        "  <resources>",
        f'    <format id="r1" name="CustomFormat_{width}x{height}_{fps_int}fps"'
        f' frameDuration="1/{fps_int}s" width="{width}" height="{height}"'
        f' colorSpace="1-1-1 (Rec. 709)"/>',
        f'    <asset id="r2" name="{escape(video_name, quote=True)}"'
        f' start="0/{fps_int}s" hasVideo="1" format="r1" hasAudio="1"'
        f' audioChannels="2" duration="{ft(video_src_dur_f)}">',
        f'      <media-rep kind="original-media" src="{escape(video_url, quote=True)}"/>',
        "    </asset>",
        '    <effect id="r3" name="Title"'
        ' uid=".../Titles.localized/Build In:Out.localized/Custom.localized/Custom.moti"/>',
        "  </resources>",
        "  <library>",
        f'    <event name="{escape(export_timestamp, quote=True)}">',
        f'      <project name="{escape(project_name, quote=True)}">',
        f'        <sequence format="r1" tcFormat="NDF" audioLayout="stereo" audioRate="48k"'
        f' duration="{ft(total_f)}">',
        "          <spine>",
    ]

    style_id = 1
    timeline_off = 0
    for i, cue in enumerate(active):
        start_f = s2f_start(cue.start)
        end_f = s2f_end(cue.end)
        dur_f_src = end_f - start_f
        if dur_f_src <= 0:
            continue
        dur_f = max(1, int(math.ceil(dur_f_src / timeline_speed)))
        translation = trans_list[i] if i < len(trans_list) else ""
        clip_lines = [
            f'            <asset-clip ref="r2" offset="{ft(timeline_off)}"'
            f' duration="{ft(dur_f)}" start="{ft(start_f)}"'
            f' name="{escape((cue.text[:40] or str(i)), quote=True)}" tcFormat="NDF">',
        ]
        if timeline_speed != 1.0:
            clip_lines.append(build_fcpxml_timemap(start_f, dur_f, dur_f_src, fps_int))
        clip_lines += [
            _build_fcpxml_title_for_cue(
                cue,
                translation,
                start_f,
                dur_f,
                fps_int,
                style_id,
                orientation,
                original_subtitle_color=original_subtitle_color,
                translation_subtitle_color=translation_subtitle_color,
            ),
            "            </asset-clip>",
        ]
        buf += clip_lines
        style_id += 1
        timeline_off += dur_f

    buf += [
        "          </spine>",
        "        </sequence>",
        "      </project>",
        "    </event>",
        "  </library>",
        "</fcpxml>",
    ]

    content = "\n".join(buf) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ FCPXML saved to {output_path}")
    return output_path
