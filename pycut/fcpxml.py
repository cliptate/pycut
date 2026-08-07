"""FCPXML generation for Final Cut Pro and DaVinci Resolve."""

from __future__ import annotations

import datetime
import math
import subprocess
from copy import deepcopy
from fractions import Fraction
from html import escape
from pathlib import Path
from typing import Callable, List, Optional

from lxml import etree

import pycut.config as config
from pycut.timeline import TimelineCue, TranscriptTimeline
from pycut.utils import (
    get_audio_duration as _get_audio_duration,
    hex_color_to_fcpxml,
)

_TITLE_EFFECT_UID = ".../Titles.localized/Build In:Out.localized/Custom.localized/Custom.moti"


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


def _fcpxml_frame_rate(frame_rate: float) -> Fraction:
    if math.isclose(frame_rate, 23.976, rel_tol=0, abs_tol=0.001):
        return Fraction(24000, 1001)
    if math.isclose(frame_rate, 29.97, rel_tol=0, abs_tol=0.001):
        return Fraction(30000, 1001)
    return Fraction(str(frame_rate))


def _frame_time(frames: int, frame_rate: int | Fraction) -> str:
    rate = frame_rate if isinstance(frame_rate, Fraction) else Fraction(frame_rate)
    return f"{frames * rate.denominator}/{rate.numerator}s"


def _parse_time(value: str | None) -> Fraction:
    raw = (value or "0s").removesuffix("s")
    return Fraction(raw)


def _input_document_path(input_path: str) -> Path:
    path = Path(input_path)
    if path.suffix.lower() == ".fcpxmld":
        path = path / "Info.fcpxml"
    if not path.is_file():
        raise RuntimeError(f"FCPXML document not found: {path}")
    return path


def rough_cut_fcpxml(
    input_path: str,
    timeline: TranscriptTimeline,
    output_path: str,
    *,
    translate: bool = False,
    source_lang: str = "zh",
    target_lang: str = "en",
    orientation: str = "landscape",
    translate_fn: Optional[Callable[[List[str], str, str], List[str]]] = None,
    original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
    translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
) -> str:
    """Rewrite the primary story spine to contain only transcript cue ranges."""
    source_path = _input_document_path(input_path)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, remove_blank_text=True)
    try:
        tree = etree.parse(str(source_path), parser)
    except (OSError, etree.XMLSyntaxError) as exc:
        raise RuntimeError(f"Invalid FCPXML document: {exc}") from exc
    root = tree.getroot()
    if root.tag != "fcpxml":
        raise RuntimeError(f"Expected an FCPXML document, got <{root.tag}>")

    # ponytail: first project only; add project selection when multi-project imports are needed.
    spines = root.xpath(".//project/sequence/spine")
    if not spines:
        raise RuntimeError("FCPXML document has no project story spine")
    spine = spines[0]
    story = list(spine)
    if not story:
        raise RuntimeError("FCPXML story spine is empty")

    sequence = spine.getparent()
    format_id = sequence.get("format")
    formats = root.xpath(".//resources/format[@id=$format_id]", format_id=format_id) if format_id else []
    frame_duration_value = formats[0].get("frameDuration") if formats else None
    frame_duration = _parse_time(frame_duration_value or "1/25s")
    if frame_duration <= 0:
        raise RuntimeError("FCPXML frameDuration must be greater than zero")
    frame_rate = 1 / frame_duration

    active_cues = [cue for cue in timeline.cues if str(cue.text or "").strip()]
    if translate and active_cues and translate_fn is not None:
        print("🌍 Translating segments for FCPXML...")
        raw = translate_fn([cue.text for cue in active_cues], source_lang, target_lang)
        translations = raw if len(raw) == len(active_cues) else [""] * len(active_cues)
    else:
        translations = [""] * len(active_cues)
    translation_index = 0
    title_effect_ref = ""
    if active_cues:
        resources = root.find("resources")
        if resources is None:
            raise RuntimeError("FCPXML document has no resources")
        effects = resources.xpath("./effect[@uid=$uid]", uid=_TITLE_EFFECT_UID)
        if effects:
            title_effect_ref = effects[0].get("id") or ""
        else:
            used_ids = {element.get("id") for element in root.xpath(".//*[@id]")}
            resource_number = 1
            while f"r{resource_number}" in used_ids:
                resource_number += 1
            title_effect_ref = f"r{resource_number}"
            etree.SubElement(
                resources,
                "effect",
                id=title_effect_ref,
                name="Title",
                uid=_TITLE_EFFECT_UID,
            )

    def start_frame(seconds: Fraction) -> int:
        frames = seconds * frame_rate
        return frames.numerator // frames.denominator

    def end_frame(seconds: Fraction) -> int:
        frames = seconds * frame_rate
        return -(-frames.numerator // frames.denominator)

    story_ranges = []
    fallback_offset = Fraction(0)
    for element in story:
        offset = _parse_time(element.get("offset")) if element.get("offset") else fallback_offset
        duration = _parse_time(element.get("duration"))
        story_ranges.append((element, offset, offset + duration))
        fallback_offset = offset + duration

    for element in story:
        spine.remove(element)

    output_frame = 0
    style_id = 1
    for cue in timeline.cues:
        cue_has_text = bool(str(cue.text or "").strip())
        translation = translations[translation_index] if cue_has_text else ""
        if cue_has_text:
            translation_index += 1
        cue_start = Fraction(str(cue.start))
        cue_end = Fraction(str(cue.end))
        for element, element_start, element_end in story_ranges:
            keep_start = max(cue_start, element_start)
            keep_end = min(cue_end, element_end)
            if keep_end <= keep_start:
                continue
            source_start = _parse_time(element.get("start")) + (keep_start - element_start)
            source_start_frame = start_frame(source_start)
            duration_frames = end_frame(keep_end) - start_frame(keep_start)
            if duration_frames <= 0:
                continue
            fragment = deepcopy(element)
            fragment.set("offset", _frame_time(output_frame, frame_rate))
            if element.tag != "transition":
                fragment.set("start", _frame_time(source_start_frame, frame_rate))
            fragment.set("duration", _frame_time(duration_frames, frame_rate))
            if title_effect_ref and cue_has_text and element.tag != "transition":
                title_xml = _build_fcpxml_title_for_cue(
                    cue,
                    translation,
                    source_start_frame,
                    duration_frames,
                    frame_rate,
                    style_id,
                    orientation,
                    original_subtitle_color=original_subtitle_color,
                    translation_subtitle_color=translation_subtitle_color,
                    effect_ref=title_effect_ref,
                )
                fragment.append(etree.fromstring(title_xml.encode("utf-8")))
                style_id += 1
            spine.append(fragment)
            output_frame += duration_frames

    sequence.set("duration", _frame_time(output_frame, frame_rate))
    destination = Path(output_path)
    if destination.resolve() == source_path.resolve():
        raise RuntimeError("FCPXML output path must not overwrite the input document")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(destination),
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
        doctype=tree.docinfo.doctype or None,
    )
    return str(destination)


def build_fcpxml_timemap(
    start_f: int,
    timeline_dur_f: int,
    source_dur_f: int,
    fps_int: int | Fraction,
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
        f'                <timept time="{_frame_time(t0, fps_int)}" value="{_frame_time(t0, fps_int)}" interp="linear"/>\n'
        f'                <timept time="{_frame_time(t1, fps_int)}" value="{_frame_time(v1, fps_int)}" interp="linear"/>\n'
        f"              </timeMap>"
    )


def _build_fcpxml_title_for_cue(
    cue: TimelineCue,
    translation: str,
    offset_frames: int,
    duration_frames: int,
    fps_int: int | Fraction,
    style_id: int,
    orientation: str,
    original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
    translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
    effect_ref: str = "r3",
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
        f'              <title ref="{xml_attr(effect_ref)}" name="{xml_attr(name_attr)}" lane="1"'
        f' offset="{_frame_time(offset_frames, fps_int)}"'
        f' duration="{_frame_time(duration_frames, fps_int)}">',
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
    fps = _fcpxml_frame_rate(frame_rate)
    effective_frame_rate = float(fps)
    timeline_speed = float(speed)

    def s2f(seconds: float) -> int:
        return int(math.ceil(round(seconds * effective_frame_rate, 9)))

    def s2f_start(seconds: float) -> int:
        return max(0, int(math.floor(round(seconds * effective_frame_rate, 9))))

    def s2f_end(seconds: float) -> int:
        return max(0, int(math.ceil(round(seconds * effective_frame_rate, 9))))

    def s2f_timeline(seconds: float) -> int:
        return int(math.ceil(round(seconds * effective_frame_rate / timeline_speed, 9)))

    def ft(n: int) -> str:
        return _frame_time(n, fps)

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
        f'    <format id="r1" name="CustomFormat_{width}x{height}_{frame_rate:g}fps"'
        f' frameDuration="{ft(1)}" width="{width}" height="{height}"'
        f' colorSpace="1-1-1 (Rec. 709)"/>',
        f'    <asset id="r2" name="{escape(video_name, quote=True)}"'
        f' start="{ft(0)}" hasVideo="1" format="r1" hasAudio="1"'
        f' audioChannels="2" duration="{ft(video_src_dur_f)}">',
        f'      <media-rep kind="original-media" src="{escape(video_url, quote=True)}"/>',
        "    </asset>",
        f'    <effect id="r3" name="Title" uid="{_TITLE_EFFECT_UID}"/>',
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
            clip_lines.append(build_fcpxml_timemap(start_f, dur_f, dur_f_src, fps))
        clip_lines += [
            _build_fcpxml_title_for_cue(
                cue,
                translation,
                start_f,
                dur_f,
                fps,
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
