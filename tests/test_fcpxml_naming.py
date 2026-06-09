#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


from pycut.clipper import VideoClipper
from pycut.timeline import TimelineCue, TranscriptTimeline
from pycut.utils import Segment


def _timeline_from_segments(segments, title="", subtitle=""):
    return TranscriptTimeline(
        cues=[
            TimelineCue(start=seg.start, end=seg.end, text=seg.text, words=list(seg.words or []))
            for seg in segments
        ],
        title=title,
        subtitle=subtitle,
    )


def test_generate_fcpxml_uses_source_filename_and_timestamped_event(tmp_path):
    clipper = VideoClipper()
    output_path = tmp_path / "output.fcpxml"
    video_path = tmp_path / "demo_video.mp4"
    segments = [
        Segment(start=0.0, end=1.0, text="hello"),
        Segment(start=1.0, end=2.0, text="world"),
    ]

    clipper.generate_fcpxml(
        video_path=str(video_path),
        timeline=_timeline_from_segments(segments),
        output_path=str(output_path),
    )

    content = output_path.read_text(encoding="utf-8")
    assert '<project name="demo_video">' in content
    assert re.search(r'<event name="\d{4}-\d{2}-\d{2}">', content)


def test_cli_help_does_not_expose_fcpxml_project_name_option():
    result = subprocess.run(
        [sys.executable, "-m", "pycut", "-h"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert "--fcpxml-project-name" not in result.stdout


def test_build_fcpxml_timemap_contains_correct_time_points():
    import pycut.fcpxml as fcpxml
    result = fcpxml.build_fcpxml_timemap(start_f=0, timeline_dur_f=25, source_dur_f=30, fps_int=25)
    assert 'time="0/25s"' in result
    assert 'time="25/25s"' in result
    assert 'value="30/25s"' in result


def _parse_title_runs(output_path):
    root = ET.fromstring(output_path.read_text(encoding="utf-8"))
    title = root.find(".//title")
    assert title is not None

    text_element = title.find("text")
    assert text_element is not None

    style_defs = {}
    for style_def in title.findall("text-style-def"):
        style = style_def.find("text-style")
        assert style is not None
        style_defs[style_def.attrib["id"]] = style.attrib

    runs = []
    for text_style in text_element.findall("text-style"):
        runs.append(
            {
                "text": text_style.text or "",
                "style": style_defs.get(text_style.attrib.get("ref", ""), {}),
            }
        )

    return runs


def test_generate_fcpxml_full_video_uses_original_segment_text_after_filtering(tmp_path):
    clipper = VideoClipper()
    output_path = tmp_path / "output.fcpxml"
    video_path = tmp_path / "demo_video.mp4"
    segments = [
        Segment(start=0.0, end=1.0, text=""),
        Segment(start=1.0, end=3.0, text="hello world"),
    ]

    clipper.generate_fcpxml(
        video_path=str(video_path),
        timeline=_timeline_from_segments([segments[1]]),
        output_path=str(output_path),
    )

    runs = _parse_title_runs(output_path)
    assert "".join(run["text"] for run in runs) == "hello world"


def test_generate_fcpxml_clip_mode_writes_segment_text(tmp_path):
    clipper = VideoClipper()
    output_path = tmp_path / "output.fcpxml"
    video_path = tmp_path / "demo_video.mp4"
    segments = [
        Segment(start=0.0, end=2.0, text="hello world"),
    ]

    clipper.generate_fcpxml(
        video_path=str(video_path),
        timeline=_timeline_from_segments(segments),
        output_path=str(output_path),
    )

    runs = _parse_title_runs(output_path)
    assert "".join(run["text"] for run in runs) == "hello world"


def test_generate_fcpxml_escapes_xml_special_characters_in_clip_titles(tmp_path):
    import pycut.fcpxml as fcpxml

    output_path = tmp_path / "output.fcpxml"
    video_path = tmp_path / 'demo "quoted" & clip.mp4'
    source_text = 'Say "hi" & <world> > friends'
    translation_text = '译文 "1 < 2" & friends'
    segments = [
        Segment(start=0.0, end=2.0, text=source_text),
    ]

    fcpxml.generate_fcpxml(
        video_path=str(video_path),
        timeline=_timeline_from_segments(segments, title=source_text, subtitle=translation_text),
        output_path=str(output_path),
        translate=True,
        translate_fn=lambda texts, _source, _target: [translation_text for _ in texts],
    )

    content = output_path.read_text(encoding="utf-8")
    root = ET.fromstring(content)

    asset_clip = root.find(".//asset-clip")
    assert asset_clip is not None
    assert asset_clip.attrib["name"] == source_text[:40]

    title = root.find(".//title")
    assert title is not None
    assert title.attrib["name"] == source_text[:50]

    runs = _parse_title_runs(output_path)
    assert "".join(run["text"] for run in runs) == f'{source_text}\n{translation_text}'

    assert 'name="Say &quot;hi&quot; &amp; &lt;world&gt; &gt; friends"' in content
    assert '&quot;hi&quot; &amp; &lt;world&gt;' in content
    assert '译文 "1 &lt; 2" &amp; friends' in content


def test_generate_fcpxml_uses_configured_original_and_translation_colors(tmp_path):
    import pycut.fcpxml as fcpxml

    output_path = tmp_path / "output.fcpxml"
    video_path = tmp_path / "demo_video.mp4"
    segments = [
        Segment(start=0.0, end=2.0, text="hello world"),
    ]

    fcpxml.generate_fcpxml(
        video_path=str(video_path),
        timeline=_timeline_from_segments(segments, title="hello"),
        output_path=str(output_path),
        translate=True,
        translate_fn=lambda texts, _source, _target: [f"tr:{text}" for text in texts],
        original_subtitle_color="#123456",
        translation_subtitle_color="#ABCDEF",
    )

    runs = _parse_title_runs(output_path)
    content = output_path.read_text(encoding="utf-8")

    assert "".join(run["text"] for run in runs) == "hello world\ntr:hello world"
    assert 'fontColor="0.0706 0.2039 0.3373 1"' in content
    assert 'fontColor="0.6706 0.8039 0.9373 1"' in content
