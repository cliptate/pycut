"""Video rendering utilities using ffmpeg."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Optional

from pycut.timeline import TranscriptTimeline


def select_video_encoder() -> str:
    """Select ffmpeg H.264 encoder by platform and hardware."""
    if platform.system().lower() == "darwin":
        return "h264_videotoolbox"
    return "libx264"


def select_ffmpeg_binary() -> str:
    """Return the ffmpeg binary that has libass (ass filter) support.

    Prefers ffmpeg-full (Homebrew formula with all optional deps) over the
    standard ffmpeg bottle which is built without --enable-libass.
    """
    candidates = [
        "/opt/homebrew/Cellar/ffmpeg-full/8.1/bin/ffmpeg",
        "ffmpeg-full",
        "ffmpeg",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-filters"],
                capture_output=True, text=True,
            )
            if " ass " in result.stdout or "subtitles" in result.stdout:
                return candidate
        except FileNotFoundError:
            continue
    return "ffmpeg"


def render_video_with_subtitles_complex(
    video_path: str,
    timeline: TranscriptTimeline,
    subtitle_path: str,
    output_path: str,
    orientation: str = "landscape",
    target_resolution: Optional[str] = None,
) -> str:
    """All-in-one video processing using filter_complex.

    Steps: extract/concatenate segments, handle orientation (pad with black bars),
    scale to target resolution, burn in subtitles.
    """
    print("🎬 Rendering video with filter_complex (all-in-one)...")

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split("x"))
    print(f"  📐 Input video dimensions: {width}x{height}")

    filter_parts = []
    cues = timeline.cues
    if not cues:
        raise ValueError("Cannot render video without timeline cues")

    segments = []
    for i, cue in enumerate(cues):
        duration = cue.end - cue.start
        segment = (
            f"[0:v]trim=start={cue.start}:duration={duration},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={cue.start}:duration={duration},asetpts=PTS-STARTPTS[a{i}];"
        )
        segments.append(segment)
    filter_parts.append("".join(segments))

    audio_concat = "".join([f"[a{i}]" for i in range(len(cues))])
    audio_concat += f"concat=n={len(cues)}:v=0:a=1[outa];"
    filter_parts.append(audio_concat)

    if len(cues) == 1:
        filter_parts.append("[v0]copy[concat_v];")
    else:
        video_concat = "".join([f"[v{i}]" for i in range(len(cues))])
        video_concat += f"concat=n={len(cues)}:v=1:a=0[concat_v];"
        filter_parts.append(video_concat)

    video_filters = []

    if orientation == "portrait" and width > height:
        print("  🔄 Converting to portrait (padding black bars)...")
        target_height = int(width * 16 / 9)
        pad_y = (target_height - height) // 2
        video_filters.append(f"pad={width}:{target_height}:0:{pad_y}:black")
    elif orientation == "landscape" and height > width:
        print("  🔄 Converting to landscape (padding black bars)...")
        target_width = int(height * 16 / 9)
        pad_x = (target_width - width) // 2
        video_filters.append(f"pad={target_width}:{height}:{pad_x}:0:black")

    if target_resolution:
        print(f"  📏 Scaling to {target_resolution}...")
        if target_resolution.endswith("p"):
            resolution_num = target_resolution[:-1]
            scale_filter = f"scale={resolution_num}:-1"
        else:
            scale_filter = f"scale={target_resolution}"
        video_filters.append(scale_filter)

    # Copy subtitle to a safe ASCII filename to avoid ffmpeg filter_complex
    # escaping issues with brackets, unicode, and spaces in file paths.
    subtitle_dir = os.path.dirname(subtitle_path)
    safe_subtitle_path = os.path.join(subtitle_dir, "safe_sub.ass")
    shutil.copy2(subtitle_path, safe_subtitle_path)
    # Use explicit `filename=` key — positional single-quoted syntax confuses ffmpeg's option parser.
    # The safe path contains only ASCII chars so no quoting is needed.
    video_filters.append(f"ass=filename={safe_subtitle_path}")

    if video_filters:
        video_filter_chain = ",".join(video_filters)
        filter_parts.append(f"[concat_v]{video_filter_chain}[vout];")
    else:
        filter_parts.append("[concat_v]copy[vout];")

    filter_complex = "".join(filter_parts)

    video_encoder = select_video_encoder()
    ffmpeg_bin = select_ffmpeg_binary()
    cmd = [
        ffmpeg_bin, "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[outa]",
        "-c:v", video_encoder,
    ]
    if video_encoder != "h264_videotoolbox":
        cmd.extend(["-profile:v", "main"])
    cmd.extend([
        "-b:v", "6000k",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "9999",
        "-hide_banner",
        "-loglevel", "error",
        "-stats",
        "-y", output_path,
    ])

    print(f"  🎬 Executing ffmpeg filter_complex with encoder: {video_encoder}, binary: {ffmpeg_bin}")
    print(f"  🔍 filter_complex: {filter_complex}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg failed (exit {result.returncode}):")
        if result.stderr:
            print(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    print(f"✅ Final video saved to {output_path}")

    return output_path
