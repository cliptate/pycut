#!/usr/bin/env python3
# coding=utf-8
"""
Video clipping CLI entry point.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pycut.config as config
from pycut.tts import synthesize_text_to_wav
from pycut.utils import normalize_hex_color
from pycut.video_io import (
    _parse_output_formats, _expand_video_inputs,
)

VideoClipper = None


def _get_video_clipper_class():
    global VideoClipper
    if VideoClipper is None:
        from pycut.clipper import VideoClipper as clipper_cls

        VideoClipper = clipper_cls
    return VideoClipper


def _resolve_default_asr_model(source_lang: str) -> str:
    return config.current_runtime_profile().default_asr_model(source_lang)


def _resolve_default_aligner_model() -> str:
    return config.current_runtime_profile().default_aligner_model()


def _resolve_default_tts_model() -> str:
    return config.current_runtime_profile().default_tts_model()


def _resolve_output_dir(video_path: str, explicit_output_dir: str | None) -> str:
    if explicit_output_dir:
        return explicit_output_dir

    video = Path(video_path).resolve()
    return str(video.parent / video.stem)


def _parse_hex_color(value: str) -> str:
    try:
        return normalize_hex_color(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_clip_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local media processing with ASR, translation, subtitles, FCPXML, and rendered video output.\n\n"
            "Examples:\n"
            "  Transcript: pycut --source-lang en --format json,srt ~/Movies/interview.mp4\n"
            "  Bilingual video: pycut --translate --source-lang en --target-lang zh --format video,ass --orientation portrait ~/Movies/video.mp4\n"
            "  Timeline: pycut --format fcpxml,json --fcpxml-speed 1.1 ~/Movies/video.mp4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_inputs", nargs="+", help="Video files, directories, or glob patterns")
    parser.add_argument(
        "--transcript",
        default=None,
        metavar="JSON_FILE",
        help="Path to existing transcript JSON file (skips ASR transcription)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: create a sibling folder named after each input file stem)",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=(
            "ASR model path "
            f"(default: en->{config.DEFAULT_EN_ASR_MODEL}, "
            f"zh->{config.DEFAULT_CHINESE_ASR_MODEL}, "
            f"other->{config.DEFAULT_FALLBACK_ASR_MODEL})"
        ),
    )
    parser.add_argument(
        "--aligner-model",
        default=None,
        help="Aligner model path (default: selected by current system)",
    )
    parser.add_argument(
        "--no-align",
        dest="enable_align",
        action="store_false",
        help="Disable word alignment and fall back to segment-level timestamps",
    )
    parser.add_argument("--segment-duration", type=int, default=300, help="Audio segment duration in seconds (default: 300)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum subtitle segment duration in seconds (default: 30.0)")
    parser.add_argument("--max-chars", type=int, default=30, help="Maximum characters per subtitle segment (default: 30)")
    parser.add_argument("--translate", action="store_true", help="Translate subtitles")
    parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    parser.add_argument("--target-lang", default="en", help="Target language code (default: en)")
    parser.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape", help="Video orientation (default: landscape)")
    parser.add_argument("--subtitle-position", choices=["original-top", "translated-top"], default="translated-top", help="Subtitle position: original-top (original above translated) or translated-top (default: translated-top)")
    parser.add_argument(
        "--original-subtitle-color",
        type=_parse_hex_color,
        default=config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        help=f"Original subtitle color in #RRGGBB format (default: {config.DEFAULT_ORIGINAL_SUBTITLE_COLOR})",
    )
    parser.add_argument(
        "--translation-subtitle-color",
        type=_parse_hex_color,
        default=config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        help=f"Translation subtitle color in #RRGGBB format (default: {config.DEFAULT_TRANSLATION_SUBTITLE_COLOR})",
    )
    parser.add_argument("--first-subtitle-delay", type=float, default=1.0, help="Delay in seconds for first subtitle screen (useful for cover frame) (default: 1.0)")
    parser.add_argument("--no-filter-empty-segments", dest="filter_empty_segments", action="store_false",
                        help="Keep empty transcript segments in subtitle/FCPXML export")
    parser.add_argument("--no-filter-fillers", dest="filter_fillers", action="store_false",
                        help="Disable filler-word filtering (e.g., um/uh) before subtitle segmentation")
    parser.add_argument("--margin-left", type=float, default=-100.0,
                        help="Extend each subtitle segment start by this many milliseconds (default: -100ms)")
    parser.add_argument("--margin-right", type=float, default=150.0,
                        help="Extend each subtitle segment end by this many milliseconds (default: 150ms)")
    parser.add_argument(
        "--format",
        default="srt",
        help="Comma-separated output formats: ass,srt,fcpxml,video,txt,json (default: srt)",
    )
    parser.add_argument("--fcpxml-frame-rate", type=float, default=25.0,
                        help="Frame rate for FCPXML export (default: 25.0)")
    parser.add_argument("--fcpxml-speed", type=float, default=1.0,
                        help="Timeline speed multiplier for FCPXML export (e.g. 1.1 = 1.1x) (default: 1.0)")
    return parser


def _build_tts_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycut tts",
        description="Generate speech WAV from text using the current system's TTS backend.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize")
    text_group.add_argument("--text-file", help="UTF-8 text file to synthesize")
    parser.add_argument("-o", "--output", required=True, help="Output WAV path")
    parser.add_argument(
        "--tts-model",
        default=None,
        help="TTS model path (default: selected by current system)",
    )
    parser.add_argument("--voice", default="Chelsie", help="MLX voice name (default: Chelsie)")
    parser.add_argument("--lang-code", default=None, help="MLX language hint, e.g. English or Chinese")
    parser.add_argument("--speed", type=float, default=None, help="MLX speed control when supported")
    parser.add_argument(
        "--split-pattern",
        "--split_pattern",
        default=None,
        help=r"MLX text split pattern for long scripts, e.g. \n for one segment per line",
    )
    parser.add_argument(
        "--max-tokens",
        "--max_tokens",
        type=int,
        default=None,
        help="MLX max generation tokens per segment",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable backend generation progress output")
    parser.add_argument("--device", default=None, help="VoxCPM device override, e.g. cuda, cpu, mps")
    parser.add_argument("--reference-audio", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--prompt-audio", default=None, help="Prompt/reference audio path")
    parser.add_argument("--prompt-text", default=None, help="Text corresponding to the prompt/reference audio")
    parser.add_argument("--cfg", type=float, default=2.0, help="VoxCPM CFG value (default: 2.0)")
    parser.add_argument("--steps", type=int, default=10, help="VoxCPM inference steps (default: 10)")
    parser.add_argument("--normalize", action="store_true", help="Enable backend text normalization")
    parser.add_argument(
        "--join-audio",
        "--join_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join MLX multi-segment output into one WAV (default: enabled)",
    )
    return parser


def _decode_tts_split_pattern(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\r", "\r")


def _run_tts(argv: list[str]):
    parser = _build_tts_parser()
    args = parser.parse_args(argv)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text or ""
    text = text.strip()
    if not text:
        parser.error("TTS text is empty")

    output_path = synthesize_text_to_wav(
        text=text,
        output_path=args.output,
        model_path=args.tts_model or _resolve_default_tts_model(),
        voice=args.voice,
        lang_code=args.lang_code,
        speed=args.speed,
        split_pattern=_decode_tts_split_pattern(args.split_pattern),
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        device=args.device,
        reference_audio=args.reference_audio,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
        normalize=args.normalize,
        join_audio=args.join_audio,
    )
    return {"tts": output_path}


def _run_clip(argv: list[str]):
    parser = _build_clip_parser()
    args = parser.parse_args(argv)
    try:
        output_formats = _parse_output_formats(args.format)
    except ValueError as exc:
        parser.error(str(exc))

    # Initialize clipper
    resolved_asr_model = args.asr_model or _resolve_default_asr_model(args.source_lang)
    resolved_aligner_model = args.aligner_model or _resolve_default_aligner_model()

    clipper_cls = _get_video_clipper_class()
    clipper = clipper_cls(
        asr_model_path=resolved_asr_model,
        aligner_model_path=resolved_aligner_model,
        enable_align=args.enable_align,
        segment_duration=args.segment_duration,
        max_duration=args.max_duration,
        max_chars=args.max_chars,
        filter_fillers=args.filter_fillers,
    )

    input_videos = _expand_video_inputs(args.video_inputs)
    if not input_videos:
        parser.error("No valid video files found in inputs")

    if args.transcript and len(input_videos) > 1:
        parser.error("--transcript can only be used with a single video input")

    all_results: Dict[str, Dict[str, str]] = {}
    for idx, video_path in enumerate(input_videos, start=1):
        print(f"\n▶️  [{idx}/{len(input_videos)}] {video_path}")
        resolved_output_dir = _resolve_output_dir(video_path, args.output_dir)
        all_results[video_path] = clipper.process_video(
            video_path=video_path,
            output_dir=resolved_output_dir,
            translate=args.translate,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            orientation=args.orientation,
            subtitle_position=args.subtitle_position,
            original_subtitle_color=args.original_subtitle_color,
            translation_subtitle_color=args.translation_subtitle_color,
            first_subtitle_delay=args.first_subtitle_delay,
            filter_empty_segments=args.filter_empty_segments,
            margin_left=args.margin_left / 1000.0,
            margin_right=args.margin_right / 1000.0,
            output_formats=output_formats,
            fcpxml_frame_rate=args.fcpxml_frame_rate,
            fcpxml_speed=args.fcpxml_speed,
            transcript_json_path=args.transcript,
        )

    if len(input_videos) == 1:
        return all_results[input_videos[0]]
    return all_results


def main(argv: list[str] | None = None):
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0] == "tts":
        return _run_tts(resolved_argv[1:])
    return _run_clip(resolved_argv)


def console_main(argv: list[str] | None = None) -> int:
    try:
        main(argv)
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(console_main())
