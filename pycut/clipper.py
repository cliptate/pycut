"""VideoClipper — main video clipping pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

import pycut.analysis as analysis
import pycut.config as config
import pycut.fcpxml as fcpxml_mod
import pycut.renderer as renderer_mod
import pycut.subtitle as subtitle_mod
from pycut.asr import MLXASRHelper, QwenASRHelper
from pycut.models import Highlight
from pycut.translation import GoogleTranslator
from pycut.utils import (
    Segment,
    _segments_to_srt,
    extract_audio,
    get_audio_duration,
)
from dataclasses import replace
from pycut.video_io import (
    DEFAULT_OUTPUT_FORMATS,
    _load_segments_from_transcript_json,
    _normalize_output_formats,
    _parse_output_formats,
)


class VideoClipper:
    """Video clipping pipeline with ASR, analysis, translation, and subtitle rendering."""
    
    def __init__(
        self,
        asr_model_path: Optional[str] = None,
        aligner_model_path: Optional[str] = None,
        enable_align: bool = True,
        gemini_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        segment_duration: int = 300,  # 5 minutes
        max_duration: float = 30.0,
        max_chars: int = 30,
        filter_fillers: bool = True,
        translator: Optional[GoogleTranslator] = None,
    ):
        config.ensure_supported_runtime()
        self.segment_duration = segment_duration
        self.asr_backend = config.select_asr_backend()
        default_asr_model = (
            config.DEFAULT_EN_ASR_MODEL if self.asr_backend == "mlx" else config.resolve_default_qwen_asr_model()
        )
        default_aligner_model = (
            config.DEFAULT_ALIGNER_MODEL if self.asr_backend == "mlx" else config.resolve_default_qwen_aligner_model()
        )
        self.asr_model_path = asr_model_path or default_asr_model
        self.aligner_model_path = aligner_model_path or default_aligner_model
        self.enable_align = enable_align
        self.translator = translator or GoogleTranslator()
        self.max_duration = max_duration
        self.max_chars = max_chars
        self.filter_fillers = filter_fillers
        helper_cls = MLXASRHelper if self.asr_backend == "mlx" else QwenASRHelper
        self.asr_helper = helper_cls(
            asr_model_path=self.asr_model_path,
            aligner_model_path=self.aligner_model_path,
            filter_fillers=self.filter_fillers,
            enable_align=self.enable_align,
        )
        
        backend_info = "MLX (Apple Silicon)" if self.asr_backend == "mlx" else "Qwen3-ASR"
        print(f"🚀 Initializing VideoClipper with {backend_info} backend")
        print(f"   Models will be loaded on demand to save memory")
        
        # Configure LLM client – delegate to analysis helper
        # Support both new api_key and legacy gemini_api_key param
        resolved_api_key = api_key or gemini_api_key
        self.llm_client = analysis.create_client(resolved_api_key, base_url, model)
        # Legacy aliases for backward compatibility in tests
        self.gemini_model = self.llm_client
        self.gemini_client = self.llm_client
        if self.llm_client is not None:
            print("✅ LLM API configured!")
        elif not analysis.OPENAI_AVAILABLE:
            print("⚠️  openai not installed (pip install openai)")
            print("   Content analysis will be skipped")
        else:
            print("⚠️  API key not provided, skipping content analysis")
    
    def _get_asr_helper(self) -> MLXASRHelper:
        helper = getattr(self, "asr_helper", None)
        if helper is None:
            backend = getattr(self, "asr_backend", config.select_asr_backend())
            helper_cls = MLXASRHelper if backend == "mlx" else QwenASRHelper
            default_asr_model = config.DEFAULT_EN_ASR_MODEL if backend == "mlx" else config.resolve_default_qwen_asr_model()
            default_aligner_model = config.DEFAULT_ALIGNER_MODEL if backend == "mlx" else config.resolve_default_qwen_aligner_model()
            helper = helper_cls(
                asr_model_path=getattr(self, "asr_model_path", default_asr_model),
                aligner_model_path=getattr(self, "aligner_model_path", default_aligner_model),
                filter_fillers=getattr(self, "filter_fillers", True),
                enable_align=getattr(self, "enable_align", True),
            )
            if hasattr(self, "asr_model"):
                helper.asr_model = self.asr_model
            if hasattr(self, "_mlx_aligner"):
                helper._mlx_aligner = self._mlx_aligner
            if hasattr(self, "vad_model"):
                helper.vad_model = self.vad_model
            self.asr_helper = helper
        return helper

    def _load_asr_model(self):
        """Load the MLX ASR model on demand."""
        self._get_asr_helper().load_models()
    
    def _unload_asr_model(self):
        """Unload ASR model to free memory."""
        self._get_asr_helper().unload_models()
    
    def _load_vad_model(self):
        """Load Silero VAD model on demand."""
        self._get_asr_helper().load_vad_model()

    def _get_gemini_client(self):
        client = getattr(self, "llm_client", None)
        if client is not None:
            return client
        client = getattr(self, "gemini_model", None)
        if client is not None:
            return client
        return getattr(self, "gemini_client", None)
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video as WAV 16kHz mono."""
        return extract_audio(video_path, output_path)

    def _select_video_encoder(self) -> str:
        return renderer_mod.select_video_encoder()

    def render_video_with_subtitles_complex(
        self,
        video_path: str,
        highlights: List[Highlight],
        subtitle_path: str,
        output_path: str,
        orientation: str = "landscape",
        target_resolution: Optional[str] = None,
    ) -> str:
        return renderer_mod.render_video_with_subtitles_complex(
            video_path=video_path,
            highlights=highlights,
            subtitle_path=subtitle_path,
            output_path=output_path,
            orientation=orientation,
            target_resolution=target_resolution,
        )

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        return get_audio_duration(audio_path)
    
    def split_audio(self, audio_path: str, output_dir: str) -> List[Tuple[str, float, float]]:
        """Split audio into segments for processing."""
        duration = self.get_audio_duration(audio_path)
        print(f"📊 Audio duration: {duration:.2f}s")
        
        segments = []
        num_segments = int(np.ceil(duration / self.segment_duration)) if np is not None else int(math.ceil(duration / self.segment_duration))
        
        print(f"✂️  Splitting audio into {num_segments} segments...")
        for i in range(num_segments):
            start_time = i * self.segment_duration
            end_time = min((i + 1) * self.segment_duration, duration)
            segment_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
            
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-acodec", "copy",
                "-y", segment_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            segments.append((segment_path, start_time, end_time))
            print(f"  ✅ Segment {i+1}/{num_segments}: {start_time:.2f}s - {end_time:.2f}s")
        
        return segments

    def split_transcript_segments(
        self,
        segments: List[Segment],
        max_duration: float,
    ) -> List[List[Segment]]:
        """Split transcription segments into chunks with max duration."""
        if not segments:
            return []

        chunks: List[List[Segment]] = []
        current: List[Segment] = []
        chunk_start = None

        for seg in segments:
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

    @staticmethod
    def _resolve_overlaps(segments: List[Segment], margin_left: float = 0.0, margin_right: float = 0.0) -> List[Segment]:
        """Apply per-segment margin offsets then resolve any overlaps using midpoint splitting."""
        if not segments:
            return []

        # Apply margins first
        shifted = [
            replace(seg, start=max(0.0, seg.start + margin_left), end=max(0.0, seg.end + margin_right))
            for seg in segments
        ]

        # Resolve overlaps between adjacent segments
        resolved = []
        overlap_count = 0
        for i, seg in enumerate(shifted):
            start = seg.start
            end = seg.end
            if resolved:
                prev = resolved[-1]
                if start < prev.end:
                    mid = (prev.end + start) / 2
                    resolved[-1] = replace(prev, end=mid)
                    start = mid
                    overlap_count += 1
            if i + 1 < len(shifted):
                next_seg = shifted[i + 1]
                if end > next_seg.start:
                    end = (end + next_seg.start) / 2
            resolved.append(replace(seg, start=start, end=end))
        if overlap_count > 0:
            print(f"🔧 Resolved {overlap_count} overlapping segment(s) using midpoint split")
        return resolved

    def _correct_transcript(
        self,
        segments: List,
        source_lang: str,
    ) -> List:
        """
        Use Gemini to fix ASR errors in segment text. Prints a diff of every
        change and returns the updated segment list.
        """
        from pycut import analysis as _analysis
        from dataclasses import replace

        client = self._get_gemini_client()
        if client is None:
            print("⚠️  --correct-words requires an API key (--api-key or OPENAI_API_KEY). Skipping.")
            return segments

        print("🔍 Correcting ASR errors with Gemini…")
        corrections = _analysis.correct_words(client, segments, source_lang)

        if not corrections:
            print("✅ No ASR corrections needed.")
            return segments

        correction_map = {c["segment_id"]: c["corrected"] for c in corrections}
        updated = []
        for i, seg in enumerate(segments):
            if i in correction_map:
                original = seg.text
                corrected = correction_map[i]
                print(f"  📝 [{seg.start:.2f}s-{seg.end:.2f}s] {original!r} → {corrected!r}")
                seg = replace(seg, text=corrected)
            updated.append(seg)

        print(f"✅ Applied {len(corrections)} correction(s).")
        return updated

    def _filter_subtitle_segments(
        self,
        segments: List[Segment],
        filter_empty_segments: bool = True,
    ) -> List[Segment]:
        """Filter empty segments and resolve any overlaps."""
        if filter_empty_segments:
            filtered = [seg for seg in segments if str(getattr(seg, "text", "") or "").strip()]
            removed = len(segments) - len(filtered)
            if removed > 0:
                print(f"🧹 Filtered {removed} empty subtitle segment(s)")
        else:
            filtered = list(segments)
        return self._resolve_overlaps(filtered)
    
    def transcribe_audio(self, audio_path: str, orientation: str = "landscape", source_lang: str = "en") -> List[Segment]:
        """Transcribe audio file with word-level timestamps."""
        print(f"🎤 Transcribing {audio_path}...")
        
        max_chars = self.max_chars
        print(f"  Using max_chars={max_chars} for {orientation} mode")
        
        # The ASR helper owns model lifetimes: VAD, ASR, and aligner are loaded
        # and released as separate phases to keep peak memory low.
        return self._transcribe_with_vad(audio_path, max_chars=max_chars, source_lang=source_lang)
    
    def _transcribe_with_vad(
        self,
        audio_path: str,
        time_offset: float = 0.0,
        max_chars: int = 60,
        source_lang: str = "en",
    ) -> List[Segment]:
        """Transcribe audio using VAD to detect speech segments first, then ASR each."""
        return self._get_asr_helper().transcribe_with_vad(
            audio_path,
            time_offset=time_offset,
            max_chars=max_chars,
            source_lang=source_lang,
            get_audio_duration=self.get_audio_duration,
        )

    def analyze_with_gemini_highlights(self, segments: List[Segment], source_lang, target_lang: str) -> List[Highlight]:
        """
        Second stage: Extract video highlights using Gemini with detailed keyword extraction.

        Delegates API call and JSON parsing to analysis.extract_highlights and
        maps the returned dicts into Highlight dataclass instances.

        Args:
            segments: List of transcription segments
            source_lang: Source language code (zh, en, ja, etc.)
            target_lang: Target language for title/subtitle

        Returns:
            List of Highlight objects with keywords for highlighting
        """
        gemini_model = self._get_gemini_client()
        if not gemini_model:
            print("⚠️  LLM API not configured, skipping highlights extraction")
            return []

        print(f"🤖 Extracting highlights with LLM (language: {source_lang})...")

        raw = analysis.extract_highlights(
            gemini_model, segments, source_lang, target_lang
        )

        highlights = [
            Highlight(
                start=h["start"],
                end=h["end"],
                title=h.get("title", ""),
                subtitle=h.get("subtitle", ""),
                content=h.get("content", ""),
                keywords=h.get("keywords", []),
                segment_keywords=h.get("segment_keywords", []),
            )
            for h in raw
        ]

        print(f"✅ Extracted {len(highlights)} highlights")
        for i, h in enumerate(highlights, 1):
            keywords_info = f" (keywords: {', '.join(h.keywords)})" if h.keywords else ""
            print(f"  {i}. [{h.start:.2f}s - {h.end:.2f}s] {h.title}{keywords_info}")
            if h.segment_keywords:
                print(f"     Segment highlights: {len(h.segment_keywords)} segments with keywords")

        return highlights
    
    def analyze_content_with_gemini(
        self, 
        segments: List[Segment], 
        source_lang: str = "zh",
        target_lang: str = "en",
        max_title_chars: int = 50,
        max_subtitle_chars: int = 80
    ) -> Tuple[Dict[str, str], List[Highlight]]:
        """
        Analyze content with Gemini - extract highlights with titles.
        
        The highlights extraction now includes title/subtitle generation,
        so we no longer need a separate title generation step.
        
        Args:
            segments: List of transcription segments
            source_lang: Source language code (zh, en, ja, etc.)
            target_lang: Target language for title/subtitle
            max_title_chars: Maximum characters for title (passed to prompt)
            max_subtitle_chars: Maximum characters for subtitle (passed to prompt)
        
        Returns:
            Tuple of (title_info, highlights)
            - title_info: Dict with 'title' and 'subtitle' from first highlight
            - highlights: List of Highlight objects
        """
        gemini_model = self._get_gemini_client()
        if not gemini_model:
            print("⚠️  LLM API not configured, skipping content analysis")
            return {}, []
        
        # Extract highlights (now includes title/subtitle per highlight)
        highlights = self.analyze_with_gemini_highlights(segments, source_lang, target_lang)
        
        # Use the first highlight's title as the video title
        if highlights and highlights[0].title:
            title_info = {
                "title": highlights[0].title,
                "subtitle": highlights[0].subtitle
            }
            print(f"✅ Using highlight title: {title_info['title']}")
            if title_info['subtitle']:
                print(f"   Subtitle: {title_info['subtitle']}")
        else:
            # Fallback if no highlights extracted
            title_info = {
                "title": "视频精华",
                "subtitle": "Video Highlights"
            }
            print("⚠️  No highlights extracted, using default title")
        
        return title_info, highlights
    
    def translate_text(self, text: str, source_lang: str = "zh", target_lang: str = "en") -> str:
        """Translate text."""
        translated = self.translate_texts_bulk([text], source_lang=source_lang, target_lang=target_lang)
        return translated[0] if translated else text

    def translate_texts_bulk(
        self,
        texts: List[str],
        source_lang: str = "zh",
        target_lang: str = "en",
    ) -> List[str]:
        """Translate texts in batch when possible."""
        return self.translator.translate_bulk(texts, source_lang=source_lang, target_lang=target_lang)
    
    def _extract_transcription_for_range(
        self,
        segments: List[Segment],
        start_time: float,
        end_time: float,
    ) -> str:
        """Extract transcription text for a specific time range."""
        return subtitle_mod.extract_transcription_for_range(segments, start_time, end_time)
    
    def _apply_keyword_highlighting(self, text: str, keywords: List[str]) -> str:
        """Apply ASS markup for keyword highlighting."""
        return subtitle_mod.apply_keyword_highlighting(text, keywords)

    def generate_ass_subtitle(
        self,
        highlights: List[Highlight],
        segments: List[Segment],
        output_path: str,
        translate: bool = False,
        source_lang: str = "zh",
        target_lang: str = "en",
        orientation: str = "landscape",
        subtitle_position: str = "original-top",
        first_subtitle_delay: float = 1.0,
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        highlight_subtitle_color: str = config.DEFAULT_HIGHLIGHT_SUBTITLE_COLOR,
    ) -> str:
        """Generate ASS subtitle file with multi-layer support."""
        translate_fn = self.translate_texts_bulk if translate else None
        return subtitle_mod.generate_ass_subtitle(
            highlights,
            segments,
            output_path,
            translate=translate,
            source_lang=source_lang,
            target_lang=target_lang,
            orientation=orientation,
            subtitle_position=subtitle_position,
            first_subtitle_delay=first_subtitle_delay,
            translate_fn=translate_fn,
            original_subtitle_color=original_subtitle_color,
            translation_subtitle_color=translation_subtitle_color,
            highlight_subtitle_color=highlight_subtitle_color,
        )

    # ------------------------------------------------------------------
    # FCPXML export helpers
    # ------------------------------------------------------------------

    def _get_video_info(self, video_path: str):
        return fcpxml_mod.get_video_info(video_path, self.get_audio_duration)

    def _build_fcpxml_timemap(self, start_f, timeline_dur_f, source_dur_f, fps_int):
        return fcpxml_mod.build_fcpxml_timemap(start_f, timeline_dur_f, source_dur_f, fps_int)

    def _build_fcpxml_title(self, text, translation, offset_frames, duration_frames, fps_int, style_id, orientation):
        return fcpxml_mod.build_fcpxml_title(text, translation, offset_frames, duration_frames, fps_int, style_id, orientation)

    def generate_fcpxml(
        self,
        video_path: str,
        highlights: List[Highlight],
        segments: List[Segment],
        output_path: str,
        frame_rate: float = 25.0,
        speed: float = 1.0,
        translate: bool = False,
        source_lang: str = "zh",
        target_lang: str = "en",
        orientation: str = "landscape",
        enable_clip: bool = True,
        filter_empty_segments: bool = True,
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        highlight_subtitle_color: str = config.DEFAULT_HIGHLIGHT_SUBTITLE_COLOR,
    ) -> str:
        return fcpxml_mod.generate_fcpxml(
            video_path=video_path,
            highlights=highlights,
            segments=segments,
            output_path=output_path,
            frame_rate=frame_rate,
            speed=speed,
            translate=translate,
            source_lang=source_lang,
            target_lang=target_lang,
            orientation=orientation,
            enable_clip=enable_clip,
            filter_empty_segments=filter_empty_segments,
            translate_fn=self.translate_texts_bulk if translate else None,
            original_subtitle_color=original_subtitle_color,
            translation_subtitle_color=translation_subtitle_color,
            highlight_subtitle_color=highlight_subtitle_color,
        )

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        translate: bool = False,
        source_lang: str = "en",
        target_lang: str = "en",
        orientation: str = "landscape",
        subtitle_position: str = "original-top",
        first_subtitle_delay: float = 1.0,
        max_title_chars: int = 50,
        max_subtitle_chars: int = 80,
        enable_clip: bool = True,
        enable_highlight: bool = False,
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        highlight_subtitle_color: str = config.DEFAULT_HIGHLIGHT_SUBTITLE_COLOR,
        filter_empty_segments: bool = True,
        margin_left: float = -0.15,
        margin_right: float = 0.15,
        output_formats: Optional[Iterable[str]] = None,
        export_fcpxml: bool = False,
        fcpxml_frame_rate: float = 25.0,
        fcpxml_speed: float = 1.0,
        transcript_json_path: Optional[str] = None,
        correct_words: bool = False,
    ) -> Dict[str, str]:
        """Complete video processing pipeline with memory management."""
        print(f"\n{'='*60}")
        print(f"🎥 Processing video: {video_path}")
        print(f"{'='*60}\n")
        
        output_dir = str(Path(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        
        results = {}
        if output_formats is None:
            selected_formats = {"fcpxml"} if export_fcpxml else set(DEFAULT_OUTPUT_FORMATS)
        else:
            selected_formats = set(_normalize_output_formats(output_formats))
            if export_fcpxml:
                selected_formats.add("fcpxml")
        want_ass = "ass" in selected_formats
        want_srt = "srt" in selected_formats
        want_fcpxml = "fcpxml" in selected_formats
        want_video = "video" in selected_formats
        want_txt = "txt" in selected_formats
        want_json = "json" in selected_formats
        render_with_highlights = enable_clip and bool({"ass", "video", "fcpxml"} & selected_formats)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = os.path.join(output_dir, f"{video_name}_transcript.json")
            transcript_meta = {"title": "", "subtitle": "", "highlights": []}

            if transcript_json_path:
                # Use provided JSON — skip ASR entirely
                segments, transcript_meta = _load_segments_from_transcript_json(transcript_json_path)
                print(f"📂 Using provided transcript: {transcript_json_path}")
                # Copy to output dir for reference if not already there
                resolved_src = os.path.realpath(transcript_json_path)
                resolved_dst = os.path.realpath(transcript_path) if os.path.exists(transcript_path) else None
                if resolved_dst != resolved_src:
                    import shutil
                    shutil.copy2(transcript_json_path, transcript_path)
            elif os.path.exists(transcript_path):
                segments, transcript_meta = _load_segments_from_transcript_json(transcript_path)
                print(f"♻️  Reusing existing transcript: {transcript_path}")
            else:
                # Step 1: Extract audio (skip if input is already an audio file)
                audio_path = os.path.join(tmpdir, "audio.wav")
                self.extract_audio(video_path, audio_path)

                # Step 2: Transcribe audio (ASR model loaded on demand)
                try:
                    segments = self.transcribe_audio(audio_path, orientation=orientation, source_lang=source_lang)
                finally:
                    # ASR/aligner are only needed for transcription. Release them
                    # before correction, chunking, translation, highlights, or render work.
                    self._unload_asr_model()

                # Save transcription in new metadata-wrapped format
                transcript_data = {
                    "title": "",
                    "subtitle": "",
                    "segments": [{"start": s.start, "end": s.end, "text": s.text, "words": s.words or []} for s in segments],
                    "highlights": []
                }
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Transcription saved to {transcript_path}")
            
            if correct_words:
                segments = self._correct_transcript(segments, source_lang)
                # Re-save transcript with corrected text
                corrected_data = {
                    "title": transcript_meta.get("title", ""),
                    "subtitle": transcript_meta.get("subtitle", ""),
                    "segments": [
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "words": seg.words,
                        }
                        for seg in segments
                    ],
                    "highlights": transcript_meta.get("highlights", []),
                }
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(corrected_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Corrected transcript saved to {transcript_path}")
            
            subtitle_segments = self._filter_subtitle_segments(segments, filter_empty_segments=filter_empty_segments)
            if margin_left != 0.0 or margin_right != 0.0:
                subtitle_segments = self._resolve_overlaps(subtitle_segments, margin_left, margin_right)
                print(f"⏱️  Applied margin: left={margin_left*1000:.0f}ms, right={margin_right*1000:.0f}ms")
            results["transcript"] = transcript_path

            if want_json and not (want_ass or want_srt or want_fcpxml or want_video or want_txt):
                return results

            if want_txt:
                txt_path = os.path.join(output_dir, f"{video_name}.txt")
                full_text = "\n".join(seg.text.strip() for seg in subtitle_segments if seg.text.strip())
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                    if full_text:
                        f.write("\n")
                results["txt"] = txt_path
                print(f"💾 Plain text saved to {txt_path}")

            if render_with_highlights:
                # Step 3: Split transcript into <=5-min chunks and analyze each with Gemini
                chunked_segments = self.split_transcript_segments(subtitle_segments, float(self.segment_duration))
                print(f"✂️  Splitting transcript into {len(chunked_segments)} chunks (<= {self.segment_duration}s)")

                for chunk_idx, chunk_segments in enumerate(chunked_segments, 1):
                    chunk_start = chunk_segments[0].start
                    chunk_end = chunk_segments[-1].end
                    print(f"\n🧩 Processing chunk {chunk_idx}/{len(chunked_segments)}: {chunk_start:.2f}s - {chunk_end:.2f}s")

                    title_info, highlights = self.analyze_content_with_gemini(
                        chunk_segments,
                        source_lang,
                        target_lang,
                        max_title_chars,
                        max_subtitle_chars
                    )

                    if highlights:
                        adjusted = []
                        for h in highlights:
                            start = max(h.start, chunk_start)
                            end = min(h.end, chunk_end)
                            if end <= start:
                                continue
                            if start != h.start or end != h.end:
                                h = Highlight(
                                    start=start,
                                    end=end,
                                    title=h.title,
                                    subtitle=h.subtitle,
                                    content=h.content,
                                    keywords=h.keywords,
                                    segment_keywords=h.segment_keywords,
                                )
                            adjusted.append(h)
                        highlights = adjusted

                    if not highlights:
                        print("⚠️  No highlights extracted, using full chunk")
                        highlights = [Highlight(
                            start=chunk_start,
                            end=chunk_end,
                            title=title_info.get("title", "完整视频"),
                            subtitle=title_info.get("subtitle", "Full Video"),
                            content="完整片段内容"
                        )]

                    chunk_suffix = f"part{chunk_idx:03d}"

                    # Save highlights as JSON for downstream processing
                    highlights_data = {
                        "title": title_info.get("title", ""),
                        "subtitle": title_info.get("subtitle", ""),
                        "segments": [{"start": s.start, "end": s.end, "text": s.text, "words": s.words or []} for s in chunk_segments],
                        "highlights": [
                            {
                                "start": h.start,
                                "end": h.end,
                                "title": h.title,
                                "subtitle": h.subtitle,
                                "content": h.content,
                                "keywords": h.keywords or [],
                                "segment_keywords": h.segment_keywords or [],
                            }
                            for h in highlights
                        ]
                    }
                    highlights_json_path = os.path.join(output_dir, f"{video_name}_{chunk_suffix}_highlights.json")
                    with open(highlights_json_path, "w", encoding="utf-8") as f:
                        json.dump(highlights_data, f, ensure_ascii=False, indent=2)
                    print(f"💾 Highlights JSON saved to {highlights_json_path}")
                    results[f"highlights_json_{chunk_suffix}"] = highlights_json_path

                    # Save summary per chunk
                    summary_path = os.path.join(output_dir, f"{video_name}_{chunk_suffix}_summary.txt")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        title = title_info.get("title", "完整视频")
                        subtitle = title_info.get("subtitle", "Full Video")
                        f.write(f"{title}:{subtitle}\n")

                        if highlights and highlights[0].keywords:
                            keywords_str = " ".join('#' + kw for kw in highlights[0].keywords)
                            f.write(f"{keywords_str}\n")
                        else:
                            f.write("\n")

                    results[f"summary_{chunk_suffix}"] = summary_path
                    print(f"💾 Summary saved to {summary_path}")

                    if want_srt:
                        srt_path = os.path.join(output_dir, f"{video_name}_{chunk_suffix}_subtitles.srt")
                        srt_content = _segments_to_srt(
                            [(s.start, s.end, s.text) for s in chunk_segments if s.text.strip()]
                        )
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(srt_content)
                        results[f"srt_{chunk_suffix}"] = srt_path

                    subtitle_path: Optional[str] = None
                    if want_ass or want_video:
                        subtitle_path = (
                            os.path.join(output_dir, f"{video_name}_{chunk_suffix}_subtitles.ass")
                            if want_ass
                            else os.path.join(tmpdir, f"{video_name}_{chunk_suffix}_subtitles.ass")
                        )
                        self.generate_ass_subtitle(
                            highlights, chunk_segments, subtitle_path,
                            translate=translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            orientation=orientation,
                            subtitle_position=subtitle_position,
                            first_subtitle_delay=first_subtitle_delay,
                            original_subtitle_color=original_subtitle_color,
                            translation_subtitle_color=translation_subtitle_color,
                            highlight_subtitle_color=highlight_subtitle_color,
                        )
                        if want_ass:
                            results[f"subtitles_{chunk_suffix}"] = subtitle_path

                    if want_fcpxml:
                        fcpxml_path = os.path.join(output_dir, f"{video_name}_{chunk_suffix}.fcpxml")
                        self.generate_fcpxml(
                            video_path=video_path,
                            highlights=highlights,
                            segments=chunk_segments,
                            output_path=fcpxml_path,
                            frame_rate=fcpxml_frame_rate,
                            speed=fcpxml_speed,
                            translate=translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            orientation=orientation,
                            enable_clip=True,
                            filter_empty_segments=filter_empty_segments,
                            original_subtitle_color=original_subtitle_color,
                            translation_subtitle_color=translation_subtitle_color,
                            highlight_subtitle_color=highlight_subtitle_color,
                        )
                        results[f"fcpxml_{chunk_suffix}"] = fcpxml_path

                    if want_video and subtitle_path:
                        final_video_path = os.path.join(output_dir, f"{video_name}_{chunk_suffix}_final.mp4")
                        self.render_video_with_subtitles_complex(
                            video_path=video_path,
                            highlights=highlights,
                            subtitle_path=subtitle_path,
                            output_path=final_video_path,
                            orientation=orientation,
                        )
                        results[f"final_video_{chunk_suffix}"] = final_video_path

            else:
                # No-clip mode: skip AI highlight extraction and export as single merged resource
                print("\n📹 No-clip mode: skip AI highlight extraction and merge chunks into one output")
                chunked_segments = self.split_transcript_segments(subtitle_segments, float(self.segment_duration))
                print(f"✂️  Splitting transcript into {len(chunked_segments)} chunks (<= {self.segment_duration}s)")
                merged_segments = [seg for chunk in chunked_segments for seg in chunk]

                if merged_segments:
                    # Keyword extraction for --highlight in no-clip mode
                    all_segment_keywords: List[Dict] = []
                    gemini_model = self._get_gemini_client()
                    if enable_highlight and gemini_model:
                        print("🔍 Highlight mode: extracting keywords via LLM...")
                        offset = 0
                        for chunk in chunked_segments:
                            chunk_kws = analysis.extract_keywords_for_segments(
                                gemini_model, chunk, source_lang, target_lang
                            )
                            for kw in chunk_kws:
                                all_segment_keywords.append({
                                    "segment_id": kw["segment_id"] + offset,
                                    "keywords": kw["keywords"],
                                })
                            offset += len(chunk)
                        print(f"✅ Extracted keywords for {len(all_segment_keywords)} segment(s)")
                    elif enable_highlight and not gemini_model:
                        print("⚠️  --highlight requires an API key; skipping keyword extraction")

                    seg_kw_lookup = {kw["segment_id"]: kw["keywords"] for kw in all_segment_keywords}

                    if self.filter_fillers:
                        merged_highlights = [
                            Highlight(
                                start=seg.start,
                                end=seg.end,
                                title="",
                                subtitle="",
                                content="",
                                segment_keywords=(
                                    [{"segment_id": global_idx, "keywords": seg_kw_lookup[global_idx]}]
                                    if global_idx in seg_kw_lookup
                                    else []
                                ),
                            )
                            for global_idx, seg in enumerate(merged_segments)
                        ]
                    else:
                        merged_highlights = [Highlight(
                            start=merged_segments[0].start,
                            end=merged_segments[-1].end,
                            title=transcript_meta.get("title", ""),
                            subtitle=transcript_meta.get("subtitle", ""),
                            content="",
                            segment_keywords=all_segment_keywords,
                        )]


                    if want_srt:
                        srt_path = os.path.join(output_dir, f"{video_name}_subtitles.srt")
                        srt_content = _segments_to_srt(
                            [(s.start, s.end, s.text) for s in merged_segments if s.text.strip()]
                        )
                        with open(srt_path, "w", encoding="utf-8") as f:
                            f.write(srt_content)
                        results["srt"] = srt_path

                    subtitle_path: Optional[str] = None
                    if want_ass or want_video:
                        subtitle_path = (
                            os.path.join(output_dir, f"{video_name}_subtitles.ass")
                            if want_ass
                            else os.path.join(tmpdir, f"{video_name}_subtitles.ass")
                        )
                        self.generate_ass_subtitle(
                            merged_highlights, merged_segments, subtitle_path,
                            translate=translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            orientation=orientation,
                            subtitle_position=subtitle_position,
                            first_subtitle_delay=0.0,
                            original_subtitle_color=original_subtitle_color,
                            translation_subtitle_color=translation_subtitle_color,
                            highlight_subtitle_color=highlight_subtitle_color,
                        )
                        if want_ass:
                            results["subtitles"] = subtitle_path

                    if want_fcpxml:
                        fcpxml_path = os.path.join(output_dir, f"{video_name}.fcpxml")
                        self.generate_fcpxml(
                            video_path=video_path,
                            highlights=merged_highlights,
                            segments=merged_segments,
                            output_path=fcpxml_path,
                            frame_rate=fcpxml_frame_rate,
                            speed=fcpxml_speed,
                            translate=translate,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            orientation=orientation,
                            enable_clip=True,
                            filter_empty_segments=filter_empty_segments,
                            original_subtitle_color=original_subtitle_color,
                            translation_subtitle_color=translation_subtitle_color,
                            highlight_subtitle_color=highlight_subtitle_color,
                        )
                        results["fcpxml"] = fcpxml_path

                    if want_video and subtitle_path:
                        final_video_path = os.path.join(output_dir, f"{video_name}_final.mp4")
                        self.render_video_with_subtitles_complex(
                            video_path=video_path,
                            highlights=merged_highlights,
                            subtitle_path=subtitle_path,
                            output_path=final_video_path,
                            orientation=orientation,
                        )
                        results["final_video"] = final_video_path

        print(f"\n{'='*60}")
        print(f"✅ Processing complete!")
        print(f"{'='*60}\n")
        print("📦 Output files:")
        for key, path in results.items():
            print(f"  - {key}: {path}")
        print()
        
        return results

 
