"""VideoClipper — main video clipping pipeline."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

import pycut.config as config
import pycut.fcpxml as fcpxml_mod
import pycut.renderer as renderer_mod
import pycut.subtitle as subtitle_mod
from pycut.asr import MLXASRHelper, QwenASRHelper
from pycut.language import detect_language as detect_spoken_language
from pycut.media_job import MediaJob
from pycut.media_workflow import MediaJobWorkflow, WorkflowAdapters
from pycut.timeline import (
    TranscriptTimeline,
    prepare_timeline,
    split_transcript_segments,
    timeline_to_segments,
)
from pycut.transcript_store import TranscriptStore
from pycut.translation import GoogleTranslator
from pycut.utils import (
    Segment,
    _segments_to_srt,
    extract_audio,
    get_audio_duration,
)


class VideoClipper:
    """Local media pipeline with ASR, translation, subtitle rendering, and TTS-adjacent exports."""
    
    def __init__(
        self,
        asr_model_path: Optional[str] = None,
        aligner_model_path: Optional[str] = None,
        enable_align: bool = True,
        segment_duration: int = 300,  # 5 minutes
        max_duration: float = 30.0,
        max_chars: int = 30,
        filter_fillers: bool = True,
        translator: Optional[GoogleTranslator] = None,
    ):
        config.ensure_supported_runtime()
        self.segment_duration = segment_duration
        runtime_profile = config.current_runtime_profile()
        self.runtime_profile = runtime_profile
        self.asr_backend = runtime_profile.asr_backend
        self._asr_model_explicit = asr_model_path is not None
        default_asr_model = runtime_profile.default_asr_model("en")
        default_aligner_model = runtime_profile.default_aligner_model()
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
    
    def _get_asr_helper(self) -> MLXASRHelper:
        helper = getattr(self, "asr_helper", None)
        if helper is None:
            backend = getattr(self, "asr_backend", config.current_runtime_profile().asr_backend)
            helper_cls = MLXASRHelper if backend == "mlx" else QwenASRHelper
            runtime_profile = getattr(self, "runtime_profile", None)
            if runtime_profile is not None and runtime_profile.asr_backend == backend:
                default_asr_model = runtime_profile.default_asr_model("en")
                default_aligner_model = runtime_profile.default_aligner_model()
            elif backend == "mlx":
                default_asr_model = config.DEFAULT_EN_ASR_MODEL
                default_aligner_model = config.DEFAULT_ALIGNER_MODEL
            else:
                default_asr_model = config.resolve_default_qwen_asr_model()
                default_aligner_model = config.resolve_default_qwen_aligner_model()
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

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video as WAV 16kHz mono."""
        return extract_audio(video_path, output_path)

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        return detect_spoken_language(audio_path)

    def _select_video_encoder(self) -> str:
        return renderer_mod.select_video_encoder()

    def render_video_with_subtitles_complex(
        self,
        video_path: str,
        timeline: TranscriptTimeline,
        subtitle_path: str,
        output_path: str,
        orientation: str = "landscape",
        target_resolution: Optional[str] = None,
    ) -> str:
        return renderer_mod.render_video_with_subtitles_complex(
            video_path=video_path,
            timeline=timeline,
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
        return split_transcript_segments(segments, max_duration)

    @staticmethod
    def _resolve_overlaps(segments: List[Segment], margin_left: float = 0.0, margin_right: float = 0.0) -> List[Segment]:
        """Apply per-segment margin offsets then resolve any overlaps using midpoint splitting."""
        return timeline_to_segments(
            prepare_timeline(
                segments,
                filter_empty_segments=False,
                margin_left=margin_left,
                margin_right=margin_right,
            )
        )

    def _filter_subtitle_segments(
        self,
        segments: List[Segment],
        filter_empty_segments: bool = True,
    ) -> List[Segment]:
        """Filter empty segments and resolve any overlaps."""
        return timeline_to_segments(
            prepare_timeline(segments, filter_empty_segments=filter_empty_segments)
        )
    
    def transcribe_audio(self, audio_path: str, orientation: str = "landscape", source_lang: str = "en") -> List[Segment]:
        """Transcribe audio file with word-level timestamps."""
        print(f"🎤 Transcribing {audio_path}...")

        if not getattr(self, "_asr_model_explicit", True):
            asr_model_path = self.runtime_profile.default_asr_model(source_lang)
            if asr_model_path != self.asr_model_path:
                self._unload_asr_model()
                self.asr_model_path = asr_model_path
                helper_cls = MLXASRHelper if self.asr_backend == "mlx" else QwenASRHelper
                self.asr_helper = helper_cls(
                    asr_model_path=self.asr_model_path,
                    aligner_model_path=self.aligner_model_path,
                    filter_fillers=self.filter_fillers,
                    enable_align=self.enable_align,
                )
        
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
    
    def generate_ass_subtitle(
        self,
        timeline: TranscriptTimeline,
        output_path: str,
        translate: bool = False,
        source_lang: str = "zh",
        target_lang: str = "en",
        orientation: str = "landscape",
        subtitle_position: str = "original-top",
        first_subtitle_delay: float = 1.0,
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
    ) -> str:
        """Generate ASS subtitle file with multi-layer support."""
        translate_fn = self.translate_texts_bulk if translate else None
        return subtitle_mod.generate_ass_subtitle(
            timeline,
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
        )

    # ------------------------------------------------------------------
    # FCPXML export helpers
    # ------------------------------------------------------------------

    def _get_video_info(self, video_path: str):
        return fcpxml_mod.get_video_info(video_path, self.get_audio_duration)

    def _build_fcpxml_timemap(self, start_f, timeline_dur_f, source_dur_f, fps_int):
        return fcpxml_mod.build_fcpxml_timemap(start_f, timeline_dur_f, source_dur_f, fps_int)

    def generate_fcpxml(
        self,
        video_path: str,
        timeline: TranscriptTimeline,
        output_path: str,
        frame_rate: float = 25.0,
        speed: float = 1.0,
        translate: bool = False,
        source_lang: str = "zh",
        target_lang: str = "en",
        orientation: str = "landscape",
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
    ) -> str:
        return fcpxml_mod.generate_fcpxml(
            video_path=video_path,
            timeline=timeline,
            output_path=output_path,
            frame_rate=frame_rate,
            speed=speed,
            translate=translate,
            source_lang=source_lang,
            target_lang=target_lang,
            orientation=orientation,
            translate_fn=self.translate_texts_bulk if translate else None,
            original_subtitle_color=original_subtitle_color,
            translation_subtitle_color=translation_subtitle_color,
        )

    def process_fcpxml(
        self,
        input_path: str,
        output_dir: str,
        *,
        transcript_json_path: Optional[str],
        translate: bool = False,
        source_lang: Optional[str] = "en",
        target_lang: str = "en",
        orientation: str = "landscape",
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        filter_empty_segments: bool = True,
        margin_left: float = -0.15,
        margin_right: float = 0.15,
    ) -> Dict[str, str]:
        """Rough-cut an FCPXML story timeline from transcript cue ranges."""
        if not transcript_json_path:
            raise RuntimeError("FCPXML rough-cut input requires an aligned --transcript JSON file")

        stem = Path(input_path).stem
        store = TranscriptStore(output_dir, stem)
        document = store.load_provided(transcript_json_path)
        timeline = prepare_timeline(
            document.segments,
            title=document.metadata.title,
            subtitle=document.metadata.subtitle,
            filter_empty_segments=filter_empty_segments,
            margin_left=margin_left,
            margin_right=margin_right,
        )
        output_path = str(Path(output_dir) / f"{stem}.fcpxml")
        fcpxml_mod.rough_cut_fcpxml(
            input_path,
            timeline,
            output_path,
            translate=translate,
            source_lang=source_lang or "auto",
            target_lang=target_lang,
            orientation=orientation,
            translate_fn=self.translate_texts_bulk if translate else None,
            original_subtitle_color=original_subtitle_color,
            translation_subtitle_color=translation_subtitle_color,
        )
        return {"transcript": str(store.path), "fcpxml": output_path}

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        translate: bool = False,
        source_lang: Optional[str] = "en",
        target_lang: str = "en",
        orientation: str = "landscape",
        subtitle_position: str = "original-top",
        first_subtitle_delay: float = 1.0,
        original_subtitle_color: str = config.DEFAULT_ORIGINAL_SUBTITLE_COLOR,
        translation_subtitle_color: str = config.DEFAULT_TRANSLATION_SUBTITLE_COLOR,
        filter_empty_segments: bool = True,
        margin_left: float = -0.15,
        margin_right: float = 0.15,
        output_formats: Optional[Iterable[str]] = None,
        export_fcpxml: bool = False,
        fcpxml_frame_rate: float = 25.0,
        fcpxml_speed: float = 1.0,
        transcript_json_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Complete local media processing pipeline with memory management."""
        print(f"\n{'='*60}")
        print(f"🎥 Processing video: {video_path}")
        print(f"{'='*60}\n")
        
        job = MediaJob(
            video_path=video_path,
            output_dir=str(Path(output_dir)),
            translate=translate,
            source_lang=source_lang,
            target_lang=target_lang,
            orientation=orientation,
            subtitle_position=subtitle_position,
            first_subtitle_delay=first_subtitle_delay,
            original_subtitle_color=original_subtitle_color,
            translation_subtitle_color=translation_subtitle_color,
            filter_empty_segments=filter_empty_segments,
            margin_left=margin_left,
            margin_right=margin_right,
            output_formats=output_formats,
            export_fcpxml=export_fcpxml,
            fcpxml_frame_rate=fcpxml_frame_rate,
            fcpxml_speed=fcpxml_speed,
            transcript_json_path=transcript_json_path,
        )
        adapters = WorkflowAdapters(
            extract_audio=self.extract_audio,
            detect_language=self.detect_language,
            transcribe_audio=self.transcribe_audio,
            unload_asr_model=self._unload_asr_model,
            generate_ass_subtitle=self.generate_ass_subtitle,
            generate_fcpxml=self.generate_fcpxml,
            render_video_with_subtitles_complex=self.render_video_with_subtitles_complex,
        )
        return MediaJobWorkflow(
            job,
            adapters=adapters,
            segment_duration=self.segment_duration,
        ).run()

 
