"""Media job workflow orchestration."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pycut.media_job import MediaJob
from pycut.timeline import TranscriptTimeline, prepare_export_timeline, prepare_timeline
from pycut.transcript_store import TranscriptMetadata
from pycut.utils import Segment, _segments_to_srt


@dataclass(frozen=True)
class WorkflowAdapters:
    """Concrete adapters used by the media job workflow."""

    extract_audio: Callable[[str, str], str]
    transcribe_audio: Callable[..., List[Segment]]
    unload_asr_model: Callable[[], None]
    generate_ass_subtitle: Callable[..., str]
    generate_fcpxml: Callable[..., str]
    render_video_with_subtitles_complex: Callable[..., str]
    detect_language: Optional[Callable[[str], tuple[str, float]]] = None


class MediaJobWorkflow:
    """Run one local media job from transcript acquisition through selected exports."""

    def __init__(
        self,
        job: MediaJob,
        *,
        adapters: WorkflowAdapters,
        segment_duration: int,
    ):
        self.job = job
        self.adapters = adapters
        self.segment_duration = segment_duration

    def _load_or_create_transcript(self, tmpdir: str) -> tuple[List[Segment], TranscriptMetadata, str]:
        job = self.job
        transcript_store = job.transcript_store()
        transcript_path = str(transcript_store.path)
        transcript_metadata = TranscriptMetadata()

        if job.transcript_json_path:
            transcript_document = transcript_store.load_provided(job.transcript_json_path)
            print(f"📂 Using provided transcript: {job.transcript_json_path}")
        elif (transcript_document := transcript_store.load_existing()) is not None:
            print(f"♻️  Reusing existing transcript: {transcript_path}")
        else:
            audio_path = os.path.join(tmpdir, "audio.wav")
            self.adapters.extract_audio(job.video_path, audio_path)
            if not job.source_lang:
                if self.adapters.detect_language is None:
                    raise RuntimeError("Source language is required when automatic detection is unavailable")
                job.source_lang, confidence = self.adapters.detect_language(audio_path)
                print(f"🌐 Detected source language: {job.source_lang} ({confidence:.1%})")
            try:
                segments = self.adapters.transcribe_audio(
                    audio_path,
                    orientation=job.orientation,
                    source_lang=job.source_lang,
                )
            finally:
                self.adapters.unload_asr_model()

            transcript_store.save(segments, metadata=transcript_metadata)
            print(f"💾 Transcription saved to {transcript_path}")
            return segments, transcript_metadata, transcript_path

        if not job.source_lang:
            job.source_lang = "auto"
        return transcript_document.segments, transcript_document.metadata, transcript_path

    def _write_txt(self, timeline: TranscriptTimeline) -> tuple[str, str]:
        txt_path = self.job.output_path(".txt")
        full_text = "\n".join(timeline.text_lines())
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            if full_text:
                f.write("\n")
        print(f"💾 Plain text saved to {txt_path}")
        return "txt", txt_path

    def _write_srt(self, timeline: TranscriptTimeline) -> tuple[str, str]:
        srt_path = self.job.output_path("_subtitles.srt")
        srt_content = _segments_to_srt(
            [(cue.start, cue.end, cue.text) for cue in timeline.cues if cue.text.strip()]
        )
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return "srt", srt_path

    def _write_ass(self, timeline: TranscriptTimeline, tmpdir: str, *, keep_file: bool) -> Optional[str]:
        subtitle_path = (
            self.job.output_path("_subtitles.ass")
            if keep_file
            else os.path.join(tmpdir, f"{self.job.video_name}_subtitles.ass")
        )
        self.adapters.generate_ass_subtitle(
            timeline,
            subtitle_path,
            translate=self.job.translate,
            source_lang=self.job.source_lang,
            target_lang=self.job.target_lang,
            orientation=self.job.orientation,
            subtitle_position=self.job.subtitle_position,
            first_subtitle_delay=0.0,
            original_subtitle_color=self.job.original_subtitle_color,
            translation_subtitle_color=self.job.translation_subtitle_color,
        )
        return subtitle_path

    def run(self) -> Dict[str, str]:
        job = self.job
        print(f"\n{'='*60}")
        print(f"🎥 Processing video: {job.video_path}")
        print(f"{'='*60}\n")

        Path(job.output_dir).mkdir(parents=True, exist_ok=True)
        results: Dict[str, str] = {}
        selected_formats = job.selected_formats()
        want_ass = "ass" in selected_formats
        want_srt = "srt" in selected_formats
        want_fcpxml = "fcpxml" in selected_formats
        want_video = "video" in selected_formats
        want_txt = "txt" in selected_formats
        want_json = "json" in selected_formats

        with tempfile.TemporaryDirectory() as tmpdir:
            segments, transcript_metadata, transcript_path = self._load_or_create_transcript(tmpdir)

            removed_segments = (
                len([seg for seg in segments if not str(getattr(seg, "text", "") or "").strip()])
                if job.filter_empty_segments
                else 0
            )
            if removed_segments > 0:
                print(f"🧹 Filtered {removed_segments} empty subtitle segment(s)")

            timeline = prepare_timeline(
                segments,
                title=transcript_metadata.title,
                subtitle=transcript_metadata.subtitle,
                filter_empty_segments=job.filter_empty_segments,
                margin_left=job.margin_left,
                margin_right=job.margin_right,
            )
            if job.margin_left != 0.0 or job.margin_right != 0.0:
                print(f"⏱️  Applied margin: left={job.margin_left*1000:.0f}ms, right={job.margin_right*1000:.0f}ms")
            results["transcript"] = transcript_path

            if want_json and not (want_ass or want_srt or want_fcpxml or want_video or want_txt):
                return results

            if want_txt:
                key, txt_path = self._write_txt(timeline)
                results[key] = txt_path

            print("\n📹 Local timeline mode: exporting transcript-based outputs")
            export_timeline, cue_chunks = prepare_export_timeline(
                timeline,
                max_duration=float(self.segment_duration),
            )
            print(f"✂️  Splitting transcript into {len(cue_chunks)} chunks (<= {self.segment_duration}s)")

            if export_timeline.cues:
                if want_srt:
                    key, srt_path = self._write_srt(export_timeline)
                    results[key] = srt_path

                subtitle_path: Optional[str] = None
                if want_ass or want_video:
                    subtitle_path = self._write_ass(export_timeline, tmpdir, keep_file=want_ass)
                    if want_ass and subtitle_path:
                        results["subtitles"] = subtitle_path

                if want_fcpxml:
                    fcpxml_path = job.output_path(".fcpxml")
                    self.adapters.generate_fcpxml(
                        video_path=job.video_path,
                        timeline=export_timeline,
                        output_path=fcpxml_path,
                        frame_rate=job.fcpxml_frame_rate,
                        speed=job.fcpxml_speed,
                        translate=job.translate,
                        source_lang=job.source_lang,
                        target_lang=job.target_lang,
                        orientation=job.orientation,
                        original_subtitle_color=job.original_subtitle_color,
                        translation_subtitle_color=job.translation_subtitle_color,
                    )
                    results["fcpxml"] = fcpxml_path

                if want_video and subtitle_path:
                    final_video_path = job.output_path("_final.mp4")
                    self.adapters.render_video_with_subtitles_complex(
                        video_path=job.video_path,
                        timeline=export_timeline,
                        subtitle_path=subtitle_path,
                        output_path=final_video_path,
                        orientation=job.orientation,
                    )
                    results["final_video"] = final_video_path

        print(f"\n{'='*60}")
        print("✅ Processing complete!")
        print(f"{'='*60}\n")
        print("📦 Output files:")
        for key, path in results.items():
            print(f"  - {key}: {path}")
        print()

        return results
