from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

try:
    import torch
except ImportError:
    torch = None

from pycut.utils import (
    Segment,
    _attach_punctuation_to_words,
    _needs_space,
    _split_vad_segment_by_punctuation,
    filter_filler_words,
    filter_text,
)

__all__ = ["MLXASRHelper", "load_mlx_stt_model"]


def load_mlx_stt_model(model_name: str):
    try:
        from mlx_audio.stt.utils import load as stt_load

        return stt_load(model_name)
    except ImportError as exc:
        raise RuntimeError("MLX backend requires mlx-audio. Install with: pip install mlx-audio") from exc


@dataclass
class _MLXTimestampItem:
    text: str
    start_time: float
    end_time: float


class MLXASRHelper:
    def __init__(
        self,
        *,
        asr_model_path: str,
        aligner_model_path: str,
        filter_fillers: bool = True,
        enable_align: bool = True,
    ):
        self.asr_model_path = asr_model_path
        self.aligner_model_path = aligner_model_path
        self.filter_fillers = filter_fillers
        self.enable_align = enable_align
        self.asr_model = None
        self._mlx_aligner = None
        self.vad_model = None

    def load_models(self):
        """Load the MLX ASR model on demand.

        Kept for backward compatibility with VideoClipper._load_asr_model().
        The aligner is loaded separately after ASR has been released.
        """
        self.load_asr_model()

    def load_asr_model(self):
        """Load only the MLX ASR model on demand."""
        if self.asr_model is not None:
            return

        print(f"📝 Loading MLX ASR model from {self.asr_model_path}...")
        self.asr_model = load_mlx_stt_model(self.asr_model_path)
        print("✅ ASR model loaded!")

    def unload_asr_model(self):
        """Unload only the ASR model."""
        if self.asr_model is None:
            return

        print("🧹 Unloading ASR model to free memory...")
        self.asr_model = None
        gc.collect()
        print("✅ ASR model unloaded!")

    def load_aligner_model(self):
        """Load only the MLX aligner model on demand."""
        if not self.enable_align:
            self._mlx_aligner = None
            print("⏭️  Alignment disabled; skipping MLX aligner load.")
            return

        if self._mlx_aligner is not None:
            return

        if self.enable_align:
            print(f"📝 Loading MLX aligner model from {self.aligner_model_path}...")
            try:
                self._mlx_aligner = load_mlx_stt_model(self.aligner_model_path)
            except ValueError as exc:
                self._mlx_aligner = None
                print(f"⚠️  Failed to load MLX aligner ({exc}); continuing without word alignment.")
        if self._mlx_aligner is not None:
            print("✅ Aligner model loaded!")

    def unload_aligner_model(self):
        """Unload only the aligner model."""
        if self._mlx_aligner is None:
            return

        print("🧹 Unloading aligner model to free memory...")
        self._mlx_aligner = None
        gc.collect()
        print("✅ Aligner model unloaded!")

    def unload_vad_model(self):
        """Unload only the VAD model."""
        if self.vad_model is None:
            return

        print("🧹 Unloading VAD model to free memory...")
        self.vad_model = None
        gc.collect()
        print("✅ VAD model unloaded!")

    def unload_models(self):
        """Unload VAD, ASR, and aligner models to free memory."""
        if self.asr_model is None and self._mlx_aligner is None and self.vad_model is None:
            return

        print("🧹 Unloading ASR/VAD models to free memory...")
        self.unload_asr_model()
        self.unload_aligner_model()
        self.unload_vad_model()
        gc.collect()
        print("✅ ASR/VAD models unloaded!")

    def load_vad_model(self):
        """Load Silero VAD model on demand."""
        if self.vad_model is not None:
            return

        from silero_vad import load_silero_vad

        print("🔊 Loading Silero VAD model...")
        self.vad_model = load_silero_vad()
        print("✅ VAD model loaded!")

    def _generate_asr_text(self, audio_path: str, source_lang: str) -> str:
        """Generate transcript text with the already-loaded ASR model."""
        try:
            result = self.asr_model.generate(audio_path, language=source_lang)
        except TypeError:
            result = self.asr_model.generate(audio_path)
        return str(getattr(result, "text", "") or "").strip()

    def _align_text(
        self,
        audio_path: str,
        text: str,
        time_offset: float,
        source_lang: str,
    ) -> List[_MLXTimestampItem]:
        """Align transcript text with the already-loaded aligner model."""
        if not self.enable_align or self._mlx_aligner is None:
            return []

        try:
            align_result = self._mlx_aligner.generate(audio_path, text=text, language=source_lang)
        except TypeError:
            align_result = self._mlx_aligner.generate(audio_path, text=text)

        aligned_items = []
        for item in align_result:
            token = str(getattr(item, "text", "")).strip()
            if not token:
                continue
            aligned_items.append(
                _MLXTimestampItem(
                    text=token,
                    start_time=float(getattr(item, "start_time", 0.0)) + time_offset,
                    end_time=float(getattr(item, "end_time", 0.0)) + time_offset,
                )
            )
        return aligned_items

    def _segments_from_text_and_alignment(
        self,
        audio_path: str,
        text: str,
        aligned_items: List[_MLXTimestampItem],
        time_offset: float,
        max_chars: int,
        *,
        get_audio_duration=None,
        fallback_duration: float | None = None,
    ) -> List[Segment]:
        """Build subtitle segments from ASR text and optional aligned words."""
        if not aligned_items:
            if fallback_duration is None:
                if get_audio_duration is None:
                    raise RuntimeError("MLX ASR helper requires get_audio_duration when alignment is unavailable")
                fallback_duration = get_audio_duration(audio_path)
            return [
                Segment(
                    start=time_offset,
                    end=time_offset + fallback_duration,
                    text=text,
                    words=[],
                )
            ]

        text = filter_text(text, filter_fillers=self.filter_fillers)
        filtered_items = filter_filler_words(aligned_items, enabled=self.filter_fillers)
        words = _attach_punctuation_to_words(filtered_items, text)
        seg_start = filtered_items[0].start_time if filtered_items else time_offset
        seg_end = filtered_items[-1].end_time if filtered_items else time_offset
        return _split_vad_segment_by_punctuation(words, seg_start, seg_end, max_chars=max_chars)

    def transcribe_audio(
        self,
        audio_path: str,
        time_offset: float = 0.0,
        max_chars: int = 60,
        source_lang: str = "en",
        *,
        get_audio_duration=None,
    ) -> List[Segment]:
        """Transcribe with MLX Parakeet and align words with MLX Qwen3-ForcedAligner."""
        self.load_asr_model()
        try:
            text = self._generate_asr_text(audio_path, source_lang)
        finally:
            self.unload_asr_model()

        if not text:
            return []

        aligned_items = []
        if self.enable_align:
            self.load_aligner_model()
            try:
                aligned_items = self._align_text(audio_path, text, time_offset, source_lang)
            finally:
                self.unload_aligner_model()

        return self._segments_from_text_and_alignment(
            audio_path,
            text,
            aligned_items,
            time_offset,
            max_chars,
            get_audio_duration=get_audio_duration,
        )

    def collect_offset_items_for_audio(
        self,
        audio_path: str,
        base_offset: float = 0.0,
        source_lang: str = "en",
        *,
        get_audio_duration=None,
    ) -> Tuple[list, str]:
        """Collect timestamp items and reconstructed text for a speech segment."""
        segments = self.transcribe_audio(
            audio_path,
            time_offset=base_offset,
            max_chars=10000,
            source_lang=source_lang,
            get_audio_duration=get_audio_duration,
        )
        items = []
        texts = []
        for seg in segments:
            words = list(seg.words or [])
            if words:
                parts = []
                prev = ""
                for word in words:
                    token = str(word.get("word", "")).strip()
                    if not token:
                        continue
                    if _needs_space(prev, token):
                        parts.append(" ")
                    parts.append(token)
                    punct = word.get("punctuation", "")
                    if punct:
                        parts.append(punct)
                    prev = token
                    items.append(
                        _MLXTimestampItem(
                            text=token,
                            start_time=float(word.get("start", seg.start)),
                            end_time=float(word.get("end", seg.end)),
                        )
                    )
                texts.append("".join(parts))
                continue

            text = str(seg.text or "").strip()
            if text:
                texts.append(text)
                items.append(_MLXTimestampItem(text=text, start_time=float(seg.start), end_time=float(seg.end)))
        return items, " ".join(t for t in texts if t)

    def transcribe_with_vad(
        self,
        audio_path: str,
        time_offset: float = 0.0,
        max_chars: int = 60,
        source_lang: str = "en",
        *,
        get_audio_duration=None,
    ) -> List[Segment]:
        """Transcribe audio using VAD to detect speech segments first, then ASR each."""
        if torch is None:
            raise RuntimeError(
                "VAD transcription requires torch. Install it to enable VAD mode "
                "(and torchaudio if you need resampling): pip install torch torchaudio"
            )

        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "VAD transcription requires soundfile. Install it with: pip install soundfile"
            ) from exc
        from silero_vad import get_speech_timestamps

        self.load_vad_model()

        audio_data, sr = sf.read(audio_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if sr != 16000:
            import torchaudio

            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            audio_tensor = waveform.squeeze(0)
        else:
            audio_tensor = torch.from_numpy(audio_data)

        try:
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                return_seconds=True,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                speech_pad_ms=30,
            )
        finally:
            self.unload_vad_model()

        print(f"🔊 VAD detected {len(speech_timestamps)} speech segments")
        # for i, ts in enumerate(speech_timestamps):
        #     print(f"  🔊 Segment {i}: {ts['start']:.2f}s - {ts['end']:.2f}s")

        if not speech_timestamps:
            return []

        all_segments: List[Segment] = []
        speech_audio_paths: List[Tuple[str, float, float]] = []

        try:
            for ts in speech_timestamps:
                seg_start = ts["start"]
                seg_end = ts["end"]
                start_sample = int(seg_start * 16000)
                end_sample = int(seg_end * 16000)
                segment_audio = audio_tensor[start_sample:end_sample].numpy()

                if len(segment_audio) < 1600:
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as seg_tmp:
                    sf.write(seg_tmp.name, segment_audio, 16000)
                    speech_audio_paths.append(
                        (seg_tmp.name, seg_start + time_offset, seg_end - seg_start)
                    )

            del audio_data
            del audio_tensor
            gc.collect()

            if not speech_audio_paths:
                return []

            asr_outputs = []
            self.load_asr_model()
            try:
                for seg_tmp_path, offset, duration in speech_audio_paths:
                    text = self._generate_asr_text(seg_tmp_path, source_lang)
                    if text:
                        asr_outputs.append((seg_tmp_path, offset, duration, text))
            finally:
                self.unload_asr_model()

            if not asr_outputs:
                return []

            aligned_by_path = {}
            if self.enable_align:
                self.load_aligner_model()
                try:
                    for seg_tmp_path, offset, _, text in asr_outputs:
                        aligned_by_path[seg_tmp_path] = self._align_text(
                            seg_tmp_path,
                            text,
                            offset,
                            source_lang,
                        )
                finally:
                    self.unload_aligner_model()

            for seg_tmp_path, offset, duration, text in asr_outputs:
                segs = self._segments_from_text_and_alignment(
                    seg_tmp_path,
                    text,
                    aligned_by_path.get(seg_tmp_path, []),
                    offset,
                    max_chars,
                    get_audio_duration=get_audio_duration,
                    fallback_duration=duration,
                )
                all_segments.extend(segs)
        finally:
            for seg_tmp_path, _, _ in speech_audio_paths:
                if os.path.exists(seg_tmp_path):
                    os.unlink(seg_tmp_path)

        print(f"✅ VAD+ASR produced {len(all_segments)} segments")
        return all_segments
