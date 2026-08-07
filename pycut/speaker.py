"""Speaker diarization with MLX Audio Sortformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pycut.config as config


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: int


def diarize_speakers(
    audio_path: str,
    *,
    model_path: str = config.DEFAULT_SPEAKER_DIARIZATION_MODEL,
    threshold: float = 0.5,
) -> list[SpeakerTurn]:
    """Return Sortformer speaker turns for an audio file."""
    if not config.is_macos_apple_silicon():
        raise RuntimeError("Speaker diarization requires macOS Apple Silicon")
    if not Path(audio_path).is_file():
        raise RuntimeError(f"Speaker diarization audio not found: {audio_path}")

    try:
        from mlx_audio.vad import load
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Speaker diarization requires mlx-audio with Sortformer support. "
            f"Original error: {exc}"
        ) from exc

    resolved_model_path = config.resolve_model_path(model_path)
    print(f"🗣️  Loading speaker diarization model from {resolved_model_path}...")
    model = load(resolved_model_path)
    try:
        result = model.generate(
            audio_path,
            threshold=threshold,
            min_duration=0.25,
            merge_gap=0.2,
            verbose=False,
        )
        return [
            SpeakerTurn(float(segment.start), float(segment.end), int(segment.speaker))
            for segment in result.segments
            if float(segment.end) > float(segment.start)
        ]
    finally:
        del model
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            mx.clear_cache()
