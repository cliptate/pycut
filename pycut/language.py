"""Spoken-language detection for macOS Apple Silicon."""

from __future__ import annotations

import pycut.config as config


def detect_language(
    audio_path: str,
    model_path: str = config.DEFAULT_LANGUAGE_ID_MODEL,
) -> tuple[str, float]:
    """Return the top ECAPA-TDNN language prediction for 16 kHz mono audio."""
    if not config.is_macos_apple_silicon():
        raise RuntimeError(
            "Automatic language detection requires macOS Apple Silicon; "
            "pass --source-lang on Linux or Windows."
        )

    try:
        from mlx_audio.lid import load
        from mlx_audio.utils import load_audio
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Automatic language detection requires mlx-audio with LID support. "
            f"Original error: {exc}"
        ) from exc

    resolved_model_path = config.resolve_model_path(model_path)
    print(f"🌐 Loading language detection model from {resolved_model_path}...")
    model = load(resolved_model_path)
    try:
        predictions = model.predict(load_audio(audio_path, sample_rate=16000), top_k=1)
        if not predictions:
            raise RuntimeError("Language detection returned no predictions")
        language, confidence = predictions[0]
        return str(language), float(confidence)
    finally:
        del model
        try:
            import mlx.core as mx
        except ImportError:
            pass
        else:
            mx.clear_cache()
