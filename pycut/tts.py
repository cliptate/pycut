from __future__ import annotations

from pathlib import Path
from typing import Optional

import pycut.config as config


def _write_wav(output_path: str, audio, sample_rate: int) -> str:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("TTS output requires numpy and soundfile. Install with: pip install numpy soundfile") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(audio, dtype="float32").reshape(-1)
    sf.write(str(output), waveform, int(sample_rate))
    print(f"💾 TTS audio saved to {output}")
    return str(output)


class MLXTTSHelper:
    def __init__(self, *, model_path: str = config.DEFAULT_MLX_TTS_MODEL):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "macOS TTS requires a working mlx-audio install. "
                f"Original error: {exc}"
            ) from exc

        resolved_model_path = config.resolve_model_path(self.model_path)
        print(f"📝 Loading MLX TTS model from {resolved_model_path}...")
        self.model = load_model(resolved_model_path)
        print("✅ TTS model loaded!")

    def synthesize(
        self,
        *,
        text: str,
        output_path: str,
        voice: str = "Chelsie",
        lang_code: Optional[str] = None,
        speed: Optional[float] = None,
        reference_audio: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        normalize: bool = False,
        **_: object,
    ) -> str:
        self.load_model()
        kwargs = {"voice": voice}
        if lang_code:
            kwargs["lang_code"] = lang_code
        if speed is not None:
            kwargs["speed"] = speed
        if normalize:
            kwargs["normalize"] = normalize
        ref_audio = reference_audio or prompt_audio
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
        if prompt_text:
            kwargs["ref_text"] = prompt_text

        chunks = []
        sample_rate = None
        for result in self.model.generate(text, **kwargs):
            audio = getattr(result, "audio", result)
            chunks.append(audio)
            sample_rate = getattr(result, "sample_rate", sample_rate)

        if not chunks:
            raise RuntimeError("MLX TTS did not produce audio")

        try:
            import mlx.core as mx
        except ImportError:
            mx = None
        import numpy as np

        arrays = []
        for chunk in chunks:
            if mx is not None and hasattr(chunk, "shape"):
                try:
                    arrays.append(np.asarray(chunk, dtype=np.float32))
                    continue
                except TypeError:
                    arrays.append(np.asarray(mx.eval(chunk), dtype=np.float32))
                    continue
            arrays.append(np.asarray(chunk, dtype=np.float32))

        audio = np.concatenate([arr.reshape(-1) for arr in arrays])
        sample_rate = sample_rate or getattr(self.model, "sample_rate", None) or getattr(
            getattr(self.model, "config", None), "sample_rate", None
        ) or 24000
        return _write_wav(output_path, audio, int(sample_rate))


class VoxCPMTTSHelper:
    def __init__(self, *, model_path: str = config.DEFAULT_VOXCPM_TTS_MODEL, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        try:
            from voxcpm import VoxCPM
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Linux/Windows TTS requires a working voxcpm, torch, and torchaudio install. "
                f"Original error: {exc}"
            ) from exc

        resolved_model_path = config.resolve_model_path(self.model_path)
        print(f"📝 Loading VoxCPM TTS model from {resolved_model_path}...")
        self.model = VoxCPM.from_pretrained(
            resolved_model_path,
            load_denoiser=False,
            device=self.device,
        )
        print("✅ TTS model loaded!")

    def synthesize(
        self,
        *,
        text: str,
        output_path: str,
        reference_audio: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = False,
        **_: object,
    ) -> str:
        self.load_model()
        audio = self.model.generate(
            text=text,
            prompt_wav_path=prompt_audio,
            prompt_text=prompt_text,
            reference_wav_path=reference_audio,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
        )
        sample_rate = getattr(getattr(self.model, "tts_model", None), "sample_rate", None) or getattr(
            self.model, "sample_rate", None
        ) or 24000
        return _write_wav(output_path, audio, int(sample_rate))


def create_tts_helper(*, model_path: str | None = None, device: Optional[str] = None):
    config.ensure_supported_runtime()
    backend = config.select_tts_backend()
    if backend == "mlx":
        return MLXTTSHelper(model_path=model_path or config.resolve_default_mlx_tts_model())
    return VoxCPMTTSHelper(model_path=model_path or config.resolve_default_voxcpm_tts_model(), device=device)


def synthesize_text_to_wav(
    *,
    text: str,
    output_path: str,
    model_path: str | None = None,
    voice: str = "Chelsie",
    lang_code: Optional[str] = None,
    speed: Optional[float] = None,
    device: Optional[str] = None,
    reference_audio: Optional[str] = None,
    prompt_audio: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
) -> str:
    helper = create_tts_helper(model_path=model_path, device=device)
    return helper.synthesize(
        text=text,
        output_path=output_path,
        voice=voice,
        lang_code=lang_code,
        speed=speed,
        reference_audio=reference_audio,
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
    )
