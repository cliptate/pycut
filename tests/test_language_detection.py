import sys
import types

import pytest


def test_detect_language_uses_ecapa_top_prediction(monkeypatch):
    import pycut.config as config
    from pycut.language import detect_language

    seen = {}

    class FakeModel:
        def predict(self, audio, top_k):
            seen["predict"] = (audio, top_k)
            return [("zh", 0.92)]

    waveform = object()
    lid_module = types.ModuleType("mlx_audio.lid")
    lid_module.load = lambda model_path: seen.setdefault("model_path", model_path) and FakeModel()
    utils_module = types.ModuleType("mlx_audio.utils")
    utils_module.load_audio = lambda path, sample_rate: seen.setdefault("audio", (path, sample_rate)) and waveform
    core_module = types.ModuleType("mlx.core")
    core_module.clear_cache = lambda: seen.setdefault("cache_cleared", True)
    mlx_module = types.ModuleType("mlx")
    mlx_module.core = core_module

    monkeypatch.setitem(sys.modules, "mlx_audio.lid", lid_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", utils_module)
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", core_module)
    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    language, confidence = detect_language("audio.wav")

    assert (language, confidence) == ("zh", pytest.approx(0.92))
    assert seen == {
        "model_path": config.DEFAULT_LANGUAGE_ID_MODEL,
        "audio": ("audio.wav", 16000),
        "predict": (waveform, 1),
        "cache_cleared": True,
    }


def test_detect_language_requires_apple_silicon(monkeypatch):
    import pycut.config as config
    from pycut.language import detect_language

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="pass --source-lang"):
        detect_language("audio.wav")


def test_media_workflow_detects_language_before_transcribing(tmp_path):
    from pycut.media_job import MediaJob
    from pycut.media_workflow import MediaJobWorkflow, WorkflowAdapters
    from pycut.utils import Segment

    events = []

    def extract_audio(video_path, audio_path):
        events.append(("extract", video_path, audio_path))
        return audio_path

    def detect(audio_path):
        events.append(("detect", audio_path))
        return "zh", 0.92

    def transcribe(audio_path, *, orientation, source_lang):
        events.append(("transcribe", audio_path, orientation, source_lang))
        return [Segment(start=0.0, end=1.0, text="你好")]

    adapters = WorkflowAdapters(
        extract_audio=extract_audio,
        detect_language=detect,
        transcribe_audio=transcribe,
        unload_asr_model=lambda: events.append(("unload",)),
        generate_ass_subtitle=lambda *_args, **_kwargs: pytest.fail("ASS should not run"),
        generate_fcpxml=lambda *_args, **_kwargs: pytest.fail("FCPXML should not run"),
        render_video_with_subtitles_complex=lambda *_args, **_kwargs: pytest.fail("render should not run"),
    )
    job = MediaJob(
        video_path=str(tmp_path / "demo.mp4"),
        output_dir=str(tmp_path / "output"),
        source_lang=None,
        output_formats=["json"],
    )

    MediaJobWorkflow(job, adapters=adapters, segment_duration=300).run()

    audio_path = events[0][2]
    assert job.source_lang == "zh"
    assert events == [
        ("extract", str(tmp_path / "demo.mp4"), audio_path),
        ("detect", audio_path),
        ("transcribe", audio_path, "landscape", "zh"),
        ("unload",),
    ]


def test_cli_omitted_source_language_defers_model_selection(monkeypatch):
    import pycut.cli as cli

    seen = {}

    class FakeVideoClipper:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def process_video(self, **kwargs):
            seen["process"] = kwargs
            return {}

    monkeypatch.setattr(cli, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(cli, "_expand_video_inputs", lambda _inputs: ["/tmp/input.wav"])
    monkeypatch.setattr(cli, "_resolve_default_aligner_model", lambda: "aligner")

    cli.main(["/tmp/input.wav"])

    assert seen["init"]["asr_model_path"] is None
    assert seen["process"]["source_lang"] is None


def test_video_clipper_detects_language_before_selecting_default_asr(monkeypatch, tmp_path):
    import pycut.config as config
    from pycut.clipper import VideoClipper
    from pycut.utils import Segment

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    clipper = VideoClipper()
    seen = {}

    monkeypatch.setattr(clipper, "extract_audio", lambda _video, audio: audio)
    monkeypatch.setattr(clipper, "detect_language", lambda _audio: ("zh", 0.92), raising=False)

    def transcribe(_audio, *, time_offset=0.0, max_chars, source_lang):
        seen.update(
            source_lang=source_lang,
            asr_model_path=clipper.asr_model_path,
            time_offset=time_offset,
            max_chars=max_chars,
        )
        return [Segment(start=0.0, end=1.0, text="你好")]

    monkeypatch.setattr(clipper, "_transcribe_with_vad", transcribe)

    clipper.process_video(
        video_path=str(tmp_path / "demo.mp4"),
        output_dir=str(tmp_path / "output"),
        source_lang=None,
        output_formats=["json"],
    )

    assert seen["source_lang"] == "zh"
    assert seen["asr_model_path"] == config.DEFAULT_CHINESE_ASR_MODEL
