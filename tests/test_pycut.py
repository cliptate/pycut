#!/usr/bin/env python3
"""
Quick test script for clipping.py functionality
Tests individual components without requiring a full video
"""

import inspect
import os
import platform
import sys
from pathlib import Path
import tomllib
import builtins
import types

import pytest

# Add repository root to path


def test_runtime_guard_accepts_linux_and_rejects_unsupported_systems(monkeypatch):
    """VideoClipper should support Linux/Windows and reject unsupported runtimes."""
    import pycut.config as config
    from pycut.clipper import VideoClipper

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")

    clipper = VideoClipper()
    assert clipper.asr_backend == "qwen"

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="macOS Apple Silicon, Linux, and Windows"):
        VideoClipper()


def test_google_translator_translate_bulk_returns_translated_texts(monkeypatch):
    """GoogleTranslator should batch-translate and preserve requested languages."""
    import pycut.translation as translation

    seen = {}

    class FakeTranslatorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def translate(self, texts, src, dest):
            seen["texts"] = list(texts)
            seen["src"] = src
            seen["dest"] = dest
            return [types.SimpleNamespace(text=f"{text}-{dest}") for text in texts]

    monkeypatch.setattr(translation, "Translator", FakeTranslatorClient, raising=False)

    translator = translation.GoogleTranslator()

    assert translator.translate_bulk(["hello", "world"], source_lang="en", target_lang="fr") == [
        "hello-fr",
        "world-fr",
    ]
    assert seen == {"texts": ["hello", "world"], "src": "en", "dest": "fr"}


def test_google_translator_translate_bulk_exits_after_three_client_errors(monkeypatch):
    """GoogleTranslator should exit after 3 consecutive client errors."""
    import pycut.translation as translation

    class FakeTranslatorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def translate(self, texts, src, dest):
            raise RuntimeError("boom")

    monkeypatch.setattr(translation, "Translator", FakeTranslatorClient, raising=False)

    translator = translation.GoogleTranslator()
    texts = ["hello", "world"]

    with pytest.raises(SystemExit) as exc_info:
        translator.translate_bulk(texts, source_lang="en", target_lang="fr")
    assert exc_info.value.code == 1


def test_video_clipper_uses_google_translator_service(monkeypatch):
    """VideoClipper should delegate translation to the injected translator service."""
    import pycut.config as config
    from pycut.clipper import VideoClipper

    class FakeTranslatorService:
        def translate_bulk(self, texts, source_lang="zh", target_lang="en"):
            assert texts == ["hello"]
            assert source_lang == "en"
            assert target_lang == "zh-cn"
            return ["你好"]

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    clipper = VideoClipper(translator=FakeTranslatorService())

    assert not hasattr(clipper, "runtime_backend")
    assert not hasattr(clipper, "translate_model_path")
    assert clipper.translate_text("hello", source_lang="en", target_lang="zh-cn") == "你好"


def test_asr_module_exposes_system_asr_helpers():
    """ASR helpers should expose both system-selected backends."""
    import pycut.asr as asr

    assert hasattr(asr, "MLXASRHelper")
    assert hasattr(asr, "QwenASRHelper")
    assert hasattr(asr, "load_mlx_stt_model")
    assert hasattr(asr, "load_qwen_asr_model")


def test_package_metadata_declares_runtime_dependencies_used_by_source():
    """Installed CLI environments must include direct runtime deps used by source."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]
    dependency_names = {dep.split(";")[0].split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip() for dep in dependencies}

    assert "numpy" in dependency_names
    assert "soundfile" in dependency_names
    assert "qwen-asr" in dependency_names
    assert "voxcpm" in dependency_names


def _write_hf_snapshot(cache_root, repo_id, commit="abc123", files=None):
    repo_dir = cache_root / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / commit
    (repo_dir / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")
    for name, content in (files or {}).items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return snapshot


def test_resolve_default_qwen_asr_prefers_complete_cached_snapshot(monkeypatch, tmp_path):
    """Incomplete 1.7B cache should fall back to the complete cached 0.6B model."""
    import pycut.config as config

    cache_root = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    _write_hf_snapshot(
        cache_root,
        "Qwen/Qwen3-ASR-1.7B",
        files={
            "config.json": b"{}",
            "model.safetensors.index.json": b'{"weight_map": {"a": "model-00001-of-00002.safetensors"}}',
        },
    )
    fallback_snapshot = _write_hf_snapshot(
        cache_root,
        "Qwen/Qwen3-ASR-0.6B",
        files={
            "config.json": b"{}",
            "model.safetensors": b"weights",
        },
    )

    assert config.resolve_hf_cached_snapshot("Qwen/Qwen3-ASR-1.7B") is None
    assert config.resolve_default_qwen_asr_model() == str(fallback_snapshot)


def test_resolve_model_path_returns_complete_cached_snapshot(monkeypatch, tmp_path):
    """Repo ids should resolve to local snapshot paths when the cache is complete."""
    import pycut.config as config

    cache_root = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    snapshot = _write_hf_snapshot(
        cache_root,
        "openbmb/VoxCPM2",
        files={
            "config.json": b"{}",
            "model.safetensors": b"weights",
        },
    )

    assert config.resolve_model_path("openbmb/VoxCPM2") == str(snapshot)


def test_expand_video_inputs_treats_existing_bracketed_media_path_as_literal_file(tmp_path):
    """Existing media files with [] in their names should not be treated as glob patterns."""
    from pycut.video_io import _expand_video_inputs

    media_path = tmp_path / "Nvidia CEO Jensen Huang on AI, Musk and Trump [c-XAL2oYelI].m4a"
    media_path.write_bytes(b"fake audio")

    assert _expand_video_inputs([str(media_path)]) == [str(media_path.resolve())]


def test_asr_loader_surface_omits_legacy_runtime_knobs():
    """The extracted ASR loader surface should not expose legacy runtime knobs."""
    import pycut.asr as asr

    legacy_knobs = {
        "device",
        "gpu_memory_utilization",
        "max_model_len",
        "runtime_backend",
        "translate_model_path",
        "use_vllm",
    }

    assert asr.__all__ == ["MLXASRHelper", "QwenASRHelper", "load_mlx_stt_model", "load_qwen_asr_model"]
    assert legacy_knobs.isdisjoint(asr.__all__)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.QwenASRHelper).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.load_mlx_stt_model).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.load_qwen_asr_model).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper.load_models).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper.load_vad_model).parameters)

    with pytest.raises(TypeError, match="device"):
        asr.load_mlx_stt_model("mlx-community/whisper-tiny", device="cpu")

    with pytest.raises(TypeError, match="device"):
        asr.load_qwen_asr_model("Qwen/Qwen3-ASR-1.7B", device="cpu")

    with pytest.raises(TypeError, match="use_vllm"):
        asr.MLXASRHelper(
            asr_model_path="mlx-community/parakeet-tdt-0.6b-v2",
            aligner_model_path="mlx-community/Qwen3-Forced-Aligner-0.6B",
            use_vllm=True,
        )


def test_video_clipper_delegates_asr_loading_to_helper(monkeypatch):
    """VideoClipper should delegate ASR model loading to the extracted helper."""
    import pycut.config as config
    import pycut.clipper as clipper_module

    seen = {}

    class FakeASRHelper:
        def __init__(self, *, asr_model_path, aligner_model_path, filter_fillers, enable_align):
            seen["init"] = {
                "asr_model_path": asr_model_path,
                "aligner_model_path": aligner_model_path,
                "filter_fillers": filter_fillers,
                "enable_align": enable_align,
            }

        def load_models(self):
            seen["load_models"] = True

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(clipper_module, "MLXASRHelper", FakeASRHelper, raising=False)

    vc = clipper_module.VideoClipper()
    vc._load_asr_model()

    assert isinstance(vc.asr_helper, FakeASRHelper)
    assert seen["init"]["filter_fillers"] is True
    assert seen["init"]["enable_align"] is True
    assert seen["load_models"] is True


def test_transcribe_audio_runs_vad_on_full_audio_before_transcript_chunking(monkeypatch):
    """Long audio should not be pre-split before VAD/ASR/alignment."""
    import pycut.clipper as clipper_module
    from pycut.utils import Segment

    clipper = clipper_module.VideoClipper.__new__(clipper_module.VideoClipper)
    clipper.max_chars = 30
    clipper.segment_duration = 300

    calls = []

    def fake_split_audio(*args, **kwargs):
        raise AssertionError("audio should not be split before VAD/ASR")

    def fake_transcribe_with_vad(audio_path, time_offset=0.0, max_chars=60, source_lang="en"):
        calls.append(
            {
                "audio_path": audio_path,
                "time_offset": time_offset,
                "max_chars": max_chars,
                "source_lang": source_lang,
            }
        )
        return [Segment(start=1.0, end=2.0, text="hello", words=[])]

    monkeypatch.setattr(clipper, "_load_asr_model", lambda: None)
    monkeypatch.setattr(clipper, "get_audio_duration", lambda _: 600.0)
    monkeypatch.setattr(clipper, "split_audio", fake_split_audio)
    monkeypatch.setattr(clipper, "_transcribe_with_vad", fake_transcribe_with_vad)

    segments = clipper.transcribe_audio("full.wav", orientation="landscape", source_lang="zh")

    assert segments == [Segment(start=1.0, end=2.0, text="hello", words=[])]
    assert calls == [
        {
            "audio_path": "full.wav",
            "time_offset": 0.0,
            "max_chars": 30,
            "source_lang": "zh",
        }
    ]


def test_process_video_unloads_asr_before_postprocessing(monkeypatch, tmp_path):
    """ASR should be released before transcript export consumers."""
    import pycut.clipper as clipper_module
    from pycut.utils import Segment

    clipper = clipper_module.VideoClipper.__new__(clipper_module.VideoClipper)
    clipper.segment_duration = 300
    clipper.filter_fillers = True

    events = []
    raw_segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]
    monkeypatch.setattr(clipper, "extract_audio", lambda video_path, output_path: events.append("extract"))

    def fake_transcribe(audio_path, orientation="landscape", source_lang="en"):
        events.append("transcribe")
        return raw_segments

    def fake_unload():
        events.append("unload")

    def fake_filter(segments, filter_empty_segments=True):
        events.append("filter")
        return list(segments)

    monkeypatch.setattr(clipper, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(clipper, "_unload_asr_model", fake_unload)
    monkeypatch.setattr(clipper, "_filter_subtitle_segments", fake_filter)

    result = clipper.process_video(
        "demo.mp4",
        str(tmp_path),
        output_formats=["json"],
        margin_left=0.0,
        margin_right=0.0,
    )

    assert events == ["extract", "transcribe", "unload"]
    assert result == {"transcript": str(tmp_path / "demo_transcript.json")}


def test_mlx_asr_helper_unload_releases_asr_aligner_and_vad():
    """Unload should release every model used by VAD+ASR transcription."""
    import pycut.asr as asr

    helper = asr.MLXASRHelper(
        asr_model_path="fake-asr",
        aligner_model_path="fake-aligner",
        filter_fillers=True,
        enable_align=True,
    )
    helper.asr_model = object()
    helper._mlx_aligner = object()
    helper.vad_model = object()

    helper.unload_models()

    assert helper.asr_model is None
    assert helper._mlx_aligner is None
    assert helper.vad_model is None


def test_mlx_asr_helper_transcribe_audio_loads_one_mlx_model_at_a_time(monkeypatch):
    """ASR should be released before the aligner is loaded."""
    import pycut.asr as asr

    events = []

    class FakeASRModel:
        def generate(self, audio_path, language="en"):
            events.append("asr_generate")
            return types.SimpleNamespace(text="hello world")

    class FakeAligner:
        def generate(self, audio_path, text, language="en"):
            events.append("align_generate")
            return [
                types.SimpleNamespace(text="hello", start_time=0.0, end_time=0.5),
                types.SimpleNamespace(text="world", start_time=0.5, end_time=1.0),
            ]

    def fake_load(model_name):
        events.append(f"load:{model_name}")
        if model_name == "fake-asr":
            return FakeASRModel()
        if model_name == "fake-aligner":
            return FakeAligner()
        raise AssertionError(model_name)

    monkeypatch.setattr(asr, "load_mlx_stt_model", fake_load)

    helper = asr.MLXASRHelper(
        asr_model_path="fake-asr",
        aligner_model_path="fake-aligner",
        filter_fillers=True,
        enable_align=True,
    )

    segments = helper.transcribe_audio("fake.wav", get_audio_duration=lambda _: 1.0)

    assert [seg.text for seg in segments] == ["hello world"]
    assert helper.asr_model is None
    assert helper._mlx_aligner is None
    assert events == [
        "load:fake-asr",
        "asr_generate",
        "load:fake-aligner",
        "align_generate",
    ]


def test_transcribe_with_vad_batches_each_model_phase_sequentially(monkeypatch):
    """VAD, ASR, and aligner should run as separate load/process/unload phases."""
    import numpy as np
    import pycut.asr as asr

    events = []

    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return FakeTensor(self.data[key])

        def numpy(self):
            return self.data

    fake_torch = types.SimpleNamespace(from_numpy=lambda data: FakeTensor(data))
    fake_soundfile = types.SimpleNamespace(
        read=lambda *args, **kwargs: (np.ones(16000, dtype=np.float32), 16000),
        write=lambda path, data, sr: Path(path).write_bytes(b"wav"),
    )

    def fake_get_speech_timestamps(audio_tensor, vad_model, **kwargs):
        events.append("vad_process")
        return [
            {"start": 0.0, "end": 0.2},
            {"start": 0.5, "end": 0.7},
        ]

    fake_silero_vad = types.SimpleNamespace(get_speech_timestamps=fake_get_speech_timestamps)

    monkeypatch.setattr(asr, "torch", fake_torch, raising=False)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "silero_vad", fake_silero_vad)

    helper = asr.MLXASRHelper(
        asr_model_path="fake-asr",
        aligner_model_path="fake-aligner",
        filter_fillers=True,
        enable_align=True,
    )

    def fake_load_vad_model():
        events.append("load:vad")
        helper.vad_model = object()

    def fake_unload_vad_model():
        events.append("unload:vad")
        helper.vad_model = None

    def fake_load_asr_model():
        assert helper.vad_model is None
        assert helper._mlx_aligner is None
        events.append("load:asr")
        helper.asr_model = object()

    def fake_generate_asr_text(audio_path, source_lang):
        assert helper.vad_model is None
        assert helper.asr_model is not None
        assert helper._mlx_aligner is None
        events.append("asr_process")
        return "hello world"

    def fake_unload_asr_model():
        events.append("unload:asr")
        helper.asr_model = None

    def fake_load_aligner_model():
        assert helper.vad_model is None
        assert helper.asr_model is None
        events.append("load:aligner")
        helper._mlx_aligner = object()

    def fake_align_text(audio_path, text, time_offset, source_lang):
        assert helper.vad_model is None
        assert helper.asr_model is None
        assert helper._mlx_aligner is not None
        events.append("align_process")
        return [
            asr._MLXTimestampItem("hello", time_offset, time_offset + 0.1),
            asr._MLXTimestampItem("world", time_offset + 0.1, time_offset + 0.2),
        ]

    def fake_unload_aligner_model():
        events.append("unload:aligner")
        helper._mlx_aligner = None

    monkeypatch.setattr(helper, "load_vad_model", fake_load_vad_model)
    monkeypatch.setattr(helper, "unload_vad_model", fake_unload_vad_model)
    monkeypatch.setattr(helper, "load_asr_model", fake_load_asr_model)
    monkeypatch.setattr(helper, "_generate_asr_text", fake_generate_asr_text)
    monkeypatch.setattr(helper, "unload_asr_model", fake_unload_asr_model)
    monkeypatch.setattr(helper, "load_aligner_model", fake_load_aligner_model)
    monkeypatch.setattr(helper, "_align_text", fake_align_text)
    monkeypatch.setattr(helper, "unload_aligner_model", fake_unload_aligner_model)

    segments = helper.transcribe_with_vad("fake.wav", get_audio_duration=lambda _: 0.2)

    assert [seg.text for seg in segments] == ["hello world", "hello world"]
    assert events == [
        "load:vad",
        "vad_process",
        "unload:vad",
        "load:asr",
        "asr_process",
        "asr_process",
        "unload:asr",
        "load:aligner",
        "align_process",
        "align_process",
        "unload:aligner",
    ]


def test_mlx_asr_helper_skips_alignment_when_disabled(monkeypatch):
    """Disabling align should skip aligner generation and fall back to a single timed segment."""
    import pycut.asr as asr

    calls = {"aligner_generate": 0}

    class FakeASRModel:
        def generate(self, audio_path, language="en"):
            return types.SimpleNamespace(text="hello world")

    class FakeAligner:
        def generate(self, *args, **kwargs):
            calls["aligner_generate"] += 1
            return [types.SimpleNamespace(text="hello", start_time=0.0, end_time=0.5)]

    helper = asr.MLXASRHelper(
        asr_model_path="fake-asr",
        aligner_model_path="fake-aligner",
        filter_fillers=True,
        enable_align=False,
    )
    helper.asr_model = FakeASRModel()
    helper._mlx_aligner = FakeAligner()
    monkeypatch.setattr(helper, "load_models", lambda: None)

    segments = helper.transcribe_audio(
        "fake.wav",
        time_offset=1.25,
        source_lang="en",
        get_audio_duration=lambda _: 2.5,
    )

    assert calls["aligner_generate"] == 0
    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].start == 1.25
    assert segments[0].end == 3.75
    assert segments[0].words == []


def test_qwen_asr_helper_generates_text_and_alignment(monkeypatch):
    """Qwen helper should adapt Qwen transcribe results into shared segments."""
    import pycut.asr as asr

    events = []

    class FakeQwenModel:
        def __init__(self, with_aligner=False):
            self.with_aligner = with_aligner

        def transcribe(self, *, audio, language=None, return_time_stamps=False):
            events.append(
                {
                    "audio": audio,
                    "language": language,
                    "return_time_stamps": return_time_stamps,
                    "with_aligner": self.with_aligner,
                }
            )
            if return_time_stamps:
                stamps = types.SimpleNamespace(
                    items=[
                        types.SimpleNamespace(text="hello", start_time=0.0, end_time=0.4),
                        types.SimpleNamespace(text="world", start_time=0.4, end_time=0.9),
                    ]
                )
                return [types.SimpleNamespace(text="hello world", time_stamps=stamps)]
            return [types.SimpleNamespace(text="hello world")]

    def fake_load(model_name, **kwargs):
        return FakeQwenModel(with_aligner=bool(kwargs.get("forced_aligner")))

    monkeypatch.setattr(asr, "load_qwen_asr_model", fake_load)

    helper = asr.QwenASRHelper(
        asr_model_path="Qwen/Qwen3-ASR-1.7B",
        aligner_model_path="Qwen/Qwen3-ForcedAligner-0.6B",
        filter_fillers=True,
        enable_align=True,
    )

    segments = helper.transcribe_audio("fake.wav", source_lang="en", get_audio_duration=lambda _: 1.0)

    assert [seg.text for seg in segments] == ["hello world"]
    assert segments[0].words[0]["word"] == "hello"
    assert helper.asr_model is None
    assert helper._qwen_aligner is None
    assert events == [
        {"audio": "fake.wav", "language": "English", "return_time_stamps": False, "with_aligner": False},
        {"audio": "fake.wav", "language": "English", "return_time_stamps": True, "with_aligner": True},
    ]


def test_video_clipper_signature_omits_legacy_backend_device_options():
    """Mac-only runtime should not expose legacy backend/device knobs."""
    from pycut.clipper import VideoClipper

    params = inspect.signature(VideoClipper).parameters

    assert "translator" in params
    assert "translate_model_path" not in params
    assert "device" not in params
    assert "use_vllm" not in params
    assert "gpu_memory_utilization" not in params
    assert "max_model_len" not in params


def test_video_clipper_instance_omits_legacy_device_state(monkeypatch, capsys):
    """Mac-only runtime should not retain legacy device state."""
    import pycut.config as config
    from pycut.clipper import VideoClipper

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    clipper = VideoClipper()

    assert not hasattr(clipper, "device")
    assert not hasattr(clipper, "_mlx_translate_model")
    assert not hasattr(clipper, "_mlx_translate_tokenizer")
    init_output = capsys.readouterr().out
    assert "on cpu" not in init_output


def test_internal_methods_omit_legacy_backend_device_references():
    """Mac-only runtime should not keep legacy standard/CUDA translation references."""
    from pycut.clipper import VideoClipper
    from pycut.translation import GoogleTranslator

    legacy_free_methods = [
        inspect.getsource(VideoClipper._select_video_encoder),
        inspect.getsource(VideoClipper.translate_text),
        inspect.getsource(VideoClipper.translate_texts_bulk),
        inspect.getsource(GoogleTranslator.translate_bulk),
    ]
    combined_source = "\n".join(legacy_free_methods)

    assert "_load_translation_model" not in combined_source
    assert "_unload_translation_model" not in combined_source
    assert "translate_pipe" not in combined_source
    assert "cuda" not in combined_source.lower()
    assert "h264_nvenc" not in combined_source
    assert "Standard translation backend" not in combined_source


def test_main_module_omits_legacy_runtime_asr_paths():
    """main.py should no longer carry legacy runtime selection or inline ASR helpers."""
    import pycut.cli as pycut_main_module

    source = inspect.getsource(pycut_main_module)

    assert "_select_runtime_backend" not in source
    assert "runtime_backend ==" not in source
    assert "Qwen3ASRModel" not in source
    assert "vllm" not in source.lower()
    assert "_mlx_stt_load" not in source
    assert "_MLXTimestampItem" not in source


def test_cli_help_omits_legacy_backend_device_options(monkeypatch, capsys):
    """CLI help should not advertise removed backend/device options."""
    import pycut.cli as pycut_main_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        pycut_main_module.main()

    help_output = capsys.readouterr().out
    assert "--device" not in help_output
    assert "--translate-model" not in help_output
    assert "--use-vllm" not in help_output
    assert "--gpu-memory-utilization" not in help_output
    assert "--max-model-len" not in help_output


def test_cli_import_does_not_load_video_pipeline():
    """Importing pycut.cli should keep the video/ASR pipeline out of TTS-only processes."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pycut.cli, sys; print('pycut.clipper' in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_tts_cli_synthesizes_inline_text(monkeypatch, tmp_path):
    """pycut tts should route text generation through the TTS helper."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config, "resolve_default_voxcpm_tts_model", lambda: config.DEFAULT_VOXCPM_TTS_MODEL)

    seen = {}

    def fake_synthesize_text_to_wav(**kwargs):
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", fake_synthesize_text_to_wav)

    output = tmp_path / "voice.wav"
    result = cli_module.main(["tts", "--text", "hello", "--output", str(output)])

    assert result == {"tts": str(output)}
    assert seen["text"] == "hello"
    assert seen["model_path"] == config.DEFAULT_VOXCPM_TTS_MODEL
    assert seen["join_audio"] is True


def test_tts_cli_can_disable_join_audio(monkeypatch, tmp_path):
    """pycut tts should expose an escape hatch for MLX joined audio generation."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config, "resolve_default_mlx_tts_model", lambda: config.DEFAULT_MLX_TTS_MODEL)

    seen_by_option = {}

    def fake_synthesize_text_to_wav(**kwargs):
        seen_by_option[kwargs["output_path"]] = dict(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", fake_synthesize_text_to_wav)

    for option in ("--no-join-audio", "--no-join_audio"):
        output = tmp_path / f"{option.removeprefix('--no-')}.wav"
        result = cli_module.main(["tts", "--text", "hello", "--output", str(output), option])

        assert result == {"tts": str(output)}
        assert seen_by_option[str(output)]["join_audio"] is False


def test_tts_cli_passes_mlx_generation_controls(monkeypatch, tmp_path):
    """pycut tts should expose MLX segmentation controls for long scripts."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config, "resolve_default_mlx_tts_model", lambda: config.DEFAULT_MLX_TTS_MODEL)

    seen = {}

    def fake_synthesize_text_to_wav(**kwargs):
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", fake_synthesize_text_to_wav)

    output = tmp_path / "voice.wav"
    result = cli_module.main([
        "tts",
        "--text",
        "line 1\nline 2",
        "--output",
        str(output),
        "--split-pattern",
        "\\n",
        "--max-tokens",
        "4096",
        "--verbose",
    ])

    assert result == {"tts": str(output)}
    assert seen["split_pattern"] == "\n"
    assert seen["max_tokens"] == 4096
    assert seen["verbose"] is True

    seen.clear()
    output = tmp_path / "voice_alias.wav"
    result = cli_module.main([
        "tts",
        "--text",
        "line 1\nline 2",
        "--output",
        str(output),
        "--split_pattern",
        "\\n",
        "--max_tokens",
        "4096",
    ])

    assert result == {"tts": str(output)}
    assert seen["split_pattern"] == "\n"
    assert seen["max_tokens"] == 4096


def test_tts_console_main_returns_zero_on_success(monkeypatch, tmp_path, capsys):
    """The console entry point should not pass a success result dict to sys.exit."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config, "resolve_default_voxcpm_tts_model", lambda: config.DEFAULT_VOXCPM_TTS_MODEL)
    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", lambda **kwargs: kwargs["output_path"])

    output = tmp_path / "voice.wav"

    assert cli_module.console_main(["tts", "--text", "hello", "--output", str(output)]) == 0
    assert capsys.readouterr().err == ""


def test_tts_console_main_reports_runtime_errors_without_traceback(monkeypatch, tmp_path, capsys):
    """TTS backend failures should produce a concise CLI error and exit non-zero."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config, "resolve_default_voxcpm_tts_model", lambda: config.DEFAULT_VOXCPM_TTS_MODEL)

    def fake_synthesize_text_to_wav(**kwargs):
        raise RuntimeError("TTS backend failed")

    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", fake_synthesize_text_to_wav)

    output = tmp_path / "voice.wav"

    assert cli_module.console_main(["tts", "--text", "hello", "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert "TTS backend failed" in captured.err
    assert "Traceback" not in captured.err


def test_clip_console_main_returns_zero_after_successful_processing(monkeypatch):
    """Writing transcript/output results should not make the console script exit non-zero."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    class FakeVideoClipper:
        def __init__(self, **kwargs):
            pass

        def process_video(self, **kwargs):
            return {"transcript": "/tmp/demo_transcript.json"}

    monkeypatch.setattr(cli_module, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(cli_module, "_expand_video_inputs", lambda inputs: ["/tmp/demo.mp4"])

    assert cli_module.console_main(["--format", "json", "/tmp/demo.mp4"]) == 0


def test_tts_cli_synthesizes_text_file(monkeypatch, tmp_path):
    """pycut tts should read UTF-8 text files."""
    import pycut.cli as cli_module
    import pycut.config as config

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(config, "resolve_default_mlx_tts_model", lambda: config.DEFAULT_MLX_TTS_MODEL)

    text_file = tmp_path / "input.txt"
    text_file.write_text("你好\n", encoding="utf-8")
    output = tmp_path / "voice.wav"
    seen = {}

    def fake_synthesize_text_to_wav(**kwargs):
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(cli_module, "synthesize_text_to_wav", fake_synthesize_text_to_wav)

    result = cli_module.main(["tts", "--text-file", str(text_file), "--output", str(output)])

    assert result == {"tts": str(output)}
    assert seen["text"] == "你好"
    assert seen["model_path"] == config.DEFAULT_MLX_TTS_MODEL


def test_mlx_tts_helper_joins_chunks_and_writes_wav(monkeypatch):
    """MLX TTS helper should join generated chunks into one WAV."""
    import numpy as np
    import pycut.tts as tts

    class FakeResult:
        def __init__(self, audio):
            self.audio = np.asarray(audio, dtype=np.float32)
            self.sample_rate = 22050

    class FakeModel:
        def generate(self, text, **kwargs):
            assert text == "hello"
            assert kwargs["voice"] == "Chelsie"
            yield FakeResult([0.1, 0.2])
            yield FakeResult([0.3])

    written = {}
    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_write_wav", lambda output_path, audio, sample_rate: written.update({
        "output_path": output_path,
        "audio": list(audio),
        "sample_rate": sample_rate,
    }) or output_path)

    helper = tts.MLXTTSHelper(model_path="fake")

    assert helper.synthesize(text="hello", output_path="out.wav", join_audio=False) == "out.wav"
    assert written == {"output_path": "out.wav", "audio": [0.1, 0.2, 0.3], "sample_rate": 22050}


def test_mlx_tts_helper_uses_mlx_audio_join_audio(monkeypatch, tmp_path):
    """MLX TTS helper should use mlx_audio's joined chunk writer."""
    import pycut.tts as tts

    output = tmp_path / "joined.wav"
    seen = {}

    class FakeAudio:
        def __array__(self, *args, **kwargs):
            raise AssertionError("join_audio should not convert chunks through numpy")

    class FakeResult:
        def __init__(self, sample_rate):
            self.audio = FakeAudio()
            self.sample_rate = sample_rate

    class FakeModel:
        def generate(self, text, **kwargs):
            seen["text"] = text
            seen["kwargs"] = kwargs
            yield FakeResult(22050)
            yield FakeResult(22050)

    def fake_write_joined_audio(file_name, audio_chunks, sample_rate, audio_format):
        seen["file_name"] = file_name
        seen["audio_chunks"] = list(audio_chunks)
        seen["sample_rate"] = sample_rate
        seen["audio_format"] = audio_format
        output.write_bytes(b"RIFF")

    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_load_mlx_write_joined_audio", lambda: fake_write_joined_audio, raising=False)
    monkeypatch.setattr(
        tts,
        "_load_mlx_generate_audio",
        lambda: (_ for _ in ()).throw(AssertionError("should not use generate_audio wrapper")),
        raising=False,
    )

    helper = tts.MLXTTSHelper(model_path="fake")

    assert helper.synthesize(text="hello", output_path=str(output), voice="Chelsie") == str(output)
    assert seen["text"] == "hello"
    assert seen["kwargs"]["voice"] == "Chelsie"
    assert "lang_code" not in seen["kwargs"]
    assert "speed" not in seen["kwargs"]
    assert seen["file_name"] == str(output)
    assert len(seen["audio_chunks"]) == 2
    assert seen["sample_rate"] == 22050
    assert seen["audio_format"] == "wav"


def test_mlx_tts_helper_passes_split_controls_to_joined_generation(monkeypatch, tmp_path):
    """MLX joined generation should preserve explicit split and token controls."""
    import pycut.tts as tts

    output = tmp_path / "joined.wav"
    seen = {}

    class FakeModel:
        def generate(self, text, **kwargs):
            seen.update(kwargs)
            yield types.SimpleNamespace(audio=b"audio", sample_rate=22050, segment_idx=0)
            yield types.SimpleNamespace(audio=b"audio", sample_rate=22050, segment_idx=1)

    def fake_write_joined_audio(file_name, audio_chunks, sample_rate, audio_format):
        output.write_bytes(b"RIFF")

    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_load_mlx_write_joined_audio", lambda: fake_write_joined_audio)

    helper = tts.MLXTTSHelper(model_path="fake")

    assert helper.synthesize(
        text="line 1\nline 2",
        output_path=str(output),
        split_pattern="\n",
        max_tokens=4096,
        verbose=True,
    ) == str(output)
    assert seen["split_pattern"] == "\n"
    assert seen["max_tokens"] == 4096
    assert seen["verbose"] is True


def test_mlx_tts_helper_adds_sample_rate_for_mlx_audio_join_audio(monkeypatch, tmp_path):
    """MLX joined output should work when the loaded model stores sample rate in config."""
    import pycut.tts as tts

    output = tmp_path / "joined.wav"
    seen = {}

    class FakeModel:
        config = types.SimpleNamespace(sample_rate=22050)

        def generate(self, text, **kwargs):
            yield types.SimpleNamespace(audio=b"audio")

    def fake_write_joined_audio(file_name, audio_chunks, sample_rate, audio_format):
        seen["sample_rate"] = sample_rate
        output.write_bytes(b"RIFF")

    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_load_mlx_write_joined_audio", lambda: fake_write_joined_audio)

    helper = tts.MLXTTSHelper(model_path="fake")

    assert helper.synthesize(text="hello", output_path=str(output)) == str(output)
    assert seen["sample_rate"] == 22050


def test_mlx_tts_helper_reports_mlx_join_audio_failures(monkeypatch, tmp_path):
    """MLX joined output failures should not silently fall back to pycut chunk writing."""
    import pycut.tts as tts

    output = tmp_path / "joined.wav"

    class FakeModel:
        sample_rate = 22050

        def generate(self, text, **kwargs):
            yield types.SimpleNamespace(audio=b"audio")

    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_load_mlx_write_joined_audio", lambda: lambda *args: None)

    helper = tts.MLXTTSHelper(model_path="fake")

    with pytest.raises(RuntimeError, match="join_audio did not create"):
        helper.synthesize(text="hello", output_path=str(output))


def test_mlx_tts_helper_passes_voice_clone_options(monkeypatch):
    """MLX TTS helper should pass voice cloning options to mlx_audio."""
    import numpy as np
    import pycut.tts as tts

    seen = {}

    class FakeResult:
        audio = np.asarray([0.1], dtype=np.float32)
        sample_rate = 24000

    class FakeModel:
        def generate(self, text, **kwargs):
            seen["text"] = text
            seen.update(kwargs)
            yield FakeResult()

    monkeypatch.setattr(tts.MLXTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_write_wav", lambda output_path, audio, sample_rate: output_path)

    helper = tts.MLXTTSHelper(model_path="fake")

    assert helper.synthesize(
        text="target",
        output_path="out.wav",
        reference_audio="reference.wav",
        prompt_text="reference transcript",
        join_audio=False,
    ) == "out.wav"
    assert seen["text"] == "target"
    assert seen["ref_audio"] == "reference.wav"
    assert seen["ref_text"] == "reference transcript"


def test_voxcpm_tts_helper_writes_generated_audio(monkeypatch):
    """VoxCPM TTS helper should write generated audio with model sample rate."""
    import numpy as np
    import pycut.tts as tts

    class FakeModel:
        tts_model = types.SimpleNamespace(sample_rate=24000)

        def generate(self, **kwargs):
            assert kwargs["text"] == "hello"
            assert kwargs["cfg_value"] == 2.0
            return np.asarray([0.1, -0.1], dtype=np.float32)

    written = {}
    monkeypatch.setattr(tts.VoxCPMTTSHelper, "load_model", lambda self: setattr(self, "model", FakeModel()))
    monkeypatch.setattr(tts, "_write_wav", lambda output_path, audio, sample_rate: written.update({
        "output_path": output_path,
        "audio": list(audio),
        "sample_rate": sample_rate,
    }) or output_path)

    helper = tts.VoxCPMTTSHelper(model_path="fake")

    assert helper.synthesize(text="hello", output_path="out.wav") == "out.wav"
    assert written == {"output_path": "out.wav", "audio": [0.1, -0.1], "sample_rate": 24000}


def test_voxcpm_tts_helper_wraps_binary_import_errors(monkeypatch):
    """A torch/torchaudio ABI failure during voxcpm import should be actionable."""
    import pycut.tts as tts

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "voxcpm":
            raise OSError("libtorchaudio undefined symbol")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    helper = tts.VoxCPMTTSHelper(model_path="fake")

    with pytest.raises(RuntimeError, match="working voxcpm, torch, and torchaudio"):
        helper.load_model()


def test_video_clipper_constructor_omits_remote_analysis_args(monkeypatch):
    """VideoClipper should no longer expose remote-analysis constructor args."""
    import inspect
    import pycut.config as config
    from pycut.clipper import VideoClipper

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    params = inspect.signature(VideoClipper).parameters

    assert "gemini_api_key" not in params
    assert "api_key" not in params
    assert "base_url" not in params
    assert "model" not in params


def test_main_module_omits_remote_analysis_helpers():
    """clipper.py should not import or call the removed remote analysis module."""
    import pycut.clipper as clipper_module

    source = inspect.getsource(clipper_module)

    assert "pycut.analysis" not in source
    assert "analysis." not in source
    assert "OpenAI" not in source
    assert "from google import genai" not in source
    assert "GEMINI_AVAILABLE =" not in source
    assert "genai.Client(" not in source


def test_transcribe_with_vad_requires_torch_when_optional_dependency_missing(monkeypatch):
    """VAD transcription should fail with installation guidance when torch is unavailable."""
    import types
    import pycut.asr as asr

    class FakeArray:
        shape = (3,)

    helper = asr.MLXASRHelper.__new__(asr.MLXASRHelper)
    helper.vad_model = object()
    helper.load_vad_model = lambda: None

    fake_soundfile = types.SimpleNamespace(
        read=lambda *args, **kwargs: (FakeArray(), 16000)
    )
    fake_silero_vad = types.SimpleNamespace(get_speech_timestamps=lambda *args, **kwargs: [])

    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "silero_vad", fake_silero_vad)
    monkeypatch.setattr(asr, "torch", None, raising=False)

    with pytest.raises(RuntimeError, match="VAD transcription requires torch"):
        helper.transcribe_with_vad("fake.wav")


def test_transcribe_with_vad_reports_soundfile_install_guidance(monkeypatch):
    """VAD transcription should explain how to install soundfile when it is missing."""
    import pycut.asr as asr

    helper = asr.MLXASRHelper.__new__(asr.MLXASRHelper)
    helper.vad_model = object()
    helper.load_vad_model = lambda: None

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "soundfile":
            raise ModuleNotFoundError("No module named 'soundfile'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(asr, "torch", object(), raising=False)

    with pytest.raises(RuntimeError, match="Install it with: pip install soundfile"):
        helper.transcribe_with_vad("fake.wav")


def test_dependency_checks_omit_legacy_vllm_cuda_paths():
    """Dependency helper checks should not reintroduce legacy vLLM/CUDA runtime knobs."""
    legacy_free_sources = [
        inspect.getsource(_check_imports),
        inspect.getsource(_check_runtime_backend),
        inspect.getsource(_check_translation_model),
        inspect.getsource(main),
    ]
    combined_source = "\n".join(legacy_free_sources)

    assert "CUDA" not in combined_source
    assert "torch.cuda" not in combined_source
    assert "AutoConfig" not in combined_source
    assert "google/translategemma-4b-it" not in combined_source


def _is_apple_silicon():
    return platform.system().lower() == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _check_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    if not _is_apple_silicon():
        print("❌ This dependency check targets macOS Apple Silicon only")
        return False

    print("✅ macOS Apple Silicon runtime detected")

    try:
        import mlx_audio  # noqa: F401
        print("✅ mlx-audio")
    except ImportError as e:
        print(f"❌ mlx-audio: {e}")
        return False
    try:
        import googletrans  # noqa: F401
        print("✅ py-googletrans")
    except ImportError as e:
        print(f"❌ py-googletrans: {e} (try: pip install -U py-googletrans 'httpx<0.28')")
        return False

    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False

    try:
        import soundfile  # noqa: F401
        print("✅ soundfile")
    except ImportError as e:
        print(f"❌ soundfile: {e}")
        return False

    return True


def test_imports():
    return _check_imports()


def _check_ffmpeg():
    """Test that ffmpeg and ffprobe are available."""
    print("\nTesting ffmpeg...")
    import subprocess
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split('\n')[0]
        print(f"✅ {version}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ ffmpeg not found: {e}")
        return False
    
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split('\n')[0]
        print(f"✅ {version}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ ffprobe not found: {e}")
        return False
    
    return True


def test_ffmpeg():
    return _check_ffmpeg()


def _check_runtime_backend():
    """Test GPU availability."""
    print("\nTesting runtime backend...")
    if not _is_apple_silicon():
        print("❌ This dependency check targets macOS Apple Silicon only")
        return False

    print("✅ Apple Silicon runtime detected")
    print("   MLX backend will be used for local ASR and translation")
    return True


def test_gpu():
    return _check_runtime_backend()


def _check_translation_model():
    """Test if translation model can be loaded (without actually loading it)."""
    print("\nTesting translation model availability...")
    if not _is_apple_silicon():
        print("❌ This dependency check targets macOS Apple Silicon only")
        return False

    model_name = "py-googletrans"
    try:
        from googletrans import Translator  # noqa: F401
        print(f"✅ py-googletrans import ok, target backend '{model_name}'")
        print("   (Note: Actual model loading happens during video processing)")
    except Exception as e:
        print(f"⚠️  Could not verify MLX py-googletrans translation backend: {e}")
    
    return True


def test_translation_model():
    return _check_translation_model()


def main():
    print("="*60)
    print("Video Clipping Script - Dependency Check")
    print("="*60)
    
    all_ok = True
    
    all_ok &= _check_imports()
    all_ok &= _check_ffmpeg()
    all_ok &= _check_runtime_backend()
    all_ok &= _check_translation_model()
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ All critical dependencies are satisfied!")
        print("\nYou can now run:")
        print("  python scripts/clipping.py /path/to/video.mp4 -o ./output")
    else:
        print("❌ Some dependencies are missing.")
        print("\nPlease install missing dependencies:")
        print("  Supported runtime: macOS Apple Silicon only")
        print("  pip install mlx-audio py-googletrans 'httpx<0.28' google-generativeai numpy soundfile")
        print("  # And install ffmpeg for your platform")
    print("="*60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())


import tempfile, json as _json, os as _os

class TestTranscriptJsonFormat:
    def _write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f)

    def test_load_new_format_returns_segments_and_metadata(self):
        from pycut.video_io import _load_segments_from_transcript_json
        data = {
            "title": "主标题",
            "subtitle": "副标题",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "words": []}],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f)
            path = f.name
        try:
            segments, meta = _load_segments_from_transcript_json(path)
            assert len(segments) == 1
            assert segments[0].text == "hello"
            assert meta["title"] == "主标题"
            assert meta["subtitle"] == "副标题"
        finally:
            _os.unlink(path)

    def test_load_old_array_format_backward_compat(self):
        from pycut.video_io import _load_segments_from_transcript_json
        data = [{"start": 0.0, "end": 1.5, "text": "world", "words": []}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f)
            path = f.name
        try:
            segments, meta = _load_segments_from_transcript_json(path)
            assert len(segments) == 1
            assert segments[0].text == "world"
            assert meta["title"] == ""
            assert meta["subtitle"] == ""
        finally:
            _os.unlink(path)

    def test_load_segments_null_in_new_format(self):
        from pycut.video_io import _load_segments_from_transcript_json
        data = {"title": "t", "subtitle": "s", "segments": None}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f)
            path = f.name
        try:
            segments, meta = _load_segments_from_transcript_json(path)
            assert segments == []
        finally:
            _os.unlink(path)

    def test_new_format_roundtrip(self):
        from pycut.video_io import _load_segments_from_transcript_json
        import sys, os as _os2
        from pycut.utils import Segment

        segs = [Segment(start=0.0, end=1.0, text="test", words=[])]
        data = {
            "title": "测试标题",
            "subtitle": "测试副标题",
            "segments": [{"start": s.start, "end": s.end, "text": s.text, "words": s.words or []} for s in segs],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
            path = f.name
        try:
            loaded_segs, meta = _load_segments_from_transcript_json(path)
            assert loaded_segs[0].text == "test"
            assert meta["title"] == "测试标题"
            assert meta["subtitle"] == "测试副标题"
        finally:
            _os.unlink(path)

    def test_load_null_items_in_old_format(self):
        from pycut.video_io import _load_segments_from_transcript_json
        data = [None, {"start": 0.0, "end": 1.0, "text": "ok", "words": []}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f)
            path = f.name
        try:
            segments, meta = _load_segments_from_transcript_json(path)
            assert len(segments) == 1
            assert segments[0].text == "ok"
        finally:
            _os.unlink(path)


def test_transcript_store_copies_provided_transcript_and_loads_metadata(tmp_path):
    import json as _json
    from pycut.transcript_store import TranscriptStore

    provided = tmp_path / "provided.json"
    provided.write_text(_json.dumps({
        "title": "stored title",
        "subtitle": "stored subtitle",
        "segments": [{"start": 0.0, "end": 1.0, "text": "stored", "words": []}],
    }), encoding="utf-8")

    store = TranscriptStore(tmp_path / "out", "demo")
    document = store.load_provided(provided)

    assert document.metadata.title == "stored title"
    assert document.segments[0].text == "stored"
    assert store.path.exists()
    assert _json.loads(store.path.read_text(encoding="utf-8"))["segments"][0]["text"] == "stored"


def test_prepare_timeline_filters_margins_and_resolves_overlaps():
    from pycut.timeline import prepare_timeline
    from pycut.utils import Segment

    timeline = prepare_timeline(
        [
            Segment(start=0.0, end=1.0, text="", words=[]),
            Segment(start=1.0, end=2.0, text="first", words=[]),
            Segment(start=2.05, end=3.0, text="second", words=[]),
        ],
        filter_empty_segments=True,
        margin_left=-0.2,
        margin_right=0.2,
    )

    assert [cue.text for cue in timeline.cues] == ["first", "second"]
    assert timeline.cues[0].start == pytest.approx(0.8)
    assert timeline.cues[0].end <= timeline.cues[1].start


def test_prepare_export_timeline_keeps_timeline_interface():
    from pycut.timeline import TimelineCue, TranscriptTimeline, prepare_export_timeline

    timeline = TranscriptTimeline(
        cues=[
            TimelineCue(start=0.0, end=1.0, text="first"),
            TimelineCue(start=1.0, end=2.0, text="second"),
        ],
        title="stored title",
        subtitle="stored subtitle",
    )

    export_timeline, chunks = prepare_export_timeline(timeline, max_duration=1.5)

    assert export_timeline.title == "stored title"
    assert export_timeline.subtitle == "stored subtitle"
    assert [cue.text for cue in export_timeline.cues] == ["first", "second"]
    assert [[cue.text for cue in chunk] for chunk in chunks] == [["first"], ["second"]]


def test_media_job_workflow_uses_provided_transcript_without_video_dependencies(tmp_path):
    import json as _json

    from pycut.media_job import MediaJob
    from pycut.media_workflow import MediaJobWorkflow, WorkflowAdapters

    provided_transcript = tmp_path / "provided.json"
    provided_transcript.write_text(
        _json.dumps({
            "title": "stored title",
            "subtitle": "stored subtitle",
            "segments": [{"start": 0.0, "end": 1.0, "text": "provided", "words": []}],
        }),
        encoding="utf-8",
    )

    events = []

    adapters = WorkflowAdapters(
        extract_audio=lambda *_: events.append("extract"),
        transcribe_audio=lambda *_args, **_kwargs: pytest.fail("ASR should not run"),
        unload_asr_model=lambda: events.append("unload"),
        generate_ass_subtitle=lambda *_args, **_kwargs: pytest.fail("ASS should not run"),
        generate_fcpxml=lambda *_args, **_kwargs: pytest.fail("FCPXML should not run"),
        render_video_with_subtitles_complex=lambda *_args, **_kwargs: pytest.fail("render should not run"),
    )
    job = MediaJob(
        video_path=str(tmp_path / "demo.mp4"),
        output_dir=str(tmp_path / "demo"),
        output_formats=["json"],
        transcript_json_path=str(provided_transcript),
    )

    results = MediaJobWorkflow(job, adapters=adapters, segment_duration=300).run()

    expected_path = tmp_path / "demo" / "demo_transcript.json"
    assert events == []
    assert results == {"transcript": str(expected_path)}
    assert _json.loads(expected_path.read_text(encoding="utf-8"))["segments"][0]["text"] == "provided"


def test_runtime_profile_resolves_backend_specific_defaults(monkeypatch):
    import pycut.config as config

    monkeypatch.setattr(config, "resolve_default_qwen_asr_model", lambda: "cached-qwen-asr")
    monkeypatch.setattr(config, "resolve_default_qwen_aligner_model", lambda: "cached-qwen-aligner")
    monkeypatch.setattr(config, "resolve_default_mlx_tts_model", lambda: "cached-mlx-tts")
    monkeypatch.setattr(config, "resolve_default_voxcpm_tts_model", lambda: "cached-voxcpm")

    mac_profile = config.current_runtime_profile(system="Darwin", machine="arm64")
    assert mac_profile.asr_backend == "mlx"
    assert mac_profile.tts_backend == "mlx"
    assert mac_profile.default_asr_model("zh-CN") == config.DEFAULT_CHINESE_ASR_MODEL
    assert mac_profile.default_aligner_model() == config.DEFAULT_ALIGNER_MODEL
    assert mac_profile.default_tts_model() == "cached-mlx-tts"

    linux_profile = config.current_runtime_profile(system="Linux", machine="x86_64")
    assert linux_profile.asr_backend == "qwen"
    assert linux_profile.tts_backend == "voxcpm"
    assert linux_profile.default_asr_model("en") == "cached-qwen-asr"
    assert linux_profile.default_aligner_model() == "cached-qwen-aligner"
    assert linux_profile.default_tts_model() == "cached-voxcpm"



class TestProcessVideoTranscriptInput:
    def test_extract_audio_not_called_when_transcript_provided(self):
        """When transcript_json_path is given, process_video must NOT call extract_audio."""
        import tempfile as _tmpfile, json as _json2, os as _os2
        from unittest.mock import patch

        data = {
            "title": "测试",
            "subtitle": "副标题",
            "segments": [{"start": 0.0, "end": 2.0, "text": "你好", "words": []}],
        }
        with _tmpfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json2.dump(data, f, ensure_ascii=False)
            json_path = f.name

        with _tmpfile.TemporaryDirectory() as output_dir:
            try:
                from pycut.clipper import VideoClipper
                clipper = VideoClipper.__new__(VideoClipper)
                clipper.filter_fillers = False
                clipper.segment_duration = 300
                clipper.max_chars = 30

                extract_called = []

                def fake_extract(video_path, output_path):
                    extract_called.append(True)

                with patch.object(clipper, 'extract_audio', side_effect=fake_extract):
                    clipper.process_video(
                        video_path="/fake/video.mp4",
                        output_dir=output_dir,
                        output_formats=["json"],
                        transcript_json_path=json_path,
                    )

                assert not extract_called, "extract_audio should NOT be called when transcript_json_path is given"
            finally:
                _os2.unlink(json_path)

    def test_title_from_json_used_in_no_clip_mode(self):
        """When JSON has title/subtitle, no-clip mode uses them for default highlight."""
        import tempfile as _tmpfile, json as _json2, os as _os2
        from unittest.mock import patch

        data = {
            "title": "从JSON来的标题",
            "subtitle": "副标题",
            "segments": [{"start": 0.0, "end": 2.0, "text": "测试", "words": []}],
        }
        with _tmpfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json2.dump(data, f, ensure_ascii=False)
            json_path = f.name

        captured_titles = []

        with _tmpfile.TemporaryDirectory() as output_dir:
            try:
                from pycut.clipper import VideoClipper
                clipper = VideoClipper.__new__(VideoClipper)
                clipper.filter_fillers = False
                clipper.segment_duration = 300
                clipper.max_chars = 30
                clipper.asr_model = None
                clipper._mlx_aligner = None

                def capture_fcpxml(self_inner, video_path, timeline, output_path, **kwargs):
                    captured_titles.append(timeline.title)

                with patch.object(VideoClipper, 'generate_fcpxml', capture_fcpxml):
                    clipper.process_video(
                        video_path="/fake/video.mp4",
                        output_dir=output_dir,
                        output_formats=["fcpxml"],
                        transcript_json_path=json_path,
                    )

                assert "从JSON来的标题" in captured_titles, f"Expected title from JSON, got: {captured_titles}"
            finally:
                _os2.unlink(json_path)



class TestTimelineJsonOutput:
    def test_timeline_cue_serialization_roundtrip(self):
        """Timeline cues can be serialized to JSON and deserialized back."""
        import json as _json
        from pycut.timeline import TimelineCue

        cues = [
            TimelineCue(start=0.0, end=10.0, text="内容", words=[])
        ]

        serialized = [
            {
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
                "words": cue.words,
            }
            for cue in cues
        ]

        result = _json.dumps(serialized, ensure_ascii=False)
        parsed = _json.loads(result)

        assert parsed[0]["text"] == "内容"


# ---------------------------------------------------------------------------
# subtitle module tests
# ---------------------------------------------------------------------------

def test_extract_transcription_for_range_returns_overlapping_text():
    import pycut.subtitle as subtitle
    from pycut.utils import Segment
    segs = [
        Segment(start=0.0, end=2.0, text="hello", words=[]),
        Segment(start=2.0, end=4.0, text="world", words=[]),
        Segment(start=5.0, end=7.0, text="bye", words=[]),
    ]
    result = subtitle.extract_transcription_for_range(segs, 1.0, 3.0)
    assert "hello" in result
    assert "world" in result
    assert "bye" not in result


# ---------------------------------------------------------------------------
# renderer module tests
# ---------------------------------------------------------------------------

def test_select_video_encoder_returns_h264_videotoolbox_on_macos(monkeypatch):
    import pycut.renderer as renderer
    monkeypatch.setattr(renderer.platform, "system", lambda: "Darwin")
    assert renderer.select_video_encoder() == "h264_videotoolbox"


def test_select_video_encoder_returns_libx264_on_linux(monkeypatch):
    import pycut.renderer as renderer
    monkeypatch.setattr(renderer.platform, "system", lambda: "Linux")
    assert renderer.select_video_encoder() == "libx264"


def test_process_video_ignores_deprecated_remote_analysis_flags(monkeypatch, tmp_path):
    """Deprecated process_video flags should not create analysis side outputs."""
    import json as _json
    import pycut.config as config
    from pycut.clipper import VideoClipper
    from pycut.utils import Segment

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    fake_segments = [
        Segment(start=0.0, end=1.0, text="hello", words=[]),
        Segment(start=1.0, end=2.0, text="world", words=[]),
    ]

    transcript_path = tmp_path / "video_transcript.json"
    transcript_path.write_text(_json.dumps({
        "title": "", "subtitle": "",
        "segments": [{"start": s.start, "end": s.end, "text": s.text, "words": []} for s in fake_segments],
    }))

    vc = VideoClipper.__new__(VideoClipper)
    vc.segment_duration = 300
    vc.filter_fillers = False
    vc.max_chars = 30
    vc.max_duration = 30.0

    monkeypatch.setattr(vc, "_unload_asr_model", lambda: None)
    monkeypatch.setattr(vc, "_filter_subtitle_segments", lambda segs, **kw: segs)
    monkeypatch.setattr(vc, "_resolve_overlaps", lambda segs, *a: segs)
    monkeypatch.setattr(vc, "generate_ass_subtitle", lambda *a, **kw: None)
    monkeypatch.setattr(vc, "generate_fcpxml", lambda *a, **kw: None)

    vc.process_video(
        video_path=str(tmp_path / "video.mp4"),
        output_dir=str(tmp_path),
        translate=False,
        source_lang="en",
        target_lang="en",
        orientation="landscape",
        subtitle_position="translated-top",
        first_subtitle_delay=0.0,
        filter_empty_segments=True,
        margin_left=0.0,
        margin_right=0.0,
        output_formats={"ass"},
        fcpxml_frame_rate=25.0,
        fcpxml_speed=1.0,
        transcript_json_path=str(transcript_path),
    )

    assert not list(tmp_path.glob("*_highlights.json"))
    assert not list(tmp_path.glob("*_summary.txt"))


def test_cli_help_omits_remote_analysis_flags(monkeypatch, capsys):
    """CLI should no longer expose remote analysis flags."""
    import pycut.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli_module.main()

    help_output = capsys.readouterr().out
    assert "--highlight" not in help_output
    assert "--api-key" not in help_output
    assert "--base-url" not in help_output
    assert "--correct-words" not in help_output
    assert "--no-clip" not in help_output


def test_cli_passes_no_align_to_video_clipper(monkeypatch):
    """CLI should disable alignment when --no-align is specified."""
    import pycut.cli as cli_module
    import pycut.config as config
    import pycut.clipper as clipper_module

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    seen = {}

    def fake_process_video(self, **kwargs):
        seen["process_video"] = kwargs
        return {}

    class FakeVideoClipper:
        def __init__(self, **kwargs):
            seen["clipper_init"] = kwargs

        process_video = fake_process_video

    monkeypatch.setattr(cli_module, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(clipper_module, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(cli_module, "_expand_video_inputs", lambda inputs: ["/tmp/input.mov"])

    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "/tmp/input.mov",
        "--no-align",
    ])

    cli_module.main()

    assert seen["clipper_init"]["enable_align"] is False


def test_cli_help_exposes_subtitle_color_defaults(monkeypatch, capsys):
    """CLI help should document subtitle color defaults for original/translation."""
    import pycut.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli_module.main()

    help_output = capsys.readouterr().out
    assert "--original-subtitle-color" in help_output
    assert "#FFFFFF" in help_output
    assert "--translation-subtitle-color" in help_output
    assert "#FFA500" in help_output
    assert "--highlight-subtitle-color" not in help_output


def test_main_passes_subtitle_colors_to_process_video(monkeypatch):
    """CLI should pass subtitle color overrides through to process_video."""
    import pycut.cli as cli_module
    import pycut.config as config
    import pycut.clipper as clipper_module

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    calls = {}

    def fake_process_video(self, **kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(clipper_module.VideoClipper, "process_video", fake_process_video)
    monkeypatch.setattr(cli_module, "_expand_video_inputs", lambda inputs: ["/fake/video.mp4"])

    monkeypatch.setattr(sys, "argv", [
        "main.py", "/fake/video.mp4",
        "--format", "ass",
        "--original-subtitle-color", "#112233",
        "--translation-subtitle-color", "#445566",
    ])

    cli_module.main()

    assert calls.get("original_subtitle_color") == "#112233"
    assert calls.get("translation_subtitle_color") == "#445566"


def test_cli_resolves_default_asr_model_from_source_language(monkeypatch):
    """CLI should map source language families to the expected default ASR model."""
    import pycut.cli as cli_module
    import pycut.config as config_module

    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config_module.platform, "machine", lambda: "arm64")

    assert cli_module._resolve_default_asr_model("en") == config_module.DEFAULT_EN_ASR_MODEL
    assert cli_module._resolve_default_asr_model("en-US") == config_module.DEFAULT_EN_ASR_MODEL
    assert cli_module._resolve_default_asr_model("zh") == config_module.DEFAULT_CHINESE_ASR_MODEL
    assert cli_module._resolve_default_asr_model("zh-CN") == config_module.DEFAULT_CHINESE_ASR_MODEL
    assert cli_module._resolve_default_asr_model("ja") == config_module.DEFAULT_FALLBACK_ASR_MODEL

    monkeypatch.setattr(config_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(config_module, "resolve_default_qwen_asr_model", lambda: config_module.DEFAULT_QWEN_ASR_MODEL)

    assert cli_module._resolve_default_asr_model("en") == config_module.DEFAULT_QWEN_ASR_MODEL
    assert cli_module._resolve_default_asr_model("zh") == config_module.DEFAULT_QWEN_ASR_MODEL


def test_cli_resolves_default_output_dir_from_source_stem():
    """CLI should place default output under a sibling directory named after the source stem."""
    import pycut.cli as cli_module

    assert cli_module._resolve_output_dir("/Users/dake/Movies/demo.mp4", None) == "/Users/dake/Movies/demo"
    assert cli_module._resolve_output_dir("/Users/dake/Movies/demo.mp4", "/tmp/custom-output") == "/tmp/custom-output"


def test_cli_uses_input_parent_as_default_output_dir(monkeypatch):
    """CLI should default output_dir to a sibling directory named after the input stem."""
    import pycut.cli as cli_module
    import pycut.config as config
    import pycut.clipper as clipper_module

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    calls = {}

    def fake_process_video(self, **kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(clipper_module.VideoClipper, "process_video", fake_process_video)
    monkeypatch.setattr(
        cli_module,
        "_expand_video_inputs",
        lambda inputs: ["/Users/dake/Movies/youtube/0226/example.mp4"],
    )

    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "/Users/dake/Movies/youtube/0226/",
        "--format",
        "video",
    ])

    cli_module.main()

    assert calls.get("output_dir") == "/Users/dake/Movies/youtube/0226/example"


def test_process_video_reuses_transcript_from_per_file_output_dir(monkeypatch, tmp_path):
    """process_video should reuse the transcript cached in the per-file output directory."""
    import json as _json

    from pycut.clipper import VideoClipper

    video_path = tmp_path / "demo.mp4"
    output_dir = tmp_path / "demo"
    transcript_path = output_dir / "demo_transcript.json"
    output_dir.mkdir()
    transcript_path.write_text(_json.dumps({
        "title": "",
        "subtitle": "",
        "segments": [{"start": 0.0, "end": 1.0, "text": "cached", "words": []}],
    }), encoding="utf-8")

    clipper = VideoClipper.__new__(VideoClipper)
    clipper.filter_fillers = False
    clipper.segment_duration = 300
    clipper.max_chars = 30
    clipper.max_duration = 30.0

    extract_called = False

    def fake_extract(video_path, output_path):
        nonlocal extract_called
        extract_called = True

    monkeypatch.setattr(clipper, "extract_audio", fake_extract)
    monkeypatch.setattr(clipper, "_filter_subtitle_segments", lambda segs, **kw: segs)
    monkeypatch.setattr(clipper, "_resolve_overlaps", lambda segs, *a: segs)
    monkeypatch.setattr(clipper, "_unload_asr_model", lambda: None)

    results = clipper.process_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        output_formats=["json"],
    )

    assert extract_called is False
    assert results["transcript"] == str(transcript_path)


def test_process_video_copies_provided_transcript_into_per_file_output_dir(monkeypatch, tmp_path):
    """process_video should copy a provided transcript into the managed per-file output directory."""
    import json as _json

    from pycut.clipper import VideoClipper

    video_path = tmp_path / "demo.mp4"
    output_dir = tmp_path / "demo"
    provided_transcript = tmp_path / "provided.json"
    provided_transcript.write_text(_json.dumps({
        "title": "",
        "subtitle": "",
        "segments": [{"start": 0.0, "end": 1.0, "text": "provided", "words": []}],
    }), encoding="utf-8")

    clipper = VideoClipper.__new__(VideoClipper)
    clipper.filter_fillers = False
    clipper.segment_duration = 300
    clipper.max_chars = 30
    clipper.max_duration = 30.0

    monkeypatch.setattr(clipper, "_filter_subtitle_segments", lambda segs, **kw: segs)
    monkeypatch.setattr(clipper, "_resolve_overlaps", lambda segs, *a: segs)
    monkeypatch.setattr(clipper, "_unload_asr_model", lambda: None)
    monkeypatch.setattr(clipper, "extract_audio", lambda *a, **kw: pytest.fail("extract_audio should not run"))

    results = clipper.process_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        output_formats=["json"],
        transcript_json_path=str(provided_transcript),
    )

    expected_transcript = output_dir / "demo_transcript.json"
    assert results["transcript"] == str(expected_transcript)
    assert expected_transcript.exists()
    assert _json.loads(expected_transcript.read_text(encoding="utf-8"))["segments"][0]["text"] == "provided"


def test_cli_respects_explicit_output_dir_and_asr_model(monkeypatch):
    """CLI should preserve explicit output-dir and asr-model arguments."""
    import pycut.cli as cli_module
    import pycut.config as config
    import pycut.clipper as clipper_module

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    seen = {}

    def fake_process_video(self, **kwargs):
        seen["process_video"] = kwargs
        return {}

    class FakeVideoClipper:
        def __init__(self, **kwargs):
            seen["clipper_init"] = kwargs

        process_video = fake_process_video

    monkeypatch.setattr(cli_module, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(clipper_module, "VideoClipper", FakeVideoClipper)
    monkeypatch.setattr(cli_module, "_expand_video_inputs", lambda inputs: ["/tmp/input.mov"])

    monkeypatch.setattr(sys, "argv", [
        "main.py",
        "/tmp/input.mov",
        "--source-lang",
        "zh",
        "--asr-model",
        "custom-asr-model",
        "--output-dir",
        "/tmp/custom-output",
    ])

    cli_module.main()

    assert seen["clipper_init"]["asr_model_path"] == "custom-asr-model"
    assert seen["process_video"]["output_dir"] == "/tmp/custom-output"


def test_cli_help_mentions_dynamic_defaults(monkeypatch):
    """CLI source should define help text for the dynamic defaults."""
    import pycut.cli as cli_module
    import pycut.config as config_module

    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config_module.platform, "machine", lambda: "arm64")

    source = inspect.getsource(cli_module)
    config_source = inspect.getsource(config_module)

    assert "video.parent / video.stem" in source
    assert cli_module._resolve_default_asr_model("en") == config_module.DEFAULT_EN_ASR_MODEL
    assert "Qwen3-ASR-1.7B-8bit" in config_source
    assert "whisper-large-v3-turbo" in config_source


def test_cli_help_shows_usage_examples(monkeypatch, capsys):
    """CLI help should include example commands in the description block."""
    import pycut.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli_module.main()

    help_output = capsys.readouterr().out
    assert "Examples:" in help_output
    assert "pycut --source-lang en --format json,srt" in help_output
    assert "pycut --translate --source-lang en --target-lang zh --format video,ass --orientation portrait" in help_output


def test_generate_ass_subtitle_uses_configured_semantic_colors(tmp_path):
    import pycut.subtitle as subtitle_mod
    from pycut.timeline import TimelineCue, TranscriptTimeline
    from pycut.utils import Segment

    output_path = tmp_path / "output.ass"
    segments = [
        Segment(start=0.0, end=2.0, text="hello world"),
    ]
    timeline = TranscriptTimeline(
        cues=[TimelineCue(start=0.0, end=2.0, text="hello world")],
        title="hello",
    )

    subtitle_mod.generate_ass_subtitle(
        timeline=timeline,
        output_path=str(output_path),
        translate=True,
        subtitle_position="translated-top",
        translate_fn=lambda texts, _source, _target: [f"tr:{text}" for text in texts],
        original_subtitle_color="#112233",
        translation_subtitle_color="#445566",
    )

    content = output_path.read_text(encoding="utf-8")

    assert "Style: OriginalTop,Arial Unicode MS,50.0,&H00332211&," in content
    assert "Style: OriginalBottom,Arial Unicode MS,35.0,&H00332211&," in content
    assert "Style: TranslationTop,Arial Unicode MS,50.0,&H00665544&," in content
    assert "Style: TranslationBottom,Arial Unicode MS,35.0,&H00665544&," in content
    assert r"{\c" not in content
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,TranslationTop" in content
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,OriginalBottom" in content


@pytest.mark.integration
@pytest.mark.slow
def test_fixture_vad_example_matches_expected_transcript():
    """Optional real-model smoke test for the provided VAD fixture."""
    if os.environ.get("PYCUT_RUN_INTEGRATION") != "1":
        pytest.skip("set PYCUT_RUN_INTEGRATION=1 to run real-model fixture tests")

    import difflib
    import re

    from pycut.clipper import VideoClipper

    fixture_dir = Path(__file__).parent / "fixtures"
    audio_path = fixture_dir / "vad_example.wav"
    txt_path = fixture_dir / "vad_example.txt"
    assert audio_path.exists()
    assert txt_path.exists()

    clipper = VideoClipper(max_chars=30)
    segments = clipper.transcribe_audio(str(audio_path), source_lang="zh")

    assert segments
    prev_end = 0.0
    for segment in segments:
        assert 0.0 <= segment.start <= segment.end <= 70.48
        assert segment.start >= prev_end or segment.start == pytest.approx(prev_end, abs=0.5)
        prev_end = max(prev_end, segment.end)

    def normalize(text):
        return re.sub(r"[\s，。！？、,.!?;；:：]+", "", text)

    actual = normalize("".join(seg.text for seg in segments))
    expected = normalize(txt_path.read_text(encoding="utf-8"))
    ratio = difflib.SequenceMatcher(None, actual, expected).ratio()

    assert ratio >= 0.65, f"transcript similarity too low: {ratio:.2f}"
