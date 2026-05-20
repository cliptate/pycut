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


def test_runtime_guard_rejects_non_macos_apple_silicon(monkeypatch):
    """VideoClipper should fail fast outside macOS Apple Silicon."""
    import pycut.config as config
    from pycut.clipper import VideoClipper

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="macOS Apple Silicon"):
        VideoClipper(gemini_api_key=None)


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

    clipper = VideoClipper(gemini_api_key=None, translator=FakeTranslatorService())

    assert not hasattr(clipper, "runtime_backend")
    assert not hasattr(clipper, "translate_model_path")
    assert clipper.translate_text("hello", source_lang="en", target_lang="zh-cn") == "你好"


def test_asr_module_exposes_mlx_helpers():
    """ASR helpers should live in the extracted MLX-focused module."""
    import pycut.asr as asr

    assert hasattr(asr, "MLXASRHelper")
    assert hasattr(asr, "load_mlx_stt_model")


def test_package_metadata_declares_runtime_dependencies_used_by_source():
    """Installed CLI environments must include direct runtime deps used by source."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]
    dependency_names = {dep.split(";")[0].split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip() for dep in dependencies}

    assert "numpy" in dependency_names
    assert "soundfile" in dependency_names


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

    assert asr.__all__ == ["MLXASRHelper", "load_mlx_stt_model"]
    assert legacy_knobs.isdisjoint(asr.__all__)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.load_mlx_stt_model).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper.load_models).parameters)
    assert legacy_knobs.isdisjoint(inspect.signature(asr.MLXASRHelper.load_vad_model).parameters)

    with pytest.raises(TypeError, match="device"):
        asr.load_mlx_stt_model("mlx-community/whisper-tiny", device="cpu")

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

    vc = clipper_module.VideoClipper(gemini_api_key=None)
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


def test_process_video_unloads_asr_before_correction_and_postprocessing(monkeypatch, tmp_path):
    """ASR should be released before LLM correction and transcript chunk consumers."""
    import pycut.clipper as clipper_module
    from pycut.utils import Segment

    clipper = clipper_module.VideoClipper.__new__(clipper_module.VideoClipper)
    clipper.segment_duration = 300
    clipper.filter_fillers = True

    events = []
    raw_segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]
    corrected_segments = [Segment(start=0.0, end=1.0, text="hello corrected", words=[])]

    monkeypatch.setattr(clipper, "extract_audio", lambda video_path, output_path: events.append("extract"))

    def fake_transcribe(audio_path, orientation="landscape", source_lang="en"):
        events.append("transcribe")
        return raw_segments

    def fake_unload():
        events.append("unload")

    def fake_correct(segments, source_lang):
        events.append("correct")
        return corrected_segments

    def fake_filter(segments, filter_empty_segments=True):
        events.append("filter")
        return list(segments)

    monkeypatch.setattr(clipper, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(clipper, "_unload_asr_model", fake_unload)
    monkeypatch.setattr(clipper, "_correct_transcript", fake_correct)
    monkeypatch.setattr(clipper, "_filter_subtitle_segments", fake_filter)

    result = clipper.process_video(
        "demo.mp4",
        str(tmp_path),
        output_formats=["json"],
        correct_words=True,
        margin_left=0.0,
        margin_right=0.0,
    )

    assert events == ["extract", "transcribe", "unload", "correct", "filter"]
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

    clipper = VideoClipper(gemini_api_key=None)

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


def test_create_gemini_client_returns_none_without_api_key():
    """Client helper should no-op when no API key is provided."""
    import pycut.analysis as analysis

    assert analysis.create_client(None) is None


def test_create_gemini_client_returns_none_when_dependency_missing(monkeypatch):
    """Client helper should safely no-op when openai is unavailable."""
    import pycut.analysis as analysis

    monkeypatch.setattr(analysis, "OPENAI_AVAILABLE", False)
    monkeypatch.setattr(analysis, "OpenAI", None)

    assert analysis.create_client("secret-key") is None


def test_create_gemini_client_builds_configured_client(monkeypatch):
    """Client helper should build OpenAI client with the configured API key."""
    import pycut.analysis as analysis

    seen = {}
    sentinel_client = types.SimpleNamespace()

    def fake_client(*, api_key, base_url=None, timeout=None):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        seen["timeout"] = timeout
        return sentinel_client

    monkeypatch.setattr(analysis, "OpenAI", fake_client)

    result = analysis.create_client("secret-key")
    assert result is sentinel_client
    assert seen["api_key"] == "secret-key"
    assert seen["base_url"] == "https://api.openai.com/v1"
    assert seen["timeout"] == 60


def test_extract_json_payload_strips_fenced_json():
    """Gemini helper should extract JSON from fenced markdown responses."""
    import pycut.analysis as analysis

    fenced = """```json
{"highlights": [{"title": "demo"}]}
```"""

    assert analysis.extract_json_payload(fenced) == '{"highlights": [{"title": "demo"}]}'


def test_extract_gemini_highlights_returns_empty_list_for_null_highlights():
    """Highlight extraction should treat a null highlights payload as no highlights."""
    import pycut.analysis as analysis
    from pycut.utils import Segment

    class FakeCompletions:
        def create(self, *, model, messages):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"highlights": null}')
                )]
            )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()),
        _pycut_model="test-model",
    )
    segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]

    assert analysis.extract_highlights(client, segments, source_lang="zh", target_lang="en") == []


def test_extract_gemini_highlights_skips_entries_missing_start_or_end():
    """Highlight extraction should skip malformed entries missing required bounds."""
    import pycut.analysis as analysis
    from pycut.utils import Segment

    class FakeCompletions:
        def create(self, *, model, messages):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="""{
  "highlights": [
    {"end": 4.0, "title": "missing start"},
    {"start": 2.0, "title": "missing end"},
    {"start": 3.0, "end": 8.5, "title": "usable"}
  ]
}""")
                )]
            )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()),
        _pycut_model="test-model",
    )
    segments = [Segment(start=0.0, end=10.0, text="hello", words=[])]

    highlights = analysis.extract_highlights(
        client, segments, source_lang="zh", target_lang="en"
    )

    assert len(highlights) == 1
    assert highlights[0]["start"] == 3.0
    assert highlights[0]["end"] == 8.5
    assert highlights[0]["title"] == "usable"


def test_sanitize_segment_keywords_skips_non_integer_segment_ids():
    """Segment keyword metadata should keep only entries with integer segment IDs."""
    import pycut.analysis as analysis

    assert analysis._sanitize_segment_keywords(
        [
            {"segment_id": "0", "keywords": ["alpha"]},
            {"segment_id": 1.5, "keywords": ["beta"]},
            {"segment_id": True, "keywords": ["gamma"]},
            {"segment_id": 2, "keywords": ["delta"]},
        ]
    ) == [{"segment_id": 2, "keywords": ["delta"]}]


def test_sanitize_highlights_payload_replaces_none_text_fields_with_empty_strings():
    """Highlight text fields should treat null values as empty strings."""
    import pycut.analysis as analysis

    assert analysis._sanitize_highlights_payload(
        [
            {
                "start": 1.0,
                "end": 2.0,
                "title": None,
                "subtitle": None,
                "content": None,
            }
        ]
    ) == [
        {
            "start": 1.0,
            "end": 2.0,
            "title": "",
            "subtitle": "",
            "content": "",
            "keywords": [],
            "segment_keywords": [],
        }
    ]


def test_sanitize_keyword_list_ignores_complex_non_string_values():
    """Keyword sanitization should drop malformed nested values instead of stringifying them."""
    import pycut.analysis as analysis

    assert analysis._sanitize_keyword_list(
        ["alpha", {"keyword": "beta"}, ["gamma"], ("delta",), 2024, "omega", None]
    ) == ["alpha", "2024", "omega"]


def test_video_clipper_delegates_gemini_client_creation(monkeypatch):
    """VideoClipper should delegate LLM client setup to analysis helpers."""
    import pycut.analysis as analysis
    import pycut.config as config
    from pycut.clipper import VideoClipper

    seen = {}
    sentinel_client = object()

    def fake_create_client(api_key, base_url=None, model=None):
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        seen["model"] = model
        return sentinel_client

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(analysis, "create_client", fake_create_client)

    clipper = VideoClipper(api_key="secret-key")

    assert seen["api_key"] == "secret-key"
    assert clipper.llm_client is sentinel_client


def test_video_clipper_delegates_gemini_highlight_payload_parsing(monkeypatch):
    """VideoClipper should map helper payloads into Highlight objects."""
    import pycut.analysis as analysis
    import pycut.clipper as clipper_module
    from pycut.models import Highlight
    from pycut.utils import Segment

    seen = {}

    def fake_extract_highlights(client, segments, source_lang, target_lang):
        seen["client"] = client
        seen["segments"] = segments
        seen["source_lang"] = source_lang
        seen["target_lang"] = target_lang
        return [
            {
                "start": 1.5,
                "end": 9.0,
                "title": "重点",
                "subtitle": "副标题",
                "content": "摘要",
                "keywords": ["关键"],
                "segment_keywords": [{"segment_id": 0, "keywords": ["测试"]}],
            }
        ]

    monkeypatch.setattr(analysis, "extract_highlights", fake_extract_highlights)

    vc = clipper_module.VideoClipper.__new__(clipper_module.VideoClipper)
    vc.llm_client = object()
    segments = [Segment(start=1.5, end=9.0, text="hello", words=[])]

    highlights = vc.analyze_with_gemini_highlights(segments, source_lang="zh", target_lang="en")

    assert seen["client"] is vc.llm_client
    assert seen["segments"] == segments
    assert seen["source_lang"] == "zh"
    assert seen["target_lang"] == "en"
    assert len(highlights) == 1
    assert isinstance(highlights[0], Highlight)
    assert highlights[0].title == "重点"
    assert highlights[0].segment_keywords == [{"segment_id": 0, "keywords": ["测试"]}]


def test_video_clipper_skips_gemini_highlights_when_model_unset(monkeypatch):
    """VideoClipper should short-circuit to [] when LLM client is unset."""
    import pycut.analysis as analysis
    import pycut.clipper as clipper_module
    from pycut.utils import Segment

    helper_calls = []

    def fake_extract_highlights(*args, **kwargs):
        helper_calls.append((args, kwargs))
        return [{"title": "should not be used"}]

    monkeypatch.setattr(analysis, "extract_highlights", fake_extract_highlights)

    clipper = clipper_module.VideoClipper.__new__(clipper_module.VideoClipper)
    segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]

    assert clipper.analyze_with_gemini_highlights(segments, source_lang="zh", target_lang="en") == []
    assert helper_calls == []


def test_main_module_delegates_gemini_helpers_to_analysis_module():
    """clipper.py should delegate LLM client setup and parsing to analysis.py."""
    import pycut.clipper as clipper_module

    source = inspect.getsource(clipper_module)

    assert "analysis.create_client" in source
    assert "analysis.extract_highlights" in source
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


def test_dependency_checks_omit_legacy_non_mac_paths():
    """Dependency helper checks should only encode the supported macOS Apple Silicon path."""
    legacy_free_sources = [
        inspect.getsource(_check_imports),
        inspect.getsource(_check_runtime_backend),
        inspect.getsource(_check_translation_model),
        inspect.getsource(main),
    ]
    combined_source = "\n".join(legacy_free_sources)

    assert "import torch" not in combined_source
    assert "qwen_asr" not in combined_source
    assert "transformers" not in combined_source
    assert "CUDA" not in combined_source
    assert "torch.cuda" not in combined_source
    assert "AutoConfig" not in combined_source
    assert "google/translategemma-4b-it" not in combined_source
    assert "pip install torch qwen-asr transformers" not in combined_source


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
        import openai
        print(f"✅ openai {openai.__version__}")
    except ImportError as e:
        print(f"⚠️  openai: {e} (optional for LLM analysis)")

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


def _check_gemini_api():
    """Test OpenAI-compatible API key."""
    print("\nTesting LLM API...")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set (content analysis will be skipped)")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        return True
    
    print(f"✅ API key is set ({api_key[:10]}...)")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client configured")
        
        # Test API with simple request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test successful' in exactly two words."}],
        )
        print(f"✅ LLM API test: {response.choices[0].message.content.strip()}")
        
    except Exception as e:
        print(f"❌ LLM API test failed: {e}")
        return False
    
    return True


def test_gemini_api():
    return _check_gemini_api()


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
    all_ok &= _check_gemini_api()
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
            "highlights": []
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
        data = {"title": "t", "subtitle": "s", "segments": None, "highlights": None}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f)
            path = f.name
        try:
            segments, meta = _load_segments_from_transcript_json(path)
            assert segments == []
            assert meta["highlights"] == []
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
            "highlights": []
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
            path = f.name
        try:
            loaded_segs, meta = _load_segments_from_transcript_json(path)
            assert loaded_segs[0].text == "test"
            assert meta["title"] == "测试标题"
            assert meta["subtitle"] == "测试副标题"
            assert meta["highlights"] == []
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



class TestProcessVideoTranscriptInput:
    def test_extract_audio_not_called_when_transcript_provided(self):
        """When transcript_json_path is given, process_video must NOT call extract_audio."""
        import tempfile as _tmpfile, json as _json2, os as _os2
        from unittest.mock import patch

        data = {
            "title": "测试",
            "subtitle": "副标题",
            "segments": [{"start": 0.0, "end": 2.0, "text": "你好", "words": []}],
            "highlights": []
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
                clipper.llm_client = None
                clipper.gemini_model = None
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
            "highlights": []
        }
        with _tmpfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            _json2.dump(data, f, ensure_ascii=False)
            json_path = f.name

        captured_highlights = []

        with _tmpfile.TemporaryDirectory() as output_dir:
            try:
                from pycut.clipper import VideoClipper
                clipper = VideoClipper.__new__(VideoClipper)
                clipper.filter_fillers = False
                clipper.segment_duration = 300
                clipper.llm_client = None
                clipper.gemini_model = None
                clipper.max_chars = 30
                clipper.asr_model = None
                clipper._mlx_aligner = None

                def capture_fcpxml(self_inner, video_path, highlights, segments, output_path, **kwargs):
                    captured_highlights.extend(highlights)

                with patch.object(VideoClipper, 'generate_fcpxml', capture_fcpxml):
                    clipper.process_video(
                        video_path="/fake/video.mp4",
                        output_dir=output_dir,
                        output_formats=["fcpxml"],
                        transcript_json_path=json_path,
                        enable_clip=False,
                    )

                titles = [h.title for h in captured_highlights]
                assert "从JSON来的标题" in titles, f"Expected title from JSON, got: {titles}"
            finally:
                _os2.unlink(json_path)



class TestHighlightsJsonOutput:
    def test_highlight_serialization_roundtrip(self):
        """Highlights can be serialized to JSON and deserialized back."""
        import json as _json
        from pycut.models import Highlight

        highlights = [
            Highlight(
                start=0.0, end=10.0,
                title="测试标题", subtitle="副标题",
                content="内容",
                keywords=["k1", "k2"],
                segment_keywords=[{"segment_id": 0, "keywords": ["w1"]}]
            )
        ]

        serialized = [
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

        result = _json.dumps(serialized, ensure_ascii=False)
        parsed = _json.loads(result)

        assert parsed[0]["title"] == "测试标题"
        assert parsed[0]["subtitle"] == "副标题"
        assert parsed[0]["keywords"] == ["k1", "k2"]
        assert parsed[0]["segment_keywords"][0]["segment_id"] == 0


# ---------------------------------------------------------------------------
# subtitle module tests
# ---------------------------------------------------------------------------

def test_apply_keyword_highlighting_wraps_keywords_in_ass_tags():
    import pycut.subtitle as subtitle
    result = subtitle.apply_keyword_highlighting("hello world", ["world"])
    assert r"{\c&H0000FFFF&\fscx110\fscy110}" in result
    assert "world" in result
    assert r"{\r}" in result


def test_apply_keyword_highlighting_returns_text_unchanged_when_no_keywords():
    import pycut.subtitle as subtitle
    assert subtitle.apply_keyword_highlighting("hello world", []) == "hello world"


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


# ---------------------------------------------------------------------------
# analysis.extract_keywords_for_segments tests
# ---------------------------------------------------------------------------

def test_extract_keywords_for_segments_returns_segment_keywords():
    """extract_keywords_for_segments should parse LLM response into segment_keywords list."""
    import pycut.analysis as analysis
    from pycut.utils import Segment

    response_text = '{"segment_keywords": [{"segment_id": 0, "keywords": ["hello"]}, {"segment_id": 2, "keywords": ["world", "foo"]}]}'

    class FakeCompletions:
        def create(self, *, model, messages):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=response_text)
                )]
            )

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()),
        _pycut_model="test-model",
    )
    segments = [
        Segment(start=0.0, end=1.0, text="hello", words=[]),
        Segment(start=1.0, end=2.0, text="there", words=[]),
        Segment(start=2.0, end=3.0, text="world", words=[]),
    ]
    result = analysis.extract_keywords_for_segments(fake_client, segments, "en", "en")

    assert result == [
        {"segment_id": 0, "keywords": ["hello"]},
        {"segment_id": 2, "keywords": ["world", "foo"]},
    ]


def test_extract_keywords_for_segments_returns_empty_on_failure():
    """extract_keywords_for_segments should return [] when LLM call fails."""
    import pycut.analysis as analysis
    from pycut.utils import Segment

    class FakeCompletions:
        def create(self, *, model, messages):
            raise RuntimeError("network error")

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()),
        _pycut_model="test-model",
    )
    segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]
    result = analysis.extract_keywords_for_segments(fake_client, segments, "en", "en")
    assert result == []


def test_extract_keywords_for_segments_skips_invalid_entries():
    """extract_keywords_for_segments should skip entries with non-integer segment_id."""
    import pycut.analysis as analysis
    from pycut.utils import Segment

    response_text = '{"segment_keywords": [{"segment_id": "bad", "keywords": ["x"]}, {"segment_id": 1, "keywords": ["ok"]}]}'

    class FakeCompletions:
        def create(self, *, model, messages):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=response_text)
                )]
            )

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions()),
        _pycut_model="test-model",
    )
    segments = [
        Segment(start=0.0, end=1.0, text="a", words=[]),
        Segment(start=1.0, end=2.0, text="b", words=[]),
    ]
    result = analysis.extract_keywords_for_segments(fake_client, segments, "en", "en")
    assert result == [{"segment_id": 1, "keywords": ["ok"]}]


# ---------------------------------------------------------------------------
# --highlight / enable_highlight integration tests
# ---------------------------------------------------------------------------

def test_process_video_no_clip_highlight_calls_keyword_extraction(monkeypatch, tmp_path):
    """process_video with enable_clip=False and enable_highlight=True should call extract_keywords_for_segments."""
    import json as _json
    import pycut.analysis as analysis
    import pycut.config as config
    from pycut.clipper import VideoClipper
    from pycut.utils import Segment

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    keyword_calls = []

    def fake_extract_keywords(client, segs, source_lang, target_lang):
        keyword_calls.append(len(segs))
        return [{"segment_id": 0, "keywords": ["test"]}]

    monkeypatch.setattr(analysis, "extract_keywords_for_segments", fake_extract_keywords)

    fake_segments = [
        Segment(start=0.0, end=1.0, text="hello", words=[]),
        Segment(start=1.0, end=2.0, text="world", words=[]),
    ]

    transcript_path = tmp_path / "video_transcript.json"
    transcript_path.write_text(_json.dumps({
        "title": "", "subtitle": "",
        "segments": [{"start": s.start, "end": s.end, "text": s.text, "words": []} for s in fake_segments],
        "highlights": [],
    }))

    vc = VideoClipper.__new__(VideoClipper)
    vc.llm_client = object()  # truthy fake client
    vc.gemini_client = vc.llm_client
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
        enable_clip=False,
        enable_highlight=True,
        translate=False,
        source_lang="en",
        target_lang="en",
        orientation="landscape",
        subtitle_position="translated-top",
        first_subtitle_delay=0.0,
        max_title_chars=6,
        max_subtitle_chars=10,
        filter_empty_segments=True,
        margin_left=0.0,
        margin_right=0.0,
        output_formats={"ass"},
        fcpxml_frame_rate=25.0,
        fcpxml_speed=1.0,
        transcript_json_path=str(transcript_path),
    )

    assert len(keyword_calls) >= 1, "extract_keywords_for_segments should have been called"


def test_process_video_no_clip_without_highlight_skips_keyword_extraction(monkeypatch, tmp_path):
    """process_video with enable_clip=False and enable_highlight=False should NOT call extract_keywords_for_segments."""
    import json as _json
    import pycut.analysis as analysis
    import pycut.config as config
    from pycut.clipper import VideoClipper
    from pycut.utils import Segment

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config.platform, "machine", lambda: "arm64")

    keyword_calls = []

    def fake_extract_keywords(client, segs, source_lang, target_lang):
        keyword_calls.append(True)
        return []

    monkeypatch.setattr(analysis, "extract_keywords_for_segments", fake_extract_keywords)

    fake_segments = [Segment(start=0.0, end=1.0, text="hello", words=[])]

    transcript_path = tmp_path / "video_transcript.json"
    transcript_path.write_text(_json.dumps({
        "title": "", "subtitle": "",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
        "highlights": [],
    }))

    vc = VideoClipper.__new__(VideoClipper)
    vc.llm_client = object()
    vc.gemini_client = vc.llm_client
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
        enable_clip=False,
        enable_highlight=False,
        translate=False,
        source_lang="en",
        target_lang="en",
        orientation="landscape",
        subtitle_position="translated-top",
        first_subtitle_delay=0.0,
        max_title_chars=6,
        max_subtitle_chars=10,
        filter_empty_segments=True,
        margin_left=0.0,
        margin_right=0.0,
        output_formats={"ass"},
        fcpxml_frame_rate=25.0,
        fcpxml_speed=1.0,
        transcript_json_path=str(transcript_path),
    )

    assert keyword_calls == [], "extract_keywords_for_segments should NOT have been called"


def test_cli_exposes_highlight_flag():
    """CLI should expose a --highlight flag."""
    import subprocess
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        ["uv", "run", "--prerelease=allow", "python", "-m", "pycut", "--help"],
        capture_output=True, text=True,
        cwd=repo_root,
    )
    assert "--highlight" in result.stdout, f"--highlight not in help output:\n{result.stdout}"


def test_main_passes_enable_highlight_to_process_video(monkeypatch):
    """main.py should pass enable_highlight=True when --highlight is specified."""
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
        "--no-clip", "--highlight",
        "--format", "ass",
    ])

    cli_module.main()

    assert calls.get("enable_highlight") is True, f"enable_highlight not True in calls: {calls}"
    assert calls.get("enable_clip") is False, f"enable_clip not False in calls: {calls}"


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
    """CLI help should document subtitle color defaults for original/translation/highlight."""
    import pycut.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli_module.main()

    help_output = capsys.readouterr().out
    assert "--original-subtitle-color" in help_output
    assert "#FFFFFF" in help_output
    assert "--translation-subtitle-color" in help_output
    assert "#FFA500" in help_output
    assert "--highlight-subtitle-color" in help_output
    assert "#FFFF00" in help_output


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
        "--highlight-subtitle-color", "#778899",
    ])

    cli_module.main()

    assert calls.get("original_subtitle_color") == "#112233"
    assert calls.get("translation_subtitle_color") == "#445566"
    assert calls.get("highlight_subtitle_color") == "#778899"


def test_cli_resolves_default_asr_model_from_source_language():
    """CLI should map source language families to the expected default ASR model."""
    import pycut.cli as cli_module
    import pycut.config as config_module

    assert cli_module._resolve_default_asr_model("en") == config_module.DEFAULT_EN_ASR_MODEL
    assert cli_module._resolve_default_asr_model("en-US") == config_module.DEFAULT_EN_ASR_MODEL
    assert cli_module._resolve_default_asr_model("zh") == config_module.DEFAULT_CHINESE_ASR_MODEL
    assert cli_module._resolve_default_asr_model("zh-CN") == config_module.DEFAULT_CHINESE_ASR_MODEL
    assert cli_module._resolve_default_asr_model("ja") == config_module.DEFAULT_FALLBACK_ASR_MODEL


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
        "highlights": [],
    }), encoding="utf-8")

    clipper = VideoClipper.__new__(VideoClipper)
    clipper.filter_fillers = False
    clipper.segment_duration = 300
    clipper.llm_client = None
    clipper.gemini_model = None
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
        "highlights": [],
    }), encoding="utf-8")

    clipper = VideoClipper.__new__(VideoClipper)
    clipper.filter_fillers = False
    clipper.segment_duration = 300
    clipper.llm_client = None
    clipper.gemini_model = None
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


def test_cli_help_mentions_dynamic_defaults():
    """CLI source should define help text for the dynamic defaults."""
    import pycut.cli as cli_module
    import pycut.config as config_module
    source = inspect.getsource(cli_module)
    config_source = inspect.getsource(config_module)

    assert "video.parent / video.stem" in source
    assert cli_module._resolve_default_asr_model("en") == config_module.DEFAULT_EN_ASR_MODEL
    assert "Qwen3-ASR-1.7B-bf16" in config_source
    assert "whisper-large-v3-turbo" in config_source


def test_cli_help_shows_usage_examples(monkeypatch, capsys):
    """CLI help should include example commands in the description block."""
    import pycut.cli as cli_module

    monkeypatch.setattr(sys, "argv", ["pycut", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli_module.main()

    help_output = capsys.readouterr().out
    assert "Examples:" in help_output
    assert "pycut --translate --source-lang zh --target-lang en" in help_output
    assert "pycut --translate --source-lang en --target-lang zh --max-chars 50 --format video --highlight --orientation portrait ~/Movies/youtube/" in help_output


def test_generate_ass_subtitle_uses_configured_semantic_colors(tmp_path):
    import pycut.subtitle as subtitle_mod
    from pycut.models import Highlight
    from pycut.utils import Segment

    output_path = tmp_path / "output.ass"
    segments = [
        Segment(start=0.0, end=2.0, text="hello world"),
    ]
    highlights = [
        Highlight(
            start=0.0,
            end=2.0,
            title="hello",
            subtitle="",
            content="hello world",
            segment_keywords=[{"segment_id": 0, "keywords": ["world"]}],
        )
    ]

    subtitle_mod.generate_ass_subtitle(
        highlights=highlights,
        segments=segments,
        output_path=str(output_path),
        translate=True,
        subtitle_position="translated-top",
        translate_fn=lambda texts, _source, _target: [f"tr:{text}" for text in texts],
        original_subtitle_color="#112233",
        translation_subtitle_color="#445566",
        highlight_subtitle_color="#778899",
    )

    content = output_path.read_text(encoding="utf-8")

    assert "Style: OriginalTop,Arial Unicode MS,50.0,&H00332211&," in content
    assert "Style: OriginalBottom,Arial Unicode MS,35.0,&H00332211&," in content
    assert "Style: TranslationTop,Arial Unicode MS,50.0,&H00665544&," in content
    assert "Style: TranslationBottom,Arial Unicode MS,35.0,&H00665544&," in content
    assert r"{\c&H00998877&\fscx110\fscy110}world{\r}" in content
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,TranslationTop" in content
    assert "Dialogue: 0,0:00:01.00,0:00:02.00,OriginalBottom" in content
