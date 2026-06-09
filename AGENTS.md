# Copilot Instructions for `pycut`

## Commands

- Install dependencies for local development with `uv sync`. This is the install flow documented in `README.md`.
- Run the CLI during development with `uv run pycut ...`, `python -m pycut ...`, or `python main.py ...`. `main.py` is only a compatibility shim; the real CLI entrypoint is `pycut.cli:main`.
- Run the full test suite with `pytest -q`.
- Run a single test with `pytest -q tests/test_translation.py::test_google_translator_reports_import_error_details`.

## High-level architecture

- `pycut/cli.py` parses CLI arguments, expands file/dir/glob inputs through `pycut.video_io`, and hands each resolved media file to `VideoClipper.process_video(...)`.
- `pycut/clipper.py` is the orchestration layer. `VideoClipper` owns the local end-to-end pipeline: runtime validation, audio extraction, ASR, optional translation, subtitle generation, FCPXML export, and rendered video output.
- `pycut/asr.py` contains `MLXASRHelper` and `QwenASRHelper`, which lazily load ASR, aligner, and VAD resources. The pipeline intentionally unloads ASR models after transcription to save memory before later stages.
- `pycut/utils.py` defines the shared `Segment` model used across transcription and export paths, while `pycut/models.py` defines `Highlight` as an internal timeline range container for subtitle/FCPXML/rendering paths.
- `pycut/subtitle.py`, `pycut/fcpxml.py`, and `pycut/renderer.py` are the output stages. They consume shared `Segment` / `Highlight` data and generate ASS/SRT, FCPXML, or ffmpeg-rendered MP4 output.
- Transcript reuse is a first-class flow: `--transcript` loads prior JSON through `pycut.video_io._load_segments_from_transcript_json(...)` so ASR can be skipped on reruns.

## Key conventions

- Treat `pycut/clipper.py` as the source of truth for workflow changes. If behavior spans ASR, translation, and output generation, it is usually coordinated there rather than spread across multiple entrypoints.
- This project is intentionally platform-gated. `pycut.config.ensure_supported_runtime()` supports macOS Apple Silicon, Linux, and Windows. Do not add unsupported runtime assumptions without updating that guard and its tests.
- `ffmpeg`/`ffprobe` are hard runtime dependencies even though they are not Python packages. Audio extraction, duration probing, segment splitting, and rendered video output all shell out to them.
- Output formats are validated centrally in `pycut/video_io.py`. Reuse `_parse_output_formats(...)` / `_normalize_output_formats(...)`, and update `SUPPORTED_OUTPUT_FORMATS` / `DEFAULT_OUTPUT_FORMATS` there when adding new formats.
- Transcript JSON is not just a raw segment list anymore. `video_io._load_segments_from_transcript_json(...)` accepts both the legacy list format and the current object format with `title`, `subtitle`, `segments`, and `highlights`.
- Segment cleanup happens before exports. `VideoClipper._filter_subtitle_segments(...)` removes empty segments by default, and `_resolve_overlaps(...)` applies margin shifts plus midpoint overlap splitting. CLI margins are in milliseconds, but `cli.py` converts them to seconds before passing them into the pipeline.
- Tests live under `tests/` and are plain `pytest` tests. `test_translation.py` covers translator import/retry behavior, `test_fcpxml_naming.py` covers FCPXML naming/styling behavior, and `test_pycut.py` covers broader runtime/configuration behavior rather than full end-to-end media processing.
