---
name: pycut
description: Use this skill whenever the user wants local pycut workflows for transcribing audio or video, generating SRT/ASS subtitles, translating subtitles, exporting FCPXML timelines, rendering burned-in subtitle videos, reusing transcript JSON, batch-processing media inputs, or synthesizing speech with `pycut tts`. This skill should trigger even when the user says this casually, for example "make subtitles from this recording", "turn this script into voice", "export a timeline", "make bilingual captions", "render captions onto the video", or "reuse this transcript".
---

# pycut Skill

Use this skill to operate or modify the local `pycut` project. In this skill, keep workflows local-first and predictable: transcription, subtitles, translation, FCPXML export, rendered video output, transcript reuse, batch processing, and text-to-speech WAV generation.

This simplified skill intentionally avoids remote content analysis features. Media commands use the local transcript timeline by default.

## Source of Truth

Prefer the current repository over memory:

- CLI entrypoint: `pycut/cli.py`
- Main media pipeline: `pycut/clipper.py`
- Media job options and output paths: `pycut/media_job.py`
- Transcript persistence: `pycut/transcript_store.py`
- Transcript timeline preparation: `pycut/timeline.py`
- TTS implementation: `pycut/tts.py`
- Runtime/model selection: `pycut/config.py`
- Supported inputs and output formats: `pycut/video_io.py`
- Subtitle/FCPXML/rendering output: `pycut/subtitle.py`, `pycut/fcpxml.py`, `pycut/renderer.py`
- Install and examples: `README.md`

If README and code disagree, follow the code and mention the discrepancy only when it affects the user's task.

## Environment Checks

Before running real media work, confirm the basics:

```bash
uv sync
uv run pycut --help
uv run pycut tts --help
ffmpeg -version
ffprobe -version
```

Do not run model-heavy transcription, rendering, or TTS commands just to prove the skill works. Run them when the user asked for actual media output or supplied a fixture where model downloads/runtime cost is acceptable.

Supported runtimes in the current code:

- macOS Apple Silicon: MLX ASR/TTS backends
- Linux/Windows: Qwen ASR and VoxCPM TTS backends
- Python 3.12+
- FFmpeg and ffprobe on `PATH`

## Command Entry Points

Use installed `pycut` for normal use:

```bash
pycut ...
pycut tts ...
```

Use the checkout for development or when working inside this repo:

```bash
uv run pycut ...
python -m pycut ...
```

`main.py` may exist in older docs as a compatibility shim, but the real console entrypoint is `pycut.cli:console_main`.

## Pick the Right Workflow

### Transcribe Only

For transcript JSON, plain text, and subtitles:

```bash
uv run pycut input.mp4 \
  --source-lang en \
  --format json,txt,srt \
  -o ./output
```

Use this when the user asks for "transcription", "captions", "subtitle file", "SRT", or "extract the text".

Useful options:

- `--source-lang zh`, `--source-lang en`, etc. controls default ASR model selection.
- `--no-align` skips forced alignment and uses segment-level timestamps.
- `--segment-duration 300` controls long-media chunk size.
- `--no-filter-fillers` keeps filler words such as "um" and "uh".
- `--max-chars 30` and `--max-duration 30` tune subtitle segmentation.

### Full Timeline Subtitles

For a complete subtitle timeline without cutting the media:

```bash
uv run pycut input.mp4 \
  --format srt,ass,json \
  -o ./output
```

Use `ass` when styling, bilingual layout, colors, or burned-in video is needed.

### Bilingual Subtitles

For translated subtitles:

```bash
uv run pycut input.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --subtitle-position translated-top \
  --format ass,srt,json \
  -o ./output
```

Color options accept `#RRGGBB`:

- `--original-subtitle-color "#FFFFFF"`
- `--translation-subtitle-color "#FFA500"`
Keep subtitle styling to local color and layout options in this simplified workflow.

### FCPXML Timeline Export

For Final Cut Pro or other FCPXML-consuming editors:

```bash
uv run pycut input.mp4 \
  --format fcpxml,json \
  --fcpxml-frame-rate 30 \
  --fcpxml-speed 1.0 \
  -o ./timeline
```

If the user asks for automatic "best moments" selection, generate a full transcript/subtitle timeline that they can review or edit manually.

### Rendered Video Output

For burned-in subtitle MP4 output:

```bash
uv run pycut input.mp4 \
  --format video,ass,json \
  --orientation landscape \
  -o ./rendered
```

Use `--orientation portrait` for vertical/short-form output.

### Transcript Reuse

Prefer transcript reuse for reruns, styling changes, new output formats, or avoiding ASR cost:

```bash
uv run pycut input.mp4 --format json -o ./output

uv run pycut input.mp4 \
  --transcript ./output/input_transcript.json \
  --format ass,srt,fcpxml \
  -o ./output-v2
```

`--transcript` only supports a single media input. The current JSON format has `title`, `subtitle`, and `segments`; legacy segment-list JSON is also accepted. Historical extra fields are ignored by this simplified skill.

### Text To Speech

For speech WAV generation:

```bash
uv run pycut tts \
  --text "Hello from pycut" \
  --output voice.wav
```

For a script file:

```bash
uv run pycut tts \
  --text-file script.txt \
  --output voice.wav
```

For voice cloning or prompt/reference audio:

```bash
uv run pycut tts \
  --text-file script.txt \
  --reference-audio reference.wav \
  --prompt-text "Transcript of the reference voice" \
  --output cloned.wav
```

Useful TTS options:

- `--tts-model` overrides the selected model.
- `--voice Chelsie`, `--lang-code English`, and `--speed 1.0` apply to MLX where supported.
- `--device cuda|cpu|mps`, `--cfg`, and `--steps` apply to VoxCPM where supported.
- `--normalize` enables backend text normalization.

## Inputs and Outputs

Supported input extensions:

- Video: `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`
- Audio: `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`

Inputs can be files, directories, globs, or multiple paths:

```bash
uv run pycut ./recordings/ --format srt,json -o ./output
uv run pycut './videos/*.mp4' --format video,srt -o ./rendered
uv run pycut a.mp4 b.mp4 c.mp4 --format txt,json
```

Supported `--format` values:

- `srt`: standard subtitle file
- `ass`: styled subtitle file
- `fcpxml`: editing timeline
- `video`: rendered MP4 with subtitles
- `txt`: plain transcript
- `json`: reusable timestamped transcript JSON

Default output directory is a sibling folder named after the input stem unless `-o/--output-dir` is provided.

## Failure Handling

Use these checks before changing code:

- If media inputs resolve to nothing, inspect extensions and quoting of glob patterns.
- If ASR/TTS fails before processing, check runtime support and installed backend dependencies.
- If FFmpeg rendering or probing fails, reproduce with `ffmpeg -version`, `ffprobe -version`, and the exact input path.
- If reruns are slow, use `--transcript` to skip ASR.
- If subtitles overlap or feel late/early, adjust `--margin-left` and `--margin-right`; CLI values are milliseconds.

## Verification

For code changes, run focused tests first, then broader tests as needed:

```bash
pytest -q
pytest -q tests/test_translation.py::test_google_translator_reports_import_error_details
pytest -q tests/test_fcpxml_naming.py
```

For CLI wiring changes, also run:

```bash
uv run pycut --help
uv run pycut tts --help
```

For media-output changes, prefer a small local fixture and verify that the requested output paths exist. Avoid asserting semantic ASR/TTS quality unless the test is explicitly designed for a fixed model and fixture.
