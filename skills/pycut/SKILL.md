---
name: pycut
description: Run local pycut workflows for audio/video transcription, SRT or ASS subtitles, subtitle translation, FCPXML export, burned-in caption video rendering, transcript JSON reuse, batch media processing, and text-to-speech WAV generation. Use when a user asks to transcribe media, create or translate captions, export an editing timeline, render captions onto video, reuse a pycut transcript, process multiple local media files, or synthesize speech with pycut.
---

# Pycut

Use the installed `pycut` command for all user workflows. Keep processing local and produce only the outputs the user requests.

## Consult the Source of Truth

- Read the repository `README.md` for current installation, examples, supported platforms, and common options.
- Run `pycut --help` or `pycut tts --help` for authoritative CLI details.
- Inspect `pycut/cli.py` and `pycut/video_io.py` only when README and help output do not answer a code-level question.
- Follow the code when documentation disagrees, and mention a discrepancy only when it affects the task.

## Install and Verify

Require Python 3.12+, FFmpeg and ffprobe on `PATH`, and a supported runtime: macOS Apple Silicon, Linux, or Windows.

Check an existing installation before doing media work:

```bash
pycut --help
ffmpeg -version
ffprobe -version
```

If `pycut` is missing and the user asked to install or set it up, follow the project installation flow and use its Aliyun mirror:

```bash
# macOS FFmpeg example
brew install ffmpeg

uv tool install \
  --default-index https://mirrors.aliyun.com/pypi/simple \
  https://github.com/cliptate/pycut.git
pycut --help
```

Use the operating system's package manager to install FFmpeg on Linux or Windows. Do not install dependencies merely to answer a usage question.

After installation, invoke `pycut` directly. Do not prefix user commands with `uv run`; reserve repository-specific development commands for explicit source-development tasks.

## Run a Workflow

1. Confirm the input paths and requested outputs.
2. Select the smallest matching command below.
3. Show the command for a how-to request, or run it when the user asked to process accessible local files.
4. Confirm that the requested output files exist. Do not run model-heavy ASR, rendering, or TTS only as a smoke test.

### Transcribe and Create Subtitles

Generate reusable transcript JSON and SRT subtitles:

```bash
pycut input.mp4 \
  --source-lang en \
  --format json,srt \
  -o ./output
```

Add `txt` for a plain transcript or `ass` for styled subtitles. Use `--no-align` only when segment-level timestamps are acceptable.

### Create Bilingual Subtitles

```bash
pycut input.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --subtitle-position translated-top \
  --format ass,srt,json \
  -o ./output
```

Use `--subtitle-position original-top` to place the original text above the translation. Use ASS when styling or bilingual layout matters.

### Export an Editing Timeline

```bash
pycut input.mp4 \
  --format fcpxml,json \
  --fcpxml-frame-rate 30 \
  -o ./timeline
```

Export the complete transcript timeline. Do not claim that pycut automatically selects highlights.

### Rough-Cut an Existing FCPXML Story

Use a transcript whose timestamps align with the primary project story timeline:

```bash
pycut project.fcpxml \
  --transcript ./project_transcript.json \
  -o ./rough-cut

pycut Library.fcpxmld \
  --transcript ./project_transcript.json \
  -o ./rough-cut
```

Treat `.fcpxmld` as an FCPXML bundle containing `Info.fcpxml`; do not confuse it
with the native Final Cut Pro `.fcpbundle` library. The output is a standalone
`.fcpxml`. Empty transcript ranges are removed by default; add
`--no-filter-empty-segments` only when the user asks to retain them.

### Render Captioned Video

```bash
pycut input.mp4 \
  --format video,ass,json \
  --orientation landscape \
  -o ./rendered
```

Use `--orientation portrait` for vertical output.

### Reuse a Transcript

Prefer transcript reuse for styling changes, new output formats, and reruns that should skip ASR:

```bash
pycut input.mp4 \
  --transcript ./output/input_transcript.json \
  --format ass,srt,fcpxml \
  -o ./output-v2
```

Pass exactly one input with `--transcript`. Accept both the current object-shaped transcript JSON and the legacy segment-list format.

### Process Multiple Inputs

Accept files, directories, quoted globs, or multiple paths:

```bash
pycut ./recordings/ --format srt,json -o ./output
pycut './videos/*.mp4' --format video,srt -o ./rendered
pycut a.mp4 b.mp4 c.mp4 --format txt,json
```

Use only supported media extensions listed in README. Quote globs so pycut can resolve them consistently.

### Synthesize Speech

Use the separate TTS subcommand and write WAV output:

```bash
pycut tts --text "Hello from pycut" --output voice.wav
pycut tts --text-file script.txt --output voice.wav
pycut tts \
  --text-file script.txt \
  --reference-audio reference.wav \
  --prompt-text "Transcript of the reference voice" \
  --output cloned.wav
```

Use `pycut tts --help` before adding backend-specific model, voice, device, or sampling options.

## Handle Failures

- Resolve empty inputs by checking paths, supported extensions, and glob quoting.
- Diagnose startup failures by checking the supported runtime and installed backend dependencies.
- Reproduce probing or rendering failures with `ffmpeg -version`, `ffprobe -version`, and the exact input path.
- Use `--transcript` when a rerun should avoid ASR.
- Adjust `--margin-left` and `--margin-right` for early, late, or overlapping subtitles; both values are milliseconds.
- Warn before commands likely to download large models or perform long-running media processing when the user has not already requested that work.
