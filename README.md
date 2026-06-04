# pycut

> AI-powered video clipping and local speech tools.

**Languages:** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` transcribes long-form video or audio, extracts highlight-worthy moments with an OpenAI-compatible LLM, and exports subtitles, timelines, or burned-in videos from a single CLI.

## Features

- Local ASR selected by system: MLX on Apple Silicon, Qwen3-ASR on Linux/Windows
- Separate `pycut tts` command for WAV speech generation
- AI highlight extraction and auto-generated titles
- Translation and bilingual subtitle layouts
- Keyword highlighting for important words inside subtitles
- Multiple export targets: `srt`, `ass`, `fcpxml`, `video`, `txt`, `json`
- Landscape and portrait output support
- Transcript JSON reuse to skip ASR on reruns
- Memory-aware pipeline that unloads models between stages

## Requirements

| Item | Requirement |
| --- | --- |
| OS | macOS on Apple Silicon (`arm64` / `aarch64`), Linux, or Windows |
| Python | 3.12+ |
| FFmpeg | Must be installed and available in `PATH` |
| API key | Required only for AI highlight extraction, keyword highlighting, or transcript correction |

ASR/TTS models are selected automatically from the current system. Intel Macs and unsupported systems are rejected at runtime.

When a complete model snapshot already exists under the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`), `pycut` uses that local snapshot path before attempting any network download. Incomplete snapshots are skipped; for Qwen ASR on Linux/Windows, a complete cached `Qwen/Qwen3-ASR-0.6B` is used when the default `Qwen/Qwen3-ASR-1.7B` cache is incomplete.

## Install

### 1. Install FFmpeg

```bash
brew install ffmpeg
```

### 2. Install `pycut`

Recommended for normal usage:

```bash
uv tool install https://github.com/cliptate/pycut.git
```

After installation, you can run `pycut` directly:

```bash
pycut --help
pycut tts --help
```

If you update later, reinstall or upgrade the tool with `uv tool`.

`pycut` installs its Python runtime dependencies, including `soundfile` for VAD-based transcription, from package metadata. If an older tool environment is missing it, reinstall the tool or run `uv tool install --reinstall https://github.com/cliptate/pycut.git`.

The project configures uv to resolve packages from the Alibaba Cloud PyPI mirror (`https://mirrors.aliyun.com/pypi/simple`) to speed up dependency downloads.

### 3. Clone the repository for local development

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
```

For local development:

```bash
uv sync
```

Then run the CLI from the repo checkout with:

```bash
uv run pycut --help
```

## Configure an API key

Set an OpenAI-compatible API key if you want AI-assisted clipping, keyword highlighting, or transcript correction:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can also pass it per run with `--api-key`. To use a compatible provider such as Gemini or DeepSeek, add `--base-url` and optionally `--model`.

```bash
# Gemini
pycut input.mp4 \
  --api-key YOUR_KEY \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai

# DeepSeek
pycut input.mp4 \
  --api-key YOUR_KEY \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-flash
```

## Quick start

Extract AI-selected highlights and export a rendered video plus subtitles:

```bash
pycut my_video.mp4 \
  --api-key YOUR_KEY \
  --format video,srt
```

Generate subtitles only, without highlight clipping:

```bash
pycut my_video.mp4 --no-clip --format srt
```

Create bilingual subtitles:

```bash
pycut my_video.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format video,srt
```

Export an FCPXML timeline:

```bash
pycut my_video.mp4 \
  --api-key YOUR_KEY \
  --format fcpxml \
  --fcpxml-frame-rate 30
```

## CLI usage

```text
pycut <video-file|directory|glob> [options]
```

When installed with `uv tool`, use:

- `pycut ...`

Development entrypoints from a local checkout:

- `uv run pycut ...`
- `python -m pycut ...`
- `python main.py ...` for compatibility only

Input expansion supports:

- A single file: `video.mp4`
- A directory: `./videos/`
- A glob: `./recordings/*.mp4`
- Multiple inputs: `a.mp4 b.mp4 c.mp4`

Supported media extensions:

- Video: `mp4`, `mov`, `mkv`, `avi`, `m4v`, `webm`
- Audio: `wav`, `mp3`, `m4a`, `aac`, `flac`, `ogg`

## Common options

### Input and output

| Option | Default | Description |
| --- | --- | --- |
| `video_inputs` | required | Media files, directories, or glob patterns |
| `-o, --output-dir` | sibling folder named after the input stem | Output directory |
| `--transcript JSON_FILE` | none | Reuse an existing transcript JSON and skip ASR |
| `--format` | `srt` | Comma-separated output formats |

### ASR

| Option | Default | Description |
| --- | --- | --- |
| `--asr-model` | auto by source language | `en` uses Parakeet, `zh*` uses Qwen3 ASR, others use Whisper Large v3 Turbo |
| `--aligner-model` | auto by system | macOS uses MLX Qwen3 aligner; Linux/Windows use `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--segment-duration` | `300` | Audio chunk size in seconds for long media |
| `--no-filter-fillers` | off | Keep filler words such as `um` / `uh` |

### AI analysis

| Option | Default | Description |
| --- | --- | --- |
| `--api-key` | env or none | OpenAI-compatible API key |
| `--base-url` | OpenAI endpoint when omitted | Compatible API base URL |
| `--model` | provider default | LLM model name |
| `--no-clip` | off | Disable AI highlight extraction and keep the full subtitle timeline |
| `--highlight` | off | Detect subtitle keywords in `--no-clip` mode |
| `--correct-words` | off | Use the LLM to correct ASR mistakes and print a diff |

### Subtitle and styling

| Option | Default | Description |
| --- | --- | --- |
| `--translate` | off | Translate subtitles |
| `--source-lang` | `en` | Source language code |
| `--target-lang` | `en` | Target language code |
| `--subtitle-position` | `translated-top` | Bilingual subtitle stacking |
| `--original-subtitle-color` | `#FFFFFF` | Original subtitle color |
| `--translation-subtitle-color` | `#FFA500` | Translation subtitle color |
| `--highlight-subtitle-color` | `#FFFF00` | Keyword highlight color |
| `--max-duration` | `30.0` | Maximum subtitle segment duration in seconds |
| `--max-chars` | `30` | Maximum characters per subtitle segment |
| `--first-subtitle-delay` | `1.0` | Delay before the first subtitle frame |
| `--max-title-chars` | `6` | Maximum highlight title length |
| `--max-subtitle-chars` | `10` | Maximum highlight subtitle length |
| `--no-filter-empty-segments` | off | Keep empty subtitle segments |
| `--margin-left` | `-100` | Start shift in milliseconds |
| `--margin-right` | `150` | End shift in milliseconds |

### Rendering and editing exports

| Option | Default | Description |
| --- | --- | --- |
| `--orientation` | `landscape` | `landscape` or `portrait` output |
| `--fcpxml-frame-rate` | `25.0` | FCPXML frame rate |
| `--fcpxml-speed` | `1.0` | FCPXML timeline speed multiplier |

## Output formats

| Format | Description |
| --- | --- |
| `srt` | Standard subtitle file |
| `ass` | Styled subtitle file with bilingual layout and highlighting |
| `fcpxml` | Timeline export for Final Cut Pro / DaVinci Resolve |
| `video` | Burned-in MP4 output |
| `txt` | Plain transcript |
| `json` | Timestamped transcript JSON reusable with `--transcript` |

## TTS command

`pycut tts` is separate from video clipping and writes a WAV file:

```bash
pycut tts --text "Hello from pycut" --output voice.wav
pycut tts --text-file script.txt --output voice.wav
```

Default TTS models:

| System | Backend | Default model |
| --- | --- | --- |
| macOS Apple Silicon | MLX Audio | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |
| Linux / Windows | VoxCPM | `openbmb/VoxCPM2` |

## Examples

Process every file in a directory:

```bash
pycut ./recordings/ \
  --api-key YOUR_KEY \
  --format video,srt,json \
  -o ./output
```

Portrait short-form video with bilingual subtitles:

```bash
pycut lecture.mp4 \
  --api-key YOUR_KEY \
  --orientation portrait \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --subtitle-position translated-top \
  --format video,ass
```

Reuse an existing transcript and skip ASR:

```bash
pycut video.mp4 --format json -o ./output

pycut video.mp4 \
  --transcript ./output/video.json \
  --api-key YOUR_KEY \
  --format video,srt
```

Keep the full timeline but still highlight keywords:

```bash
pycut interview.mp4 \
  --no-clip \
  --highlight \
  --api-key YOUR_KEY \
  --format ass,srt
```

Translate Chinese speech to English and export FCPXML:

```bash
pycut ~/Movies/vad_example.wav \
  --translate \
  --source-lang zh \
  --target-lang en \
  --no-clip \
  --highlight \
  --api-key YOUR_KEY \
  --format fcpxml \
  -o ~/Movies/youtube/
```

Correct transcript wording before exporting subtitles:

```bash
pycut podcast.mp4 \
  --api-key YOUR_KEY \
  --correct-words \
  --no-clip \
  --format srt,json
```

## Pipeline

```text
media
  -> audio extraction
  -> ASR + alignment
  -> optional AI highlight extraction / keyword detection / transcript correction
  -> subtitle generation
  -> SRT / ASS / FCPXML / MP4 / TXT / JSON outputs
```

## License

MIT
