# pycut

> Local video/audio transcription, subtitle, timeline, rendering, and speech tools.

**Languages:** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` transcribes long-form video or audio locally, exports subtitles and editing timelines, renders burned-in subtitle videos, and provides a separate text-to-speech command.

## Features

- Local ASR selected by system: MLX on Apple Silicon, Qwen3-ASR on Linux/Windows
- Automatic ECAPA-TDNN spoken-language detection on Apple Silicon when `--source-lang` is omitted
- Separate `pycut tts` command for WAV speech generation
- Translation and bilingual subtitle layouts
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

ASR/TTS models are selected automatically from the current system. Intel Macs and unsupported systems are rejected at runtime.

When a complete model snapshot already exists under the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`), `pycut` uses that local snapshot path before attempting any network download. Incomplete snapshots are skipped; for Qwen ASR on Linux/Windows, a complete cached `Qwen/Qwen3-ASR-0.6B` is used when the default `Qwen/Qwen3-ASR-1.7B` cache is incomplete.

## Install

Install FFmpeg:

```bash
brew install ffmpeg
```

Install `pycut`:

```bash
uv tool install https://github.com/cliptate/pycut.git
```

For local development:

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
uv run pycut --help
```

## Quick Start

Generate subtitles and reusable transcript JSON:

```bash
pycut my_video.mp4 --source-lang en --format srt,json
```

On Apple Silicon, omit `--source-lang` to detect the spoken language with
`beshkenadze/lang-id-voxlingua107-ecapa-mlx`. Linux and Windows require an
explicit `--source-lang` because MLX language detection is unavailable there.

Create bilingual subtitles:

```bash
pycut my_video.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format ass,srt,json
```

Render a portrait video with burned-in subtitles:

```bash
pycut lecture.mp4 \
  --orientation portrait \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --subtitle-position translated-top \
  --format video,ass,json
```

Export an FCPXML timeline:

```bash
pycut my_video.mp4 \
  --format fcpxml,json \
  --fcpxml-frame-rate 30
```

Rough-cut an existing Final Cut Pro story timeline from an aligned transcript:

```bash
pycut project.fcpxml --transcript project_transcript.json -o ./rough-cut
pycut Library.fcpxmld --transcript project_transcript.json -o ./rough-cut
pycut project.fcpxml --transcript project_transcript.json --translate --source-lang en --target-lang zh -o ./rough-cut
```

For `.fcpxmld` bundles, pycut reads the root `Info.fcpxml` document. Transcript
timestamps must align with the primary project story timeline. By default, empty
transcript ranges are removed; pass `--no-filter-empty-segments` to retain them.
Existing resources, effects, clip settings, and project metadata are preserved;
transcript titles (plus translations when requested) are added to the rough cut.
The result is a standalone `.fcpxml` file. Native Final Cut Pro `.fcpbundle`
libraries are not FCPXML input.

Reuse an existing transcript and skip ASR:

```bash
pycut video.mp4 --format json -o ./output

pycut video.mp4 \
  --transcript ./output/video_transcript.json \
  --format video,srt,fcpxml \
  -o ./output-v2
```

## CLI Usage

```text
pycut <video-file|directory|glob> [options]
```

Development entrypoints from a local checkout:

- `uv run pycut ...`
- `python -m pycut ...`

Input expansion supports:

- A single file: `video.mp4`
- A directory: `./videos/`
- A glob: `./recordings/*.mp4`
- Multiple inputs: `a.mp4 b.mp4 c.mp4`

Supported inputs:

- Video: `mp4`, `mov`, `mkv`, `avi`, `m4v`, `webm`
- Audio: `wav`, `mp3`, `m4a`, `aac`, `flac`, `ogg`
- Final Cut Pro XML: `.fcpxml` document or `.fcpxmld` bundle

## Common Options

### Input and Output

| Option | Default | Description |
| --- | --- | --- |
| `video_inputs` | required | Media files, `.fcpxml` documents, `.fcpxmld` bundles, directories, or glob patterns |
| `-o, --output-dir` | sibling folder named after the input stem | Output directory |
| `--transcript JSON_FILE` | none | Reuse an existing transcript JSON and skip ASR |
| `--format` | `srt` | Comma-separated output formats |

### ASR

| Option | Default | Description |
| --- | --- | --- |
| `--asr-model` | auto by source language | `en` uses Parakeet, `zh*` uses Qwen3 ASR, others use Whisper Large v3 Turbo |
| `--aligner-model` | auto by system | macOS uses MLX Qwen3 aligner; Linux/Windows use `Qwen/Qwen3-ForcedAligner-0.6B` |
| `--no-align` | off | Skip forced alignment and use segment-level timestamps |
| `--segment-duration` | `300` | Transcript chunk size in seconds for long media |
| `--no-filter-fillers` | off | Keep filler words such as `um` / `uh` |

### Subtitle and Styling

| Option | Default | Description |
| --- | --- | --- |
| `--translate` | off | Translate subtitles |
| `--source-lang` | auto on Apple Silicon | Source language code; required on Linux/Windows |
| `--target-lang` | `en` | Target language code |
| `--subtitle-position` | `translated-top` | Bilingual subtitle stacking |
| `--original-subtitle-color` | `#FFFFFF` | Original subtitle color |
| `--translation-subtitle-color` | `#FFA500` | Translation subtitle color |
| `--max-duration` | `30.0` | Maximum subtitle segment duration in seconds |
| `--max-chars` | `30` | Maximum characters per subtitle segment |
| `--first-subtitle-delay` | `1.0` | Delay before the first subtitle frame |
| `--no-filter-empty-segments` | off | Keep empty subtitle segments |
| `--margin-left` | `-100` | Start shift in milliseconds |
| `--margin-right` | `150` | End shift in milliseconds |

### Rendering and Editing Exports

| Option | Default | Description |
| --- | --- | --- |
| `--orientation` | `landscape` | `landscape` or `portrait` output |
| `--fcpxml-frame-rate` | `25.0` | FCPXML frame rate |
| `--fcpxml-speed` | `1.0` | FCPXML timeline speed multiplier |

## Output Formats

| Format | Description |
| --- | --- |
| `srt` | Standard subtitle file |
| `ass` | Styled subtitle file with bilingual layout |
| `fcpxml` | Timeline export for Final Cut Pro / DaVinci Resolve |
| `video` | Burned-in MP4 output |
| `txt` | Plain transcript |
| `json` | Timestamped transcript JSON reusable with `--transcript` |

## TTS Command

`pycut tts` is separate from video processing and writes a WAV file:

```bash
pycut tts --text "Hello from pycut" --output voice.wav
pycut tts --text-file script.txt --output voice.wav
pycut tts --text "Hello in the cloned voice" --reference-audio reference.wav --prompt-text "Reference transcript" --output voice.wav
```

On MLX Audio, multi-segment generation is joined into the requested WAV by default. Use
`--no-join-audio` to fall back to pycut's local chunk writer for compatibility checks.

Default TTS models:

| System | Backend | Default model |
| --- | --- | --- |
| macOS Apple Silicon | MLX Audio | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |
| Linux / Windows | VoxCPM | `openbmb/VoxCPM2` |

## Pipeline

```text
media
  -> audio extraction
  -> ASR + alignment
  -> optional translation
  -> subtitle generation
  -> SRT / ASS / FCPXML / MP4 / TXT / JSON outputs
```

## License

MIT
