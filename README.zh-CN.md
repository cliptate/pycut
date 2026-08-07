# pycut

> 本地视频/音频转录、字幕、时间线、渲染与语音工具。

**语言：** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` 可以在本地转录长视频或音频，导出字幕和剪辑时间线，渲染烧录字幕的视频，并提供独立的文字转语音命令。

## 功能

- 按系统自动选择本地 ASR：Apple Silicon 使用 MLX，Linux/Windows 使用 Qwen3-ASR
- 独立的 `pycut tts` 命令，可生成 WAV 语音
- 支持翻译与双语字幕布局
- 支持 `srt`、`ass`、`fcpxml`、`video`、`txt`、`json`
- 支持横屏和竖屏输出
- 可复用转录 JSON，跳过重复 ASR
- 处理流程按阶段卸载模型，更适合长视频

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | macOS Apple Silicon（`arm64` / `aarch64`）、Linux 或 Windows |
| Python | 3.12+ |
| FFmpeg | 已安装并可在 `PATH` 中找到 |

ASR/TTS 模型会根据当前系统自动选择。Intel Mac 和不支持的系统会在运行时被拒绝。

## 安装

安装 FFmpeg：

```bash
brew install ffmpeg
```

安装 `pycut`：

```bash
uv tool install https://github.com/cliptate/pycut.git
```

本地开发：

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
uv run pycut --help
```

## 快速开始

生成字幕和可复用转录 JSON：

```bash
pycut my_video.mp4 --source-lang en --format srt,json
```

在 Apple Silicon 上可省略 `--source-lang`，此时使用
`beshkenadze/lang-id-voxlingua107-ecapa-mlx` 自动检测语音语言。Linux 和
Windows 不支持 MLX 语言检测，必须显式指定 `--source-lang`。

生成双语字幕：

```bash
pycut my_video.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format ass,srt,json
```

生成竖屏烧录字幕视频：

```bash
pycut lecture.mp4 \
  --orientation portrait \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --subtitle-position translated-top \
  --format video,ass,json
```

导出 FCPXML 时间线：

```bash
pycut my_video.mp4 \
  --format fcpxml,json \
  --fcpxml-frame-rate 30
```

根据已有时间戳转录，对 Final Cut Pro 主故事时间线进行粗剪：

```bash
pycut project.fcpxml --transcript project_transcript.json -o ./rough-cut
pycut Library.fcpxmld --transcript project_transcript.json -o ./rough-cut
```

对于 `.fcpxmld` bundle，pycut 会读取根目录的 `Info.fcpxml`。转录时间戳
必须与第一个项目的主故事时间线对齐。默认移除文本为空的时间段；使用
`--no-filter-empty-segments` 可保留。资源与项目元数据保持不变，输出为独立
`.fcpxml` 文件。Final Cut Pro 原生 `.fcpbundle` 资料库不是 FCPXML 输入。

复用已有转录并跳过 ASR：

```bash
pycut video.mp4 --format json -o ./output

pycut video.mp4 \
  --transcript ./output/video_transcript.json \
  --format video,srt,fcpxml \
  -o ./output-v2
```

如果你是从源码仓库中运行，请把上面的 `pycut` 替换为 `uv run pycut`。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--transcript` | 无 | 复用已有转录 JSON，跳过 ASR |
| `--format` | `srt` | 输出格式，支持 `ass,srt,fcpxml,video,txt,json` |
| `--asr-model` | 自动 | 覆盖 ASR 模型路径 |
| `--aligner-model` | 自动 | 覆盖对齐模型路径 |
| `--no-align` | 关闭 | 跳过强制时间对齐，使用分段级时间戳 |
| `--segment-duration` | `300` | 长媒体转录分块时长，单位秒 |
| `--translate` | 关闭 | 启用字幕翻译 |
| `--source-lang` | Apple Silicon 自动检测 | 源语言；Linux/Windows 必填 |
| `--target-lang` | `en` | 目标语言 |
| `--subtitle-position` | `translated-top` | 双语字幕上下位置 |
| `--original-subtitle-color` | `#FFFFFF` | 原文字幕颜色 |
| `--translation-subtitle-color` | `#FFA500` | 译文字幕颜色 |
| `--orientation` | `landscape` | `landscape` 或 `portrait` |
| `--fcpxml-frame-rate` | `25.0` | FCPXML 帧率 |
| `--fcpxml-speed` | `1.0` | FCPXML 时间线速度倍率 |

## 输出格式

| 格式 | 说明 |
| --- | --- |
| `srt` | 标准字幕 |
| `ass` | 带样式和双语布局的高级字幕 |
| `fcpxml` | Final Cut Pro / DaVinci Resolve 时间线 |
| `video` | 烧录字幕的 MP4 视频 |
| `txt` | 纯文本转录 |
| `json` | 可供 `--transcript` 复用的时间戳转录 JSON |

## TTS 命令

`pycut tts` 独立于视频处理流程，用于生成 WAV 语音文件：

```bash
pycut tts --text "你好，pycut" --output voice.wav
pycut tts --text-file script.txt --output voice.wav
pycut tts --text "使用克隆声音生成这句话" --reference-audio reference.wav --prompt-text "参考音频对应文本" --output voice.wav
```

在 MLX Audio 后端，多段生成默认会合并到指定的 WAV 文件。可使用
`--no-join-audio` 回退到 pycut 本地 chunk 写入逻辑，便于兼容性检查。

## 处理流程

```text
媒体输入
  -> 音频提取
  -> ASR + 对齐
  -> 可选翻译
  -> 字幕生成
  -> 导出 SRT / ASS / FCPXML / MP4 / TXT / JSON
```

## License

MIT
