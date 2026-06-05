# pycut

> 面向 Apple Silicon Mac 的 AI 自动剪辑工具。

**语言：** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` 可以把长视频或音频自动转录、用兼容 OpenAI 的大模型提取高光片段，并导出字幕、时间线或烧录视频，全部通过一条命令完成。

## 功能

- 基于 MLX 的本地 ASR，针对 Apple Silicon 优化
- AI 自动提取精彩片段并生成标题
- 支持翻译与双语字幕布局
- 支持字幕关键词高亮
- 支持 `srt`、`ass`、`fcpxml`、`video`、`txt`、`json`
- 支持横屏和竖屏输出
- 可复用转录 JSON，跳过重复 ASR
- 处理流程按阶段卸载模型，更适合长视频

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | macOS Apple Silicon（`arm64` / `aarch64`） |
| Python | 3.12+ |
| FFmpeg | 已安装并可在 `PATH` 中找到 |
| API Key | 仅在 AI 裁剪、关键词高亮或转录纠错时需要 |

`pycut` 当前仅支持 macOS + Apple Silicon，不支持 Intel Mac 或其他系统。

## 安装

### 1. 安装 FFmpeg

```bash
brew install ffmpeg
```

### 2. 安装 `pycut`

日常使用推荐直接安装命令行工具：

```bash
uv tool install https://github.com/cliptate/pycut.git
```

安装完成后可直接使用：

```bash
pycut --help
```

后续更新时，继续使用 `uv tool` 重新安装或升级即可。

`pycut` 会根据包元数据安装 Python 运行时依赖，其中包括 VAD 转录所需的 `soundfile`。如果旧的工具环境里缺少它，重新安装工具，或执行 `uv tool install --reinstall https://github.com/cliptate/pycut.git`。

### 3. 克隆仓库用于本地开发

如果你要调试、修改代码或运行测试，克隆仓库更合适：

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
```

然后在仓库目录内运行：

```bash
uv run pycut --help
```

## 配置 API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

也可以在运行时传入 `--api-key`。如果使用 Gemini、DeepSeek 等兼容 OpenAI 的接口，增加 `--base-url` 即可：

```bash
pycut input.mp4 \
  --api-key YOUR_KEY \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai
```

## 快速开始

提取高光并输出视频加字幕：

```bash
pycut my_video.mp4 \
  --api-key YOUR_KEY \
  --format video,srt
```

只生成字幕，不做 AI 剪辑：

```bash
pycut my_video.mp4 --no-clip --format srt
```

生成双语字幕：

```bash
pycut my_video.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format video,srt
```

如果你是从源码仓库中运行，请把上面的 `pycut` 替换为 `uv run pycut`。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--transcript` | 无 | 复用已有转录 JSON，跳过 ASR |
| `--format` | `srt` | 输出格式，支持 `ass,srt,fcpxml,video,txt,json` |
| `--no-align` | 关闭 | 跳过强制时间对齐，使用分段级时间戳 |
| `--api-key` | 环境变量或无 | OpenAI 兼容 API Key |
| `--no-clip` | 关闭 | 不做 AI 高光裁剪，保留完整字幕时间线 |
| `--highlight` | 关闭 | 在 `--no-clip` 模式下做关键词高亮 |
| `--correct-words` | 关闭 | 用 LLM 修正 ASR 错词 |
| `--translate` | 关闭 | 启用字幕翻译 |
| `--source-lang` | `en` | 源语言，英文为默认语言 |
| `--target-lang` | `en` | 目标语言 |
| `--orientation` | `landscape` | `landscape` 或 `portrait` |
| `--fcpxml-frame-rate` | `25.0` | FCPXML 帧率 |

## 输出格式

| 格式 | 说明 |
| --- | --- |
| `srt` | 标准字幕 |
| `ass` | 带样式和高亮的高级字幕 |
| `fcpxml` | Final Cut Pro / DaVinci Resolve 时间线 |
| `video` | 烧录字幕的 MP4 视频 |
| `txt` | 纯文本转录 |
| `json` | 可供 `--transcript` 复用的时间戳转录 JSON |

## TTS 命令

`pycut tts` 独立于视频剪辑流程，用于生成 WAV 语音文件：

```bash
pycut tts --text "你好，pycut" --output voice.wav
pycut tts --text-file script.txt --output voice.wav
pycut tts --text "使用克隆声音生成这句话" --reference-audio reference.wav --prompt-text "参考音频对应文本" --output voice.wav
```

## 示例

竖屏双语短视频：

```bash
pycut lecture.mp4 \
  --api-key YOUR_KEY \
  --orientation portrait \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format video,ass
```

复用已有转录：

```bash
pycut video.mp4 --format json -o ./output

pycut video.mp4 \
  --transcript ./output/video.json \
  --api-key YOUR_KEY \
  --format video,srt
```

## 处理流程

```text
媒体输入
  -> 音频提取
  -> ASR + 对齐
  -> 可选 AI 高光提取 / 关键词检测 / 错词纠正
  -> 字幕生成
  -> 导出 SRT / ASS / FCPXML / MP4 / TXT / JSON
```

## License

MIT
