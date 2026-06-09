# pycut

> 文字起こし、字幕、編集タイムライン、Text-to-Speech のためのローカルメディアツール。

**言語:** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` は動画や音声をローカルで処理します。CLI から文字起こし、字幕生成、FCPXML タイムライン出力、字幕焼き込み動画、音声合成を実行できます。

## 主な機能

- MLX/Qwen モデルによるローカル ASR
- `srt`、`ass`、`fcpxml`、`video`、`txt`、`json` 出力
- 任意の翻訳と二言語字幕
- `--transcript` による transcript JSON の再利用
- ファイル、フォルダ、glob の一括入力
- `pycut tts` によるローカル Text-to-Speech

## 動作要件

| 項目 | 要件 |
| --- | --- |
| OS | macOS、Linux、Windows。MLX 高速化は Apple Silicon が最適 |
| Python | 3.12+ |
| FFmpeg | インストール済みで `PATH` から利用可能 |

## インストール

```bash
brew install ffmpeg
uv tool install https://github.com/cliptate/pycut.git
```

ローカル開発:

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
uv run pycut --help
```

## クイックスタート

transcript JSON と SRT を生成:

```bash
pycut my_video.mp4 --source-lang en --format json,srt
```

二言語字幕付き動画を出力:

```bash
pycut lecture.mp4 \
  --translate \
  --source-lang en \
  --target-lang zh-CN \
  --format video,ass \
  --orientation portrait
```

Text-to-Speech:

```bash
pycut tts "こんにちは" --output hello.wav
```

## よく使うオプション

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--transcript` | なし | 既存の transcript JSON を再利用 |
| `--format` | `srt` | 出力形式: `ass,srt,fcpxml,video,txt,json` |
| `--translate` | off | 字幕翻訳を有効化 |
| `--source-lang` | `en` | 入力言語 |
| `--target-lang` | `en` | 出力言語 |
| `--orientation` | `landscape` | `landscape` または `portrait` |

## パイプライン

```text
メディア入力
  -> 音声抽出
  -> ローカル ASR + アラインメント
  -> 任意の翻訳
  -> SRT / ASS / FCPXML / MP4 / TXT / JSON を出力
```

## License

MIT
