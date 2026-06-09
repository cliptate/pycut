# pycut

> Lokales Medienwerkzeug für Transkription, Untertitel, Schnitt-Timelines und Text-to-Speech.

**Sprachen:** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut` verarbeitet Video- und Audiodateien lokal. Die CLI kann transkribieren, Untertitel und FCPXML-Timelines erzeugen, Untertitel in Videos einbrennen und Sprachdateien aus Text erzeugen.

## Funktionen

- Lokale ASR mit MLX/Qwen-Modellen
- Export nach `srt`, `ass`, `fcpxml`, `video`, `txt`, `json`
- Optional übersetzte und zweisprachige Untertitel
- Wiederverwendung von Transcript-JSON mit `--transcript`
- Batch-Eingaben über Dateien, Ordner und Globs
- Lokales Text-to-Speech über `pycut tts`

## Voraussetzungen

| Punkt | Anforderung |
| --- | --- |
| Betriebssystem | macOS, Linux oder Windows; MLX-Beschleunigung ist auf Apple Silicon am besten |
| Python | 3.12+ |
| FFmpeg | Muss installiert und in `PATH` verfügbar sein |

## Installation

```bash
brew install ffmpeg
uv tool install https://github.com/cliptate/pycut.git
```

Für lokale Entwicklung:

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
uv run pycut --help
```

## Schnellstart

Transcript-JSON und SRT erzeugen:

```bash
pycut my_video.mp4 --source-lang en --format json,srt
```

Video mit zweisprachigen Untertiteln exportieren:

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
pycut tts "Hallo Welt" --output hello.wav
```

## Wichtige Optionen

| Option | Standard | Beschreibung |
| --- | --- | --- |
| `--transcript` | keiner | Vorhandenes Transcript-JSON wiederverwenden |
| `--format` | `srt` | Ausgabeformate: `ass,srt,fcpxml,video,txt,json` |
| `--translate` | aus | Untertitel übersetzen |
| `--source-lang` | `en` | Quellsprache |
| `--target-lang` | `en` | Zielsprache |
| `--orientation` | `landscape` | `landscape` oder `portrait` |

## Pipeline

```text
Medien
  -> Audio-Extraktion
  -> lokale ASR + Alignment
  -> optionale Übersetzung
  -> Export als SRT / ASS / FCPXML / MP4 / TXT / JSON
```

## License

MIT
