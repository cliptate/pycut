# pycut

> 전사, 자막, 편집 타임라인, Text-to-Speech를 위한 로컬 미디어 도구.

**언어:** [English](README.md) | [中文](README.zh-CN.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

`pycut`은 영상과 오디오를 로컬에서 처리합니다. CLI로 전사, 자막 생성, FCPXML 타임라인 출력, 자막이 입혀진 비디오, 음성 합성을 실행할 수 있습니다.

## 주요 기능

- MLX/Qwen 모델 기반 로컬 ASR
- `srt`, `ass`, `fcpxml`, `video`, `txt`, `json` 출력
- 선택적 번역과 이중 언어 자막
- `--transcript`로 transcript JSON 재사용
- 파일, 폴더, glob 일괄 입력
- `pycut tts` 로컬 Text-to-Speech

## 요구 사항

| 항목 | 요구 사항 |
| --- | --- |
| OS | macOS, Linux, Windows. MLX 가속은 Apple Silicon에서 가장 적합 |
| Python | 3.12+ |
| FFmpeg | 설치되어 있고 `PATH`에서 실행 가능해야 함 |

## 설치

```bash
brew install ffmpeg
uv tool install https://github.com/cliptate/pycut.git
```

로컬 개발:

```bash
git clone https://github.com/cliptate/pycut.git
cd pycut
uv sync
uv run pycut --help
```

## 빠른 시작

transcript JSON과 SRT 생성:

```bash
pycut my_video.mp4 --source-lang en --format json,srt
```

이중 언어 자막 비디오 출력:

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
pycut tts "안녕하세요" --output hello.wav
```

## 자주 쓰는 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--transcript` | 없음 | 기존 transcript JSON 재사용 |
| `--format` | `srt` | 출력 형식: `ass,srt,fcpxml,video,txt,json` |
| `--translate` | 꺼짐 | 자막 번역 활성화 |
| `--source-lang` | `en` | 원본 언어 |
| `--target-lang` | `en` | 대상 언어 |
| `--orientation` | `landscape` | `landscape` 또는 `portrait` |

## 파이프라인

```text
미디어 입력
  -> 오디오 추출
  -> 로컬 ASR + 정렬
  -> 선택적 번역
  -> SRT / ASS / FCPXML / MP4 / TXT / JSON 출력
```

## License

MIT
