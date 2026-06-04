#!/usr/bin/env python3
# coding=utf-8
"""
Shared utilities for text processing, segment building, SRT formatting,
audio helpers, and common data classes.

Used by both ``server.py`` (repo root) and ``scripts/clipping.py``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text processing constants
# ---------------------------------------------------------------------------

_ASCII_WORD_RE = re.compile(r"^[A-Za-z0-9]+$")
_ASCII_CHAR_RE = re.compile(r"[a-zA-Z0-9]")
_END_PUNCT = set("。！？!?." + "；;")
_SOFT_PUNCT = set("，,、:：")
_PUNCT_ALL = _END_PUNCT | _SOFT_PUNCT
_FILLER_WORDS = {
    "um", "uh", "erm", "ah", "eh",
    "额", "啊", "嗯", "呃", "唔",
}

# POS-based split weights matching Swift SegmentSplitByNatural.tagWeights.
# Applied to the *next* token after each candidate split boundary.
_NL_TAG_WEIGHTS = {
    "Punctuation": 10.0,
    "Conjunction": 8.0,
    "Preposition": 5.0,
    "Verb": 3.0,
    "Noun": 1.0,
}
_HEX_COLOR_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


# ---------------------------------------------------------------------------
# Text processing functions
# ---------------------------------------------------------------------------

def _is_ascii_word(text: str) -> bool:
    """Check if text is a pure ASCII word (letters and digits only)."""
    return bool(_ASCII_WORD_RE.match(text))


def _contains_ascii(text: str) -> bool:
    """Check if text contains any ASCII alphanumeric characters."""
    return bool(_ASCII_CHAR_RE.search(text))


def _needs_space(prev_text: str, cur_text: str) -> bool:
    """
    Check if space is needed between two words.

    Rules:
    1. If current word contains ASCII characters, always add space
    2. No space between pure CJK/punctuation characters

    Examples:
        "That's" + "what's" => "That's what's" (both contain ASCII)
        "hello" + "world" => "hello world" (both contain ASCII)
        "你好" + "world" => "你好 world" (current contains ASCII)
        "world" + "你好" => "world你好" (current is pure CJK)
        "你好" + "世界" => "你好世界" (both are pure CJK)
    """
    if not prev_text:
        return False
    # If current text contains any ASCII alphanumeric, add space
    # This handles: word+word, CJK+word, word'+word', etc.
    if _contains_ascii(cur_text):
        return True
    # No space between pure CJK characters or punctuation
    return False


def _normalize_filler_token(token: str) -> str:
    """Normalize a token for filler-word matching."""
    return token.strip().lower().strip(".,!?;:，。！？；：、")


def filter_filler_words(items: Iterable, enabled: bool = True) -> List:
    """Filter common disfluency/filler words from timestamp items."""
    if not enabled:
        return list(items)
    filtered: List = []
    for item in items:
        token = str(getattr(item, "text", "")).strip()
        if not token:
            continue
        if _normalize_filler_token(token) in _FILLER_WORDS:
            continue
        filtered.append(item)
    return filtered

def filter_text(text: str, filter_fillers: bool = True) -> str:
    """Filter filler words from text string."""
    if not filter_fillers:
        return text
    for filler in _FILLER_WORDS:
        text = text.replace(filler, "")
    return text


def normalize_hex_color(value: str) -> str:
    """Normalize a color value to ``#RRGGBB``."""
    if not isinstance(value, str) or not _HEX_COLOR_RE.match(value.strip()):
        raise ValueError(f"Invalid color '{value}'. Expected #RRGGBB.")
    normalized = value.strip().upper()
    return normalized if normalized.startswith("#") else f"#{normalized}"


def hex_color_to_ass(value: str) -> str:
    """Convert ``#RRGGBB`` to ASS primary colour format ``&H00BBGGRR``."""
    normalized = normalize_hex_color(value)
    rr = normalized[1:3]
    gg = normalized[3:5]
    bb = normalized[5:7]
    return f"&H00{bb}{gg}{rr}"


def hex_color_to_fcpxml(value: str) -> str:
    """Convert ``#RRGGBB`` to FCPXML RGBA float format."""
    normalized = normalize_hex_color(value)
    channels = [int(normalized[idx:idx + 2], 16) / 255.0 for idx in (1, 3, 5)]

    def _format_channel(channel: float) -> str:
        return f"{channel:.4f}".rstrip("0").rstrip(".")

    return " ".join(_format_channel(channel) for channel in channels) + " 1"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """Transcription segment with timing."""
    start: float
    end: float
    text: str
    words: List[Dict] = field(default_factory=list)

def _attach_punctuation_to_words(items: Iterable, text: str = "") -> List[dict]:
    """
    Build word dicts from timestamp items, attaching punctuation from original text.

    Forced-alignment items don't contain punctuation tokens, so punctuation
    is extracted from the original transcript text by locating each word in
    the text (in order) and reading any punctuation characters that follow it.

    Args:
        items: Iterable of objects with .text, .start_time, .end_time attributes
        text:  The original transcript string (source of punctuation).

    Returns:
        List of dicts with keys: word, start, end, punctuation
    """
    words: List[dict] = []
    pos = 0
    text_lower = text.lower() if text else ""
    text_len = len(text_lower)

    for it in items:
        token = str(getattr(it, "text", "")).strip()
        if not token:
            continue

        punct = ""
        if text:
            idx = text_lower.find(token.lower(), pos)
            if idx != -1:
                search_pos = idx + len(token)
                while search_pos < text_len and text[search_pos] == " ":
                    search_pos += 1
                while search_pos < text_len and text[search_pos] in _PUNCT_ALL:
                    punct += text[search_pos]
                    search_pos += 1
                pos = idx + len(token)

        words.append({
            "word": token,
            "start": float(getattr(it, "start_time", 0.0)),
            "end": float(getattr(it, "end_time", 0.0)),
            "punctuation": punct,
        })
    return words


def _nl_tagger_best_split_idx(words: List[dict]) -> Optional[int]:
    """
    Use macOS NLTagger (lexicalClass) to find the best word-level split index.

    Scores each candidate boundary i (split after word i) by:
        score = tagWeight(words[i+1]) / (1 + |charEnd(i) - center| × 0.5)

    Returns index i such that the best split is words[:i+1] | words[i+1:],
    or None if NLTagger is unavailable or < 2 words.
    """
    if len(words) < 2:
        return None
    try:
        from NaturalLanguage import NLTagger, NLTokenUnitWord  # macOS only
    except ImportError:
        return None

    # Build text without punctuation (already handled upstream)
    text_parts: List[str] = []
    prev = ""
    for w in words:
        tok = w["word"]
        if _needs_space(prev, tok):
            text_parts.append(" ")
        text_parts.append(tok)
        prev = tok
    text = "".join(text_parts)

    # Cumulative char offsets at the end of each word (for mapping back to word index)
    cum: List[int] = []
    total = 0
    prev = ""
    for w in words:
        tok = w["word"]
        if _needs_space(prev, tok):
            total += 1
        total += len(tok)
        cum.append(total)
        prev = tok

    tagger = NLTagger.alloc().initWithTagSchemes_(["LexicalClass"])
    tagger.setString_(text)

    token_uppers: List[int] = []
    token_weights: List[float] = []

    def _cb(tag, token_range, stop):
        tag_str = str(tag) if tag else ""
        if tag_str == "Whitespace":
            return
        weight = _NL_TAG_WEIGHTS.get(tag_str, 1.0)
        token_uppers.append(token_range.location + token_range.length)
        token_weights.append(weight)

    tagger.enumerateTagsInRange_unit_scheme_options_usingBlock_(
        (0, len(text)), NLTokenUnitWord, "LexicalClass", 0, _cb,
    )

    if len(token_uppers) < 2:
        return None

    center = len(text) / 2.0
    best_char_upper: Optional[int] = None
    max_score = -1.0
    for i in range(len(token_uppers) - 1):
        distance = abs(token_uppers[i] - center)
        score = token_weights[i + 1] / (1.0 + distance * 0.5)
        if score > max_score:
            max_score = score
            best_char_upper = token_uppers[i]

    if best_char_upper is None:
        return None

    # Map char offset → last word whose cumulative end ≤ best_char_upper
    best_word_idx: Optional[int] = None
    for i, c in enumerate(cum):
        if c <= best_char_upper:
            best_word_idx = i

    if best_word_idx is None or best_word_idx >= len(words) - 1:
        return None

    return best_word_idx


def _split_bucket_by_nltokenizer(
    words: List[dict],
    seg_start: float,
    seg_end: float,
    max_chars: int,
) -> List["Segment"]:
    """
    Recursively split *words* into Segments ≤ *max_chars* using macOS NLTagger
    POS-weight scoring (mirrors Swift SegmentSplitByNatural).
    Emits as a single segment if NLTagger is unavailable or no split is found.
    """
    def _emit(ws: List[dict]) -> "Segment":
        t = ""
        lt = ""
        for w in ws:
            cur = w["word"]
            if _needs_space(lt, cur):
                t += " "
            t += cur
            lt = cur
        s = max(ws[0]["start"], seg_start)
        e = min(ws[-1]["end"], seg_end)
        if e < s:
            e = s
        cl: List[dict] = []
        for w in ws:
            cs = max(min(w["start"], e), s)
            ce = max(min(w["end"], e), s)
            ce = max(ce, cs)
            cl.append({**w, "start": cs, "end": ce})
        return Segment(start=s, end=e, text=t, words=cl)

    def _build_text(ws: List[dict]) -> str:
        t, lt = "", ""
        for w in ws:
            cur = w["word"]
            if _needs_space(lt, cur):
                t += " "
            t += cur
            lt = cur
        return t

    def _fallback_split_idx(ws: List[dict]) -> Optional[int]:
        if len(ws) < 2:
            return None
        rendered = []
        lengths = []
        last = ""
        total = 0
        for w in ws:
            cur = w["word"]
            if _needs_space(last, cur):
                rendered.append(" ")
                total += 1
            rendered.append(cur)
            total += len(cur)
            lengths.append(total)
            last = cur
        target = max_chars if max_chars > 0 else max(1, total // 2)
        best_idx = min(range(len(lengths) - 1), key=lambda idx: abs(lengths[idx] - target))
        return best_idx

    def _split_recursive(ws: List[dict]) -> List["Segment"]:
        if len(ws) < 2 or len(_build_text(ws)) <= max_chars:
            return [_emit(ws)]
        idx = _nl_tagger_best_split_idx(ws)
        # print(f"_split_recursive: text='{_build_text(ws)}' idx={idx}")
        if idx is None:
            idx = _fallback_split_idx(ws)
        if idx is None:
            return [_emit(ws)]
        first, second = ws[: idx + 1], ws[idx + 1 :]
        if not first or not second:
            return [_emit(ws)]
        return _split_recursive(first) + _split_recursive(second)

    return _split_recursive(words)


def _split_vad_segment_by_punctuation(
    words: List[dict],
    seg_start: float,
    seg_end: float,
    max_chars: int = 0,
) -> List["Segment"]:
    """
    Split a VAD segment's word list into Segments on punctuation boundaries.

    Two-pass strategy when *max_chars* > 0:
    1. Primary split on punctuation marks (fast, language-agnostic).
    2. Any segment whose text still exceeds *max_chars* is further subdivided
       by ``_split_bucket_by_nltokenizer`` (macOS NLTokenizer sentences, with
       POS-weight word scoring as fallback).

    Args:
        words: Output of _attach_punctuation_to_words (dicts with word/start/end/punctuation).
        seg_start: Absolute start time of the VAD segment (lower bound for timing).
        seg_end: Absolute end time of the VAD segment (upper bound for timing).
        max_chars: If > 0, further split any post-punctuation segment exceeding this limit.

    Returns:
        List of Segment objects.
    """
    if not words:
        return []

    # Phase 1: collect raw word-buckets split on punctuation marks.
    buckets: List[List[dict]] = []
    bucket: List[dict] = []
    for w in words:
        bucket.append(w)
        if w["punctuation"]:
            buckets.append(bucket)
            bucket = []
    if bucket:
        buckets.append(bucket)

    # print(f"buckets: {[ [w['word'] for w in b] for b in buckets ]}")

    result: List["Segment"] = []

    def _emit_bucket(b: List[dict]) -> None:
        text = ""
        last_word = ""
        for w in b:
            cur = w["word"]
            if _needs_space(last_word, cur):
                text += " "
            text += cur
            last_word = cur
        start = max(b[0]["start"], seg_start)
        end = min(b[-1]["end"], seg_end)
        if end < start:
            end = start
        clamped_words = []
        for w in b:
            cs = max(min(w["start"], end), start)
            ce = max(min(w["end"], end), start)
            ce = max(ce, cs)  # ensure end >= start
            clamped_words.append({**w, "start": cs, "end": ce})
        result.append(Segment(start=start, end=end, text=text, words=clamped_words))

    # Phase 2: emit each bucket independently.  If max_chars is set, delegate
    # to the NL tokenizer splitter which emits a single segment when the bucket
    # fits, or splits it naturally when it exceeds the limit.
    for b in buckets:
        if max_chars > 0:
            result.extend(
                _split_bucket_by_nltokenizer(b, seg_start, seg_end, max_chars)
            )
        else:
            _emit_bucket(b)

    return result

# ---------------------------------------------------------------------------
# SRT formatting
# ---------------------------------------------------------------------------

def _format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(
    segments: List[Tuple[float, float, str]],
    margin_left: float = 0.0,
    margin_right: float = 0.0,
) -> str:
    lines: List[str] = []
    prev_end: float = 0.0
    for i, (st, ed, text) in enumerate(segments, start=1):
        st = st + margin_left
        ed = ed + margin_right
        if st < prev_end:
            st = prev_end
        if ed < st:
            ed = st
        prev_end = ed
        lines.append(str(i))
        lines.append(f"{_format_srt_time(st)} --> {_format_srt_time(ed)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video as WAV 16kHz mono."""
    print(f"🎵 Extracting audio from {video_path}...")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✅ Audio extracted to {output_path}")
    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
