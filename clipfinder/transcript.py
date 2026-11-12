"""Loading transcripts (.srt / .vtt / Whisper .json / timestamped .txt) and chunking them."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)  # empty unless Whisper gave word timings

    def timed_words(self) -> list[Word]:
        """Word timings: Whisper's own if present, otherwise spread across the
        segment in proportion to word length (good enough for captions)."""
        if self.words:
            return self.words
        tokens = self.text.split()
        if not tokens:
            return []
        weights = [len(t) + 1 for t in tokens]
        total, t, out = sum(weights), self.start, []
        span = max(self.end - self.start, 0.01)
        for tok, w in zip(tokens, weights):
            d = span * w / total
            out.append(Word(t, t + d, tok))
            t += d
        return out


def ts_to_seconds(ts: str) -> float:
    """'01:02:03,456' / '01:02:03.456' / '02:03' / '123.4' -> seconds."""
    ts = ts.strip().replace(",", ".")
    if ":" not in ts:
        return float(ts)
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


_CUE_RE = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})"
)


_INLINE_TS = re.compile(r"<(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})>")


def _strip_tags(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).strip()


def _inline_words(raw: str, start: float, end: float) -> list[Word]:
    """YouTube auto-captions time each word inline: 'so<00:00:01.120><c> what</c>'."""
    parts = _INLINE_TS.split(raw)              # [text0, ts1, text1, ts2, text2, ...]
    times = [start] + [ts_to_seconds(t) for t in parts[1::2]]
    chunks = [_strip_tags(t) for t in parts[0::2]]
    words = []
    for i, (t0, chunk) in enumerate(zip(times, chunks)):
        toks = chunk.split()
        if not toks:
            continue
        t1 = times[i + 1] if i + 1 < len(times) else max(end, t0)
        step = (t1 - t0) / len(toks)
        words += [Word(t0 + k * step, t0 + (k + 1) * step, w) for k, w in enumerate(toks)]
    return words


def load_srt_vtt(text: str) -> list[Segment]:
    text = text.replace("\r\n", "\n")
    # YouTube's automatic captions roll: each cue repeats the previous line and
    # adds the new words with inline timestamps. Detect that and keep only the
    # new text, with the per-word times (used for the animated captions).
    rolling = bool(_INLINE_TS.search(text))
    # Cues end at a truly empty line or the next timing line. (YouTube puts
    # lines holding a single space *inside* cues, so "blank-looking" can't end one.)
    cues: list[tuple[float, float, list[str]]] = []
    for line in text.split("\n"):
        m = _CUE_RE.search(line)
        if m:
            cues.append((ts_to_seconds(m.group(1)), ts_to_seconds(m.group(2)), []))
        elif line.strip() == "":
            # An empty line ends a cue, but only once it has text: YouTube files can
            # have an empty (or single-space) line between the timing and the words.
            if cues and any(cues[-1][2]) and cues[-1][2][-1] is not None:
                cues[-1] = (cues[-1][0], cues[-1][1], cues[-1][2] + [None])  # closed
        elif cues and (not cues[-1][2] or cues[-1][2][-1] is not None):
            cues[-1][2].append(line.strip())

    segs: list[Segment] = []
    prev: set[str] = set()
    for start, end, lines in cues:
        raw = [l for l in lines if l]
        clean = [_strip_tags(l) for l in raw]
        if rolling:
            keep = [(r, c) for r, c in zip(raw, clean) if c and c not in prev]
            prev = {c for c in clean if c}
        else:
            keep = [(r, c) for r, c in zip(raw, clean) if c]
        body = " ".join(c for _, c in keep)
        if body:
            words = []
            if rolling:
                for r, _ in keep:
                    words += _inline_words(r, start, end)
            segs.append(Segment(start, end, body, words))
    return segs


def _words(items) -> list[Word]:
    return [Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
            for w in items or [] if str(w.get("word", "")).strip()]


def load_json(text: str) -> list[Segment]:
    data = json.loads(text)
    items = data.get("segments", data) if isinstance(data, dict) else data
    return [Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip(),
                    _words(s.get("words")))
            for s in items if str(s.get("text", "")).strip()]


_TXT_RE = re.compile(
    r"^\s*\[\s*([\d:.,]+)\s*(?:-|–|-->)?\s*([\d:.,]+)?\s*\]\s*(.+)$"
)


def load_txt(text: str) -> list[Segment]:
    raw = []
    for line in text.splitlines():
        m = _TXT_RE.match(line)
        if m:
            start = ts_to_seconds(m.group(1))
            end = ts_to_seconds(m.group(2)) if m.group(2) else None
            raw.append([start, end, m.group(3).strip()])
    if not raw:
        raise ValueError("No timestamped lines found. Use lines like '[00:01:23] text'.")
    for i, r in enumerate(raw):  # fill missing end times from the next line
        if r[1] is None:
            r[1] = raw[i + 1][0] if i + 1 < len(raw) else r[0] + 5
    return [Segment(*r) for r in raw]


def load_transcript(path: Path) -> list[Segment]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext in (".srt", ".vtt"):
        segs = load_srt_vtt(text)
    elif ext == ".json":
        segs = load_json(text)
    else:
        segs = load_txt(text)
    if not segs:
        raise ValueError(f"Couldn't read any lines from {path}")
    return sorted(segs, key=lambda s: s.start)


def chunk_segments(segs: list[Segment], window: float, overlap: float) -> list[list[Segment]]:
    """Split the transcript into overlapping time windows so long streams fit."""
    if window <= 0 or not 0 <= overlap < window:
        # Each step moves forward by window - overlap; zero or less never ends.
        raise ValueError(f"Need window > 0 and 0 <= overlap < window (got window={window}, "
                         f"overlap={overlap}).")
    chunks, t, last = [], segs[0].start, segs[-1].end
    while t < last:
        chunk = [s for s in segs if s.end > t and s.start < t + window]
        if chunk:
            chunks.append(chunk)
        t += window - overlap
    return chunks


def format_segments(segs: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segs)


def segments_between(segs: list[Segment], t0: float, t1: float) -> list[Segment]:
    return [s for s in segs if s.end > t0 and s.start < t1]
