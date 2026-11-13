"""Non-speech signals for finding clips: audio loudness spikes and chat bursts.

The transcript can't see a silent chase or a clutch play, but the game audio
and the chat usually react. Both become short timestamped notes that go into
Claude's prompt next to the transcript.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .transcript import Segment, ts_to_seconds


@dataclass
class Event:
    start: float
    end: float
    kind: str      # "audio", "chat" or "sound" (see sounds.py)
    strength: float  # audio: dB above normal; chat: x normal rate; sound: 0-1 likelihood
    detail: str = ""


# --------------------------------------------------------------------------- #
# Audio loudness (ffmpeg measures RMS per second; no Python audio libraries)
# --------------------------------------------------------------------------- #

def loudness_cmd(video: Path) -> list[str]:
    # Resample inside the filter chain (-ar/-ac would apply after it), so each
    # astats window is exactly one second of audio.
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn",
            "-af", "aformat=channel_layouts=mono,aresample=8000,asetnsamples=n=8000:p=0,"
                   "astats=metadata=1:reset=1,"
                   "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-"]


def parse_loudness(text: str, floor: float = -90.0) -> list[float]:
    """ametadata output -> one dB value per second (index = second of the video).

    Values are placed by their pts_time, so the result stays aligned with the
    video even if ffmpeg's window size differs; several values in one second
    are averaged in the power domain."""
    sums: dict[int, list[float]] = {}
    t = 0.0
    for line in text.splitlines():
        if "pts_time:" in line:
            try:
                t = float(line.rsplit("pts_time:", 1)[1].split()[0])
            except ValueError:
                pass
        elif "RMS_level=" in line:
            try:
                db = float(line.split("=", 1)[1].strip())
            except ValueError:
                db = floor
            db = floor if math.isinf(db) or db < floor else db
            sums.setdefault(int(t), []).append(10 ** (db / 10))
    if not sums:
        return []
    return [10 * math.log10(sum(v) / len(v)) if (v := sums.get(i)) else floor
            for i in range(max(sums) + 1)]


def measure_loudness(video: Path, cache: Path | None = None) -> list[float]:
    if cache and cache.exists():
        return json.loads(cache.read_text())
    res = subprocess.run(loudness_cmd(video), capture_output=True, text=True, check=True)
    levels = parse_loudness(res.stdout)
    if cache:
        cache.write_text(json.dumps([round(v, 1) for v in levels]))
    return levels


def rolling_median(values: list[float], radius: int) -> list[float]:
    return [statistics.median(values[max(i - radius, 0):i + radius + 1])
            for i in range(len(values))]


def merge_runs(hits: list[tuple[float, float]], step: float, gap: float) -> list[tuple[float, float, float]]:
    """[(t, strength)] -> [(start, end, peak)] joining hits closer than `gap`."""
    runs: list[list[float]] = []
    for t, v in hits:
        if runs and t - runs[-1][1] <= gap:
            runs[-1][1] = t + step
            runs[-1][2] = max(runs[-1][2], v)
        else:
            runs.append([t, t + step, v])
    return [tuple(r) for r in runs]


def audio_events(levels: list[float], threshold_db: float = 8.0, min_level: float = -35.0,
                 radius: int = 60) -> list[Event]:
    """Seconds that are much louder than the surrounding two minutes."""
    if not levels:
        return []
    base = rolling_median(levels, radius)
    hits = [(float(t), lv - b) for t, (lv, b) in enumerate(zip(levels, base))
            if lv - b >= threshold_db and lv >= min_level]
    return [Event(a, b, "audio", round(peak, 1)) for a, b, peak in merge_runs(hits, 1.0, 2.0)]


# --------------------------------------------------------------------------- #
# Chat (Twitch chat replays from common downloaders)
# --------------------------------------------------------------------------- #

_CHAT_LINE = re.compile(r"^\s*\[?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*\]?\s*([^:]{1,40}):\s*(.*)$")


def load_chat(path: Path) -> list[tuple[float, str]]:
    """(seconds, message) from TwitchDownloader JSON, chat-downloader JSON, or
    text lines like '[0:12:34] user: message'."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "comments" in data:  # TwitchDownloaderCLI
            return sorted((float(c["content_offset_seconds"]),
                           str((c.get("message") or {}).get("body", "")))
                          for c in data["comments"])
        items = data if isinstance(data, list) else data.get("messages", [])
        return sorted((float(m["time_in_seconds"]), str(m.get("message", "")))  # chat-downloader
                      for m in items if "time_in_seconds" in m)
    out = []
    for line in text.splitlines():
        m = _CHAT_LINE.match(line)
        if m:
            out.append((ts_to_seconds(m.group(1)), m.group(3)))
    return sorted(out)


def chat_events(messages: list[tuple[float, str]], offset: float = 0.0, window: float = 10.0,
                ratio: float = 2.5, min_count: int = 5, radius: int = 30) -> list[Event]:
    """Windows where chat is much busier than the surrounding ~10 minutes."""
    if not messages:
        return []
    last = max(t for t, _ in messages) + offset
    n = int(last // window) + 1
    counts = [0] * n
    texts: list[list[str]] = [[] for _ in range(n)]
    for t, msg in messages:
        i = int((t + offset) // window)
        if 0 <= i < n:
            counts[i] += 1
            texts[i].append(msg)
    base = rolling_median([float(c) for c in counts], radius)
    hits = []
    for i, (c, b) in enumerate(zip(counts, base)):
        if c >= min_count and c >= ratio * max(b, 1.0):
            hits.append((i * window, c / max(b, 1.0)))
    events = []
    for a, b, peak in merge_runs(hits, window, window):
        words = Counter()
        for i in range(int(a // window), int(b // window)):
            # count each token once per message so one spammer doesn't dominate
            for msg in texts[i]:
                words.update(set(re.findall(r"[\w']+|[^\w\s]{2,}", msg)))
        top = [f'"{w}" x{k}' for w, k in words.most_common(3) if k >= 2]
        events.append(Event(a, b, "chat", round(peak, 1), ", ".join(top)))
    return events


# --------------------------------------------------------------------------- #
# Notes for the prompt
# --------------------------------------------------------------------------- #

def has_speech(segs: list[Segment], a: float, b: float) -> bool:
    return any(s.end > a and s.start < b for s in segs)


def describe(e: Event, segs: list[Segment]) -> str:
    rng = f"[{e.start:.1f}-{e.end:.1f}]"
    if e.kind == "audio":
        quiet = "" if has_speech(segs, e.start, e.end) else " (no speech)"
        return f"{rng} audio spike, {e.strength:+.0f} dB above normal{quiet}"
    if e.kind == "sound":
        quiet = "" if has_speech(segs, e.start, e.end) else " (no speech)"
        return f"{rng} sounds like {e.detail} ({e.strength:.0%} likely){quiet}"
    if e.kind == "music":
        return f"{rng} music playing (likely the in-game radio)"
    detail = f": {e.detail}" if e.detail else ""
    return f"{rng} chat burst, {e.strength:.1f}x normal rate{detail}"


class Signals:
    """All events for a video; hands out prompt notes for any time range."""

    def __init__(self, events: list[Event], segs: list[Segment], per_range: int = 20):
        self.events = sorted(events, key=lambda e: e.start)
        self.segs = segs
        self.per_range = per_range

    def __len__(self) -> int:
        return len(self.events)

    def notes(self, t0: float, t1: float) -> str:
        inside = [e for e in self.events if e.end > t0 and e.start < t1]
        if not inside:
            return ""
        # Too many? Take the strongest of each kind in turn (strengths aren't comparable
        # across kinds: dB vs chat ratio vs likelihood), then show them in time order.
        by_kind: dict[str, list[Event]] = {}
        for e in sorted(inside, key=lambda e: e.strength, reverse=True):
            by_kind.setdefault(e.kind, []).append(e)
        keep, queues = [], list(by_kind.values())
        while len(keep) < self.per_range and any(queues):
            for q in queues:
                if q and len(keep) < self.per_range:
                    keep.append(q.pop(0))
        keep.sort(key=lambda e: e.start)
        return "\n".join(describe(e, self.segs) for e in keep)


SIGNALS_HEADER = """Other signals for this section. The transcript only has speech, so these catch \
moments it misses: an audio spike is a sudden loud moment in the game or the room, \
"sounds like" is what a sound model heard (laughter, screaming, gunfire, explosions, sirens, \
crashes, tyre squeals), a chat burst is viewers reacting, "music playing" is usually the in-game \
radio: its songs are copyrighted, and clips with them get muted or claimed, so when moments are \
otherwise similar prefer the one without music, and don't run a clip into music it doesn't need. Sound labels are probabilities \
from game audio mixed with voice, so treat them as hints. A strong spike or burst with little speech can still be a great \
clip; start it a few seconds before the signal so the build-up is included. Chat reacts \
5-15 seconds after the moment."""
