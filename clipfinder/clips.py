"""The Clip record, snapping/cleaning the clip list, and the clips.json manifest."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .transcript import Segment


@dataclass
class Clip:
    start: float
    end: float
    score: float
    title: str
    hook: str
    category: str
    reason: str
    # Filled in after selection; used by the feedback layer (clipfinder/feedback.py).
    id: str = ""                                   # stable id, e.g. "stream2@612.4"
    text: str = ""                                 # what's said in the clip
    features: dict = field(default_factory=dict)   # measurements the re-ranker learns from
    predicted: float | None = None                 # re-ranker score, when one is in use

    REQUIRED = ("start", "end", "score", "title", "hook", "category", "reason")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        kw = {k: d[k] for k in cls.REQUIRED}
        kw.update({k: d[k] for k in ("id", "text", "features", "predicted") if k in d})
        return cls(**kw)


SNAP_OUT = 3.0  # an edge may move outward this far (more setup / more payoff)
SNAP_IN = 1.0   # ...but inward only this far: Claude starts clips early on purpose


def snap_to_segments(clip: Clip, segs: list[Segment],
                     out_tol: float = SNAP_OUT, in_tol: float = SNAP_IN) -> Clip:
    """Move start/end to real sentence boundaries so clips don't cut mid-word.

    Edges move outward up to `out_tol` seconds but inward only `in_tol`, so the
    build-up Claude deliberately left before the action, or a silent explosion
    before the next line of speech, is never trimmed away. The start can never
    pass the end. If no boundary is close enough, that edge stays put.
    """
    starts = [s.start for s in segs
              if clip.start - out_tol <= s.start <= clip.start + in_tol and s.start < clip.end]
    if starts:
        clip.start = min(starts, key=lambda x: abs(x - clip.start))
    ends = [s.end for s in segs
            if clip.end - in_tol <= s.end <= clip.end + out_tol and s.end > clip.start]
    if ends:
        clip.end = min(ends, key=lambda x: abs(x - clip.end))
    return clip


def clean_clips(clips: list[Clip], segs: list[Segment], min_len: float,
                max_len: float, min_score: float, top: int,
                video_len: float | None, pad: float = 0.0) -> list[Clip]:
    """Snap, enforce length, drop overlaps (keeping the higher score) and keep the top N.

    `pad` is the total padding added at cut time (before + after). Length limits
    apply to the final padded clip, so the content is held to
    [min_len - pad, max_len - pad].
    """
    lo = max(min_len - pad, 1.0)
    hi = max(max_len - pad, lo)
    out = []
    for c in clips:
        if c.end <= c.start:
            continue
        c = snap_to_segments(c, segs)
        if c.duration > hi:
            c.end = c.start + hi
        if c.duration < lo:
            c.end = c.start + lo
        if video_len and c.end > video_len:
            # Near the end of the video: keep the length by starting earlier instead.
            c.start = max(c.start - (c.end - video_len), 0.0)
            c.end = video_len
        if c.score >= min_score and c.duration > 0:
            out.append(c)

    out.sort(key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for c in out:
        if all(c.end <= k.start or c.start >= k.end for k in kept):
            kept.append(c)
    kept = kept[:top] if top else kept
    return sorted(kept, key=lambda c: c.start)


# --------------------------------------------------------------------------- #
# clips.json: the "recipe". Small (KBs), keep it forever; re-render on demand.
# --------------------------------------------------------------------------- #

def save_manifest(path: Path, clips: list[Clip], **meta) -> None:
    data = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()},
            "clips": [asdict(c) for c in clips]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_manifest(path: Path, **fields) -> None:
    """Set top-level fields (e.g. segments) in clips.json, keeping everything else."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(fields)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> tuple[dict, list[Clip]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, [Clip.from_dict(c) for c in data["clips"]]
