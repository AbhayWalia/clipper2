"""The feedback layer: learn from how posted clips actually performed.

1. Library  - every posted clip with its stats (views, watch time, shares) in one
              JSON file, filled by `python feedback.py record/import`.
2. Examples - the channel's best and worst performers go into Claude's system
              prompt, so it learns what *this* audience likes. No training needed.
3. Ranker   - a small ridge-regression model on clip features (Claude's score,
              length, audio spikes, chat bursts, speech rate, category) predicting
              performance. It only switches on if it beats Claude's score alone in
              cross-validation, so it can't make picks worse.

Pure Python on purpose: a few hundred rows and ~15 features need no numpy.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from datetime import date
from pathlib import Path

from .clips import Clip
from .sounds import GROUPS as SOUND_GROUPS
from .transcript import Segment, segments_between

DEFAULT_LIBRARY = Path("library.json")
DEFAULT_RANKER = Path("ranker.json")
CATEGORIES = ["chaos", "funny", "reaction", "skill", "story", "other"]
CHAT_LAG = 15.0  # chat reacts after the moment, so look this far past the clip end

MIN_EXAMPLES = 5      # posted clips with stats before examples are used
MIN_TRAIN = 30        # ... before a ranker can be trained
GOOD_TRAIN = 200      # ... before a ranker is reasonably trustworthy


# --------------------------------------------------------------------------- #
# Features (computed when clips are picked, stored in clips.json)
# --------------------------------------------------------------------------- #

def clip_text(clip: Clip, segs: list[Segment], limit: int = 1200) -> str:
    text = " ".join(s.text for s in segments_between(segs, clip.start, clip.end))
    return text[:limit]


def clip_features(clip: Clip, segs: list[Segment], signals=None) -> dict:
    dur = max(clip.duration, 0.1)
    inside = segments_between(segs, clip.start, clip.end)
    words = sum(len(s.text.split()) for s in inside)
    covered = sum(min(s.end, clip.end) - max(s.start, clip.start) for s in inside)
    f = {
        "claude_score": float(clip.score),
        "duration": round(dur, 1),
        "hook_chars": len(clip.hook),
        "words_per_sec": round(words / dur, 2),
        "speech_fraction": round(min(max(covered / dur, 0.0), 1.0), 2),
        "audio_peak_db": 0.0, "audio_spikes": 0,
        "chat_peak_ratio": 0.0, "chat_bursts": 0,
        **{f"sound_{g}": 0.0 for g in SOUND_GROUPS},
    }
    if signals is not None:
        audio = [e for e in signals.events if e.kind == "audio"
                 and e.end > clip.start and e.start < clip.end]
        chat = [e for e in signals.events if e.kind == "chat"
                and e.end > clip.start and e.start < clip.end + CHAT_LAG]
        f["audio_peak_db"] = max((e.strength for e in audio), default=0.0)
        f["audio_spikes"] = len(audio)
        f["chat_peak_ratio"] = max((e.strength for e in chat), default=0.0)
        f["chat_bursts"] = len(chat)
        for g in SOUND_GROUPS:
            f[f"sound_{g}"] = max((e.strength for e in signals.events if e.kind == "sound"
                                   and e.detail == g and e.end > clip.start and e.start < clip.end),
                                  default=0.0)
    for c in CATEGORIES:
        f[f"cat_{c}"] = 1.0 if clip.category == c else 0.0
    return f


FEATURES = ["claude_score", "duration", "hook_chars", "words_per_sec", "speech_fraction",
            "audio_peak_db", "audio_spikes", "chat_peak_ratio", "chat_bursts"] + \
           [f"sound_{g}" for g in SOUND_GROUPS] + \
           [f"cat_{c}" for c in CATEGORIES]


# --------------------------------------------------------------------------- #
# Library of posted clips
# --------------------------------------------------------------------------- #

def load_library(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_library(path: Path, lib: dict) -> None:
    path.write_text(json.dumps(lib, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def record_post(lib: dict, clip: Clip, *, streamer: str | None, video: str | None,
                platform: str, url: str = "", views: int | None = None,
                avg_watch: float | None = None, shares: int | None = None,
                likes: int | None = None, comments: int | None = None,
                today: str | None = None) -> dict:
    """Add or update one clip's stats on one platform. Stats can be updated as
    they grow (re-run with the new numbers)."""
    if not clip.id:
        raise ValueError("This clip has no id; re-run the clip finder so clips.json gets ids.")
    entry = lib.setdefault(clip.id, {})
    entry.update({
        "id": clip.id, "streamer": streamer or "", "video": video or "",
        "title": clip.title, "hook": clip.hook, "category": clip.category,
        "duration": round(clip.duration, 1), "claude_score": clip.score,
        "text": clip.text, "features": clip.features or entry.get("features", {}),
    })
    post = entry.setdefault("posts", {}).setdefault(platform, {})
    for k, v in (("url", url), ("views", views), ("avg_watch", avg_watch), ("shares", shares),
                 ("likes", likes), ("comments", comments)):
        if v not in (None, ""):
            post[k] = v
    post["updated"] = today or date.today().isoformat()
    return entry


def totals(entry: dict) -> dict:
    """Combine platforms: summed views/shares, view-weighted watch time."""
    posts = entry.get("posts", {}).values()
    views = sum(int(p.get("views") or 0) for p in posts)
    watch = [(float(p["avg_watch"]), int(p.get("views") or 0)) for p in posts
             if p.get("avg_watch") not in (None, "")]
    w = sum(v for _, v in watch)
    avg_watch = (sum(a * v for a, v in watch) / w) if w else (watch[0][0] if watch else None)
    dur = entry.get("duration") or 0
    return {
        "views": views,
        "shares": sum(int(p.get("shares") or 0) for p in posts),
        "avg_watch": avg_watch,
        "retention": (avg_watch / dur) if (avg_watch and dur) else None,
    }


def with_stats(lib: dict) -> list[dict]:
    return [e for e in lib.values() if totals(e)["views"] > 0]


def relative_performance(entries: list[dict]) -> dict[str, float]:
    """log(views) relative to the streamer's typical clip (or everyone's, if the
    streamer has fewer than 3). This measures how good the *moment* was, not how
    big the channel is, which is what picking clips within a stream needs."""
    lv = {e["id"]: math.log1p(totals(e)["views"]) for e in entries}
    by_streamer: dict[str, list[float]] = {}
    for e in entries:
        by_streamer.setdefault(e.get("streamer", ""), []).append(lv[e["id"]])
    overall = statistics.median(lv.values()) if lv else 0.0
    out = {}
    for e in entries:
        vals = by_streamer[e.get("streamer", "")]
        base = statistics.median(vals) if len(vals) >= 3 else overall
        out[e["id"]] = lv[e["id"]] - base
    return out


# --------------------------------------------------------------------------- #
# Few-shot examples for the prompt
# --------------------------------------------------------------------------- #

def pick_examples(lib: dict, n_good: int = 8, n_bad: int = 3,
                  streamer: str | None = None) -> tuple[list[dict], list[dict]]:
    entries = with_stats(lib)
    if len(entries) < MIN_EXAMPLES:
        return [], []
    perf = relative_performance(entries)
    # Prefer the same streamer's clips, then the rest; stable order for caching.
    def key(e):
        return (0 if streamer and e.get("streamer") == streamer else 1, -perf[e["id"]], e["id"])
    ranked = sorted(entries, key=key)
    good = [e for e in ranked if perf[e["id"]] > 0][:n_good]
    bad_pool = sorted(entries, key=lambda e: (0 if streamer and e.get("streamer") == streamer else 1,
                                              perf[e["id"]], e["id"]))
    bad = [e for e in bad_pool if perf[e["id"]] < 0 and e not in good][:n_bad]
    return good, bad


def _example(e: dict) -> str:
    t = totals(e)
    ret = f' retention="{t["retention"]:.0%}"' if t["retention"] is not None else ""
    text = e.get("text") or "(no speech)"
    return (f'<example streamer="{e.get("streamer", "")}" views="{t["views"]}"{ret} '
            f'category="{e.get("category", "")}" length="{e.get("duration", 0):.0f}s">\n'
            f'title: {e.get("title", "")}\nhook: {e.get("hook", "")}\ntranscript: {text}\n</example>')


def examples_prompt(lib: dict, n_good: int = 8, n_bad: int = 3, streamer: str | None = None) -> str:
    good, bad = pick_examples(lib, n_good, n_bad, streamer)
    if not good:
        return ""
    parts = ["Results from clips this channel actually posted. Learn what this audience "
             "responds to and weigh it alongside the general guidance above.",
             "\nClips that performed well:"]
    parts += [_example(e) for e in good]
    if bad:
        parts.append("\nClips that underperformed (avoid moments like these):")
        parts += [_example(e) for e in bad]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Ranker: ridge regression, standardized features, k-fold cross-validation
# --------------------------------------------------------------------------- #

def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_ridge(X: list[list[float]], y: list[float], lam: float = 1.0) -> dict:
    d = len(X[0])
    means = [statistics.fmean(r[j] for r in X) for j in range(d)]
    stds = [statistics.pstdev([r[j] for r in X]) or 1.0 for j in range(d)]
    Z = [[(r[j] - means[j]) / stds[j] for j in range(d)] for r in X]
    ym = statistics.fmean(y)
    yc = [v - ym for v in y]
    ata = [[sum(z[i] * z[j] for z in Z) + (lam if i == j else 0.0) for j in range(d)] for i in range(d)]
    aty = [sum(z[i] * v for z, v in zip(Z, yc)) for i in range(d)]
    return {"means": means, "stds": stds, "weights": _solve(ata, aty), "intercept": ym}


def predict(model: dict, x: list[float]) -> float:
    return model["intercept"] + sum(
        w * (v - m) / s for w, v, m, s in zip(model["weights"], x, model["means"], model["stds"]))


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if len(set(ra)) < 2 or len(set(rb)) < 2:
        return 0.0
    return statistics.correlation(ra, rb)


def feature_row(features: dict) -> list[float]:
    return [float(features.get(k, 0.0) or 0.0) for k in FEATURES]


def train_ranker(lib: dict, lam: float = 1.0, folds: int = 5, seed: int = 0) -> dict:
    entries = [e for e in with_stats(lib) if e.get("features")]
    if len(entries) < MIN_TRAIN:
        raise ValueError(f"Need at least {MIN_TRAIN} posted clips with views and features to "
                         f"train; have {len(entries)}. Keep recording stats.")
    perf = relative_performance(entries)
    X = [feature_row(e["features"]) for e in entries]
    y = [perf[e["id"]] for e in entries]

    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    oof = [0.0] * len(entries)
    for f in range(folds):
        test = set(idx[f::folds])
        tr = [i for i in idx if i not in test]
        m = fit_ridge([X[i] for i in tr], [y[i] for i in tr], lam)
        for i in test:
            oof[i] = predict(m, X[i])
    model_rho = spearman(oof, y)
    base_rho = spearman([e["features"].get("claude_score", e.get("claude_score", 0)) for e in entries], y)

    model = fit_ridge(X, y, lam)
    model.update({
        "features": FEATURES, "trained_on": len(entries), "lambda": lam,
        "cv_spearman": round(model_rho, 3), "baseline_spearman": round(base_rho, 3),
        "enabled": model_rho > base_rho + 0.02,
        "trained": date.today().isoformat(),
    })
    return model


def load_ranker(path: Path) -> dict | None:
    if not path.exists():
        return None
    model = json.loads(path.read_text(encoding="utf-8"))
    return model if model.get("features") == FEATURES else None  # stale if features changed


def rank_clips(clips: list[Clip], model: dict, top: int) -> list[Clip]:
    for c in clips:
        c.predicted = round(predict(model, feature_row(c.features)), 3)
    best = sorted(clips, key=lambda c: c.predicted, reverse=True)
    best = best[:top] if top else best
    return sorted(best, key=lambda c: c.start)
