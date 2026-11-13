"""How posted clips did, for the web UI: the same library.json and ranker.json as
`python feedback.py ...`, so the UI and the command line always agree.

Recording views and watch time is what lets the clipper learn: from 5 clips with stats,
the best and worst become examples in Claude's prompt; from 30, a re-ranker can be
trained, and it's only switched on if it beats Claude's own score on clips it hasn't seen.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

from . import feedback as fb
from .clips import Clip, load_manifest

PLATFORM = re.compile(r"^[a-z][a-z0-9_ -]{0,30}$")


def library_path(root: Path) -> Path:
    return root / fb.DEFAULT_LIBRARY


def ranker_path(root: Path) -> Path:
    return root / fb.DEFAULT_RANKER


def _count(v, name: str, cast=int):
    if v in (None, ""):
        return None
    try:
        n = cast(str(v).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"{name} must be a number.")
    if n < 0:
        raise ValueError(f"{name} can't be negative.")
    return n


def entry_for(lib: dict, clip: Clip, video: str | None) -> dict | None:
    """This clip's library entry, if it has one."""
    if not clip.id:
        return None
    key = fb.library_key(lib, clip, video)
    return lib.get(key)


def clip_stats(lib: dict, clip_dict: dict, video: str | None) -> dict | None:
    """What the gallery shows on a clip card: totals plus each platform's numbers."""
    try:
        clip = Clip.from_dict(clip_dict)
    except (KeyError, TypeError):
        return None
    e = entry_for(lib, clip, video)
    if not e or not e.get("posts"):
        return None
    t = fb.totals(e)
    return {"views": t["views"], "retention": t["retention"], "posts": e["posts"]}


def record(root: Path, recipe: Path, n: int, form: dict) -> dict:
    """Save one clip's numbers on one platform (re-saving updates them)."""
    meta, clips = load_manifest(recipe)
    if not isinstance(n, int) or not 0 < n <= len(clips):
        raise ValueError("Clip not found.")
    clip = clips[n - 1]
    if not clip.id:
        raise ValueError("This clip has no id (it's from an old run). Find clips again to log results.")
    platform = str(form.get("platform") or "").strip().lower()
    if not PLATFORM.match(platform):
        raise ValueError("Pick a platform.")
    stats = {"views": _count(form.get("views"), "Views"),
             "avg_watch": _count(form.get("avg_watch"), "Average watch time", float),
             "likes": _count(form.get("likes"), "Likes"),
             "comments": _count(form.get("comments"), "Comments"),
             "shares": _count(form.get("shares"), "Shares")}
    if stats["views"] is None:
        raise ValueError("Views are needed (the rest is optional).")
    url = str(form.get("url") or "").strip()[:500]
    path = library_path(root)
    lib = fb.load_library(path)
    entry = fb.record_post(lib, clip, streamer=meta.get("streamer"), video=meta.get("video"),
                           platform=platform, url=url, **stats)
    fb.save_library(path, lib)
    t = fb.totals(entry)
    return {"views": t["views"], "retention": t["retention"], "posts": entry["posts"],
            "with_stats": len(fb.with_stats(lib))}


def summary(root: Path, limit: int = 200) -> dict:
    lib = fb.load_library(library_path(root))
    entries = fb.with_stats(lib)
    perf = fb.relative_performance(entries) if entries else {}
    rows = []
    for e in sorted(entries, key=lambda e: perf.get(e["id"], 0), reverse=True)[:limit]:
        t = fb.totals(e)
        rows.append({"id": e["id"], "title": e.get("title", ""), "streamer": e.get("streamer", ""),
                     "views": t["views"], "retention": t["retention"],
                     "claude_score": e.get("claude_score"), "vs_typical": round(perf[e["id"]], 2),
                     "platforms": sorted(e.get("posts", {}))})
    ranker = None
    rp = ranker_path(root)
    if rp.is_file():
        try:
            m = json.loads(rp.read_text(encoding="utf-8"))
            ranker = {k: m.get(k) for k in ("trained_on", "enabled", "cv_spearman",
                                             "baseline_spearman", "trained")}
            ranker["current"] = fb.load_ranker(rp) is not None  # stale if features changed since
        except (OSError, ValueError):
            pass
    return {"with_stats": len(entries), "entries": rows, "ranker": ranker,
            "min_examples": fb.MIN_EXAMPLES, "min_train": fb.MIN_TRAIN,
            "good_train": fb.GOOD_TRAIN}


def import_csv(root: Path, text: str, platform: str, recipes: list[Path]) -> dict:
    """Pasted CSV (columns: clip, platform, url, views, avg_watch, shares, likes, comments).
    `clip` is a clip id, or a number when only one clips.json is given."""
    from .feedback_cli import cmd_import
    if not text.strip():
        raise ValueError("Paste a CSV first.")
    platform = (platform or "tiktok").strip().lower()
    if not PLATFORM.match(platform):
        raise ValueError("Pick a platform.")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = Path(f.name)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            cmd_import(argparse.Namespace(csv=tmp, clips_json=recipes, platform=platform,
                                          library=library_path(root)))
    except (KeyError, ValueError) as e:
        raise ValueError(f"Couldn't read that CSV ({e}). It needs a header row with at least "
                         f"'clip' and 'views'.")
    finally:
        tmp.unlink(missing_ok=True)
    return {"log": out.getvalue().strip()}


def train(root: Path) -> dict:
    try:
        model = fb.train_ranker(fb.load_library(library_path(root)))
    except ValueError as e:
        raise ValueError(str(e))
    ranker_path(root).write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    top = sorted(zip(model["features"], model["weights"]), key=lambda kv: abs(kv[1]), reverse=True)
    return {"trained_on": model["trained_on"], "enabled": model["enabled"],
            "cv_spearman": model["cv_spearman"], "baseline_spearman": model["baseline_spearman"],
            "factors": [[k, round(v, 2)] for k, v in top[:5]]}
