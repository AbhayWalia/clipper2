"""`python feedback.py ...`: record how posted clips did, then learn from it.

  record   one clip's stats on one platform (re-run as the numbers grow)
  import   many at once from a CSV (e.g. pasted from TikTok / YouTube analytics)
  list     what's in the library, best first
  examples preview the examples Claude will see
  train    fit the re-ranker; it only switches on if it beats Claude's score alone
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import feedback as fb
from .clips import Clip, load_manifest


def _num(v, cast=float):
    if v in (None, ""):
        return None
    return cast(str(v).replace(",", "").replace("_", ""))


def find_clip(clips: list[Clip], which: str) -> Clip:
    if which.isdigit():
        i = int(which)
        if not 1 <= i <= len(clips):
            sys.exit(f"Clip {i} doesn't exist; this clips.json has {len(clips)} (numbered from 1).")
        return clips[i - 1]
    for c in clips:
        if c.id == which:
            return c
    sys.exit(f"No clip with id {which!r} in that clips.json.")


def cmd_record(a) -> None:
    meta, clips = load_manifest(a.clips_json)
    clip = find_clip(clips, a.clip)
    lib = fb.load_library(a.library)
    e = fb.record_post(lib, clip, streamer=meta.get("streamer"), video=meta.get("video"),
                       platform=a.platform, url=a.url, views=a.views, avg_watch=a.avg_watch,
                       shares=a.shares, likes=a.likes, comments=a.comments)
    fb.save_library(a.library, lib)
    t = fb.totals(e)
    print(f"Recorded {clip.id} on {a.platform}: {t['views']:,} views total. "
          f"Library: {len(fb.with_stats(lib))} clips with stats.")


def cmd_import(a) -> None:
    manifests = [load_manifest(p) for p in a.clips_json]
    by_id = {c.id: (c, meta) for meta, clips in manifests for c in clips if c.id}
    lib = fb.load_library(a.library)
    done = skipped = 0
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            which = row.get("clip") or row.get("clip_id") or row.get("id") or ""
            if which.isdigit():
                if len(manifests) != 1:
                    print(f"  line {n}: clip number needs exactly one --clips-json; skipped")
                    skipped += 1
                    continue
                meta, clips = manifests[0]
                clip = find_clip(clips, which)
            elif which in by_id:
                clip, meta = by_id[which]
            elif which in lib:  # already known: just update the numbers
                e = lib[which]
                clip = Clip(0, e.get("duration", 0), e.get("claude_score", 0), e.get("title", ""),
                            e.get("hook", ""), e.get("category", "other"), "", id=which,
                            text=e.get("text", ""), features=e.get("features", {}))
                meta = {"streamer": e.get("streamer"), "video": e.get("video")}
            else:
                print(f"  line {n}: unknown clip {which!r}; skipped")
                skipped += 1
                continue
            fb.record_post(lib, clip, streamer=meta.get("streamer"), video=meta.get("video"),
                           platform=row.get("platform") or a.platform, url=row.get("url", ""),
                           views=_num(row.get("views"), int),
                           avg_watch=_num(row.get("avg_watch") or row.get("average_watch_time")),
                           shares=_num(row.get("shares"), int), likes=_num(row.get("likes"), int),
                           comments=_num(row.get("comments"), int))
            done += 1
    fb.save_library(a.library, lib)
    print(f"Imported {done} row(s), skipped {skipped}. "
          f"Library: {len(fb.with_stats(lib))} clips with stats.")


def cmd_list(a) -> None:
    lib = fb.load_library(a.library)
    entries = fb.with_stats(lib)
    if not entries:
        print(f"No clips with stats in {a.library} yet. Add some with: python feedback.py record ...")
        return
    perf = fb.relative_performance(entries)
    entries.sort(key=lambda e: perf[e["id"]], reverse=True)
    print(f"{'vs typical':>10}  {'views':>9}  {'ret.':>5}  {'AI':>3}  streamer / title")
    for e in entries[:a.top or None]:
        t = fb.totals(e)
        ret = f"{t['retention']:.0%}" if t["retention"] is not None else "-"
        print(f"{perf[e['id']]:+10.2f}  {t['views']:>9,}  {ret:>5}  {e.get('claude_score', 0):>3.0f}  "
              f"{e.get('streamer') or '-'} / {e.get('title', '')}")
    print(f"\n{len(entries)} clips with stats. Examples switch on at {fb.MIN_EXAMPLES}, "
          f"training at {fb.MIN_TRAIN} (better with {fb.GOOD_TRAIN}+).")


def cmd_examples(a) -> None:
    text = fb.examples_prompt(fb.load_library(a.library), a.good, a.bad, a.streamer)
    print(text or f"Not enough clips with stats yet (need {fb.MIN_EXAMPLES}).")


def cmd_train(a) -> None:
    try:
        model = fb.train_ranker(fb.load_library(a.library), lam=a.lam)
    except ValueError as e:
        sys.exit(str(e))
    a.out.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
    print(f"Trained on {model['trained_on']} clips -> {a.out}")
    print(f"  How well each ranks clips it hasn't seen (Spearman, 5-fold cross-validation):")
    print(f"    Claude's score alone: {model['baseline_spearman']:+.3f}")
    print(f"    re-ranker:            {model['cv_spearman']:+.3f}")
    if model["enabled"]:
        print("  The re-ranker beats Claude's score, so the clip finder will use it.")
    else:
        print("  The re-ranker doesn't beat Claude's score yet, so it stays OFF. "
              "Record more clips and train again.")
    if model["trained_on"] < fb.GOOD_TRAIN:
        print(f"  Note: under {fb.GOOD_TRAIN} clips the numbers are noisy; retrain as data grows.")
    w = sorted(zip(model["features"], model["weights"]), key=lambda kv: abs(kv[1]), reverse=True)
    print("  Strongest factors (+ helps, - hurts): " +
          ", ".join(f"{k} {v:+.2f}" for k, v in w[:5]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="feedback.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", type=Path, default=fb.DEFAULT_LIBRARY, help="Library file")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="Record one posted clip's stats")
    r.add_argument("clips_json", type=Path)
    r.add_argument("clip", help="Clip number (as in the file names, from 1) or clip id")
    r.add_argument("--platform", default="tiktok", help="tiktok / youtube / instagram / ...")
    r.add_argument("--url", default="")
    r.add_argument("--views", type=lambda v: _num(v, int))
    r.add_argument("--avg-watch", type=_num, help="Average watch time in seconds")
    r.add_argument("--shares", type=lambda v: _num(v, int))
    r.add_argument("--likes", type=lambda v: _num(v, int))
    r.add_argument("--comments", type=lambda v: _num(v, int))
    r.set_defaults(func=cmd_record)

    i = sub.add_parser("import", help="Import stats from a CSV",
                       description="CSV columns: clip (number or id), platform, url, views, "
                                   "avg_watch, shares, likes, comments. Only clip and views are required.")
    i.add_argument("csv", type=Path)
    i.add_argument("--clips-json", type=Path, action="append", default=[],
                   help="clips.json the rows refer to (repeat for several streams)")
    i.add_argument("--platform", default="tiktok", help="Default when the CSV has no platform column")
    i.set_defaults(func=cmd_import)

    ls = sub.add_parser("list", help="Show the library, best first")
    ls.add_argument("--top", type=int, default=30)
    ls.set_defaults(func=cmd_list)

    ex = sub.add_parser("examples", help="Preview the examples Claude will see")
    ex.add_argument("--streamer")
    ex.add_argument("--good", type=int, default=8)
    ex.add_argument("--bad", type=int, default=3)
    ex.set_defaults(func=cmd_examples)

    t = sub.add_parser("train", help="Train the re-ranker")
    t.add_argument("--out", type=Path, default=fb.DEFAULT_RANKER)
    t.add_argument("--lam", type=float, default=1.0, help="Regularisation (higher = simpler model)")
    t.set_defaults(func=cmd_train)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
