"""The feedback layer: library, examples, re-ranker and the feedback.py command."""

import csv
import json
import math
import random

import pytest

from clipfinder import feedback as fb
from clipfinder import llm
from clipfinder.clips import Clip, load_manifest, save_manifest
from clipfinder.feedback_cli import main as fb_main
from clipfinder.signals import Event, Signals
from clipfinder.transcript import Segment


def clip(i=0, start=10.0, end=40.0, score=8.0, cat="chaos", **kw):
    return Clip(start, end, score, f"title {i}", f"hook {i}", cat, "r",
                id=kw.get("id", f"vod@{start:.1f}"), text=kw.get("text", f"words {i}"),
                features=kw.get("features", {}))


def test_old_manifests_still_load():
    c = Clip.from_dict({"start": 1, "end": 2, "score": 5, "title": "t", "hook": "h",
                        "category": "funny", "reason": "r"})
    assert c.id == "" and c.features == {} and c.predicted is None


def test_clip_features_with_signals():
    segs = [Segment(10, 20, "one two three four five"), Segment(30, 35, "six")]
    sig = Signals([Event(22, 24, "audio", 15.0), Event(45, 55, "chat", 4.0, ""),
                   Event(200, 210, "chat", 9.0, "")], segs)
    f = fb.clip_features(clip(start=10, end=40, cat="skill"), segs, sig)
    assert f["duration"] == 30 and f["words_per_sec"] == 0.2
    assert f["speech_fraction"] == 0.5          # 15 of 30 seconds have speech
    assert f["audio_peak_db"] == 15 and f["audio_spikes"] == 1
    assert f["chat_peak_ratio"] == 4 and f["chat_bursts"] == 1   # chat lag: 45 < 40 + 15
    assert f["cat_skill"] == 1 and f["cat_chaos"] == 0
    assert list(fb.clip_features(clip(), segs).keys())[:9] == fb.FEATURES[:9]


def test_record_post_merges_platforms():
    lib = {}
    c = clip(features={"claude_score": 8})
    fb.record_post(lib, c, streamer="alice", video="v.mp4", platform="tiktok",
                   views=1000, avg_watch=15, today="2026-11-10")
    fb.record_post(lib, c, streamer="alice", video="v.mp4", platform="youtube",
                   views=3000, avg_watch=25, shares=7, today="2026-11-10")
    fb.record_post(lib, c, streamer="alice", video="v.mp4", platform="tiktok",
                   views=2000, today="2026-11-11")       # stats grew; watch time kept
    t = fb.totals(lib[c.id])
    assert t["views"] == 5000 and t["shares"] == 7
    assert t["avg_watch"] == pytest.approx((15 * 2000 + 25 * 3000) / 5000)
    assert t["retention"] == pytest.approx(t["avg_watch"] / 30)
    with pytest.raises(ValueError):
        fb.record_post(lib, clip(id=""), streamer="a", video="v", platform="tiktok", views=1)


def make_lib(n, streamers=("alice", "bob"), signal=True, seed=1):
    """Synthetic library: performance driven by audio spikes, not by Claude's score."""
    rnd = random.Random(seed)
    lib = {}
    for i in range(n):
        s = streamers[i % len(streamers)]
        audio = rnd.uniform(0, 20)
        score = rnd.uniform(5, 10)
        base = 10000 if s == "alice" else 500           # big vs small channel
        quality = audio if signal else rnd.uniform(0, 20)
        views = int(base * math.exp(quality / 8 + rnd.gauss(0, 0.3)))
        c = clip(i, start=float(i * 100), score=score, cat=rnd.choice(fb.CATEGORIES),
                 features={"claude_score": score, "duration": 30, "audio_peak_db": audio,
                           "hook_chars": 20})
        fb.record_post(lib, c, streamer=s, video="v", platform="tiktok", views=views,
                       avg_watch=20, today="2026-11-10")
    return lib


def test_relative_performance_removes_channel_size():
    lib = make_lib(40)
    perf = fb.relative_performance(fb.with_stats(lib))
    alice = [v for k, v in perf.items() if lib[k]["streamer"] == "alice"]
    bob = [v for k, v in perf.items() if lib[k]["streamer"] == "bob"]
    # Both centred on 0 even though alice gets ~20x the views.
    assert abs(sorted(alice)[len(alice) // 2]) < 0.3 and abs(sorted(bob)[len(bob) // 2]) < 0.3


def test_examples_need_minimum_and_prefer_streamer():
    assert fb.examples_prompt(make_lib(fb.MIN_EXAMPLES - 1)) == ""
    lib = make_lib(30)
    text = fb.examples_prompt(lib, n_good=4, n_bad=2, streamer="bob")
    assert text.count("<example ") == 6
    good = text.split("underperformed")[0]
    assert good.count('streamer="bob"') == 4          # same streamer's winners first
    assert text == fb.examples_prompt(lib, n_good=4, n_bad=2, streamer="bob")  # stable for caching


def test_ridge_recovers_a_linear_rule():
    rnd = random.Random(0)
    X = [[rnd.uniform(0, 10), rnd.uniform(0, 10)] for _ in range(200)]
    y = [3 * a - 2 * b + 5 for a, b in X]
    m = fb.fit_ridge(X, y, lam=0.001)
    assert fb.predict(m, [1, 1]) == pytest.approx(6, abs=0.05)


def test_spearman():
    assert fb.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert fb.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert fb.spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_train_turns_on_only_when_it_beats_claude():
    with pytest.raises(ValueError):
        fb.train_ranker(make_lib(fb.MIN_TRAIN - 1))
    m = fb.train_ranker(make_lib(120))
    assert m["enabled"] and m["cv_spearman"] > m["baseline_spearman"] + 0.3
    noise = fb.train_ranker(make_lib(120, signal=False))
    assert not noise["enabled"]


def test_rank_clips_picks_by_prediction():
    m = fb.train_ranker(make_lib(120))
    a = clip(1, start=0, score=10, features={"claude_score": 10, "audio_peak_db": 0, "duration": 30})
    b = clip(2, start=100, score=6, features={"claude_score": 6, "audio_peak_db": 18, "duration": 30})
    top = fb.rank_clips([a, b], m, top=1)
    assert top == [b] and b.predicted > a.predicted


def test_stale_ranker_is_ignored(tmp_path):
    p = tmp_path / "ranker.json"
    p.write_text(json.dumps({"features": ["old"], "enabled": True}))
    assert fb.load_ranker(p) is None


def test_cached_system_prompt_param():
    cfg = llm.LLMConfig(system="S", cache_system=True)
    sysp = llm.request_params(cfg, "m", "u", llm.CLIP_SCHEMA)["system"]
    assert sysp == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    assert llm.request_params(llm.LLMConfig(system="S"), "m", "u", llm.CLIP_SCHEMA)["system"] == "S"


def test_feedback_cli_end_to_end(tmp_path, capsys):
    mf, lib, ranker = tmp_path / "clips.json", tmp_path / "library.json", tmp_path / "ranker.json"
    clips = [clip(i, start=float(i * 60), features={"claude_score": 8}) for i in range(3)]
    save_manifest(mf, clips, video="v.mp4", streamer="alice")

    fb_main(["--library", str(lib), "record", str(mf), "2", "--views", "12,000",
             "--avg-watch", "14.5", "--platform", "youtube", "--url", "https://y/1"])
    entry = json.loads(lib.read_text())["vod@60.0"]
    assert entry["streamer"] == "alice" and entry["posts"]["youtube"]["views"] == 12000

    rows = tmp_path / "stats.csv"
    with open(rows, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Clip", "Views", "avg_watch", "Platform"])
        w.writerow(["1", "500", "8", "tiktok"])
        w.writerow(["vod@120.0", "90000", "", "tiktok"])
        w.writerow(["nope", "1", "", ""])
    fb_main(["--library", str(lib), "import", str(rows), "--clips-json", str(mf)])
    out = capsys.readouterr().out
    assert "Imported 2 row(s), skipped 1" in out
    fb_main(["--library", str(lib), "list"])
    assert "90,000" in capsys.readouterr().out

    with pytest.raises(SystemExit):  # not enough data yet
        fb_main(["--library", str(lib), "train", "--out", str(ranker)])
    lib.write_text(json.dumps(make_lib(80)))
    fb_main(["--library", str(lib), "train", "--out", str(ranker)])
    assert "will use it" in capsys.readouterr().out
    assert fb.load_ranker(ranker)["enabled"]


def test_import_skips_bad_rows_and_saves_the_rest(tmp_path, capsys):
    # A bad clip number used to sys.exit mid-import, losing every row before it.
    mf, lib = tmp_path / "clips.json", tmp_path / "library.json"
    save_manifest(mf, [clip(i, start=float(i * 60)) for i in range(3)], video="v.mp4")
    rows = tmp_path / "stats.csv"
    with open(rows, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "views"])
        w.writerow(["1", "100"])
        w.writerow(["16", "5"])        # no such clip
        w.writerow(["2", "lots"])      # not a number
        w.writerow(["3", "300"])
    fb_main(["--library", str(lib), "import", str(rows), "--clips-json", str(mf)])
    out = capsys.readouterr().out
    assert "line 3: Clip 16 doesn't exist" in out and "line 4: a number couldn't be read" in out
    assert "Imported 2 row(s), skipped 2" in out
    saved = json.loads(lib.read_text())
    assert set(saved) == {"vod@0.0", "vod@120.0"}


def test_record_still_stops_on_unknown_clip(tmp_path):
    mf = tmp_path / "clips.json"
    save_manifest(mf, [clip(0)], video="v.mp4")
    with pytest.raises(SystemExit, match="Clip 5 doesn't exist"):
        fb_main(["--library", str(tmp_path / "l.json"), "record", str(mf), "5", "--views", "1"])
