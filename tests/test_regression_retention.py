"""Regression tests: retention is measured against the posted length (padding included)."""

import pytest

from clipfinder import feedback as fb
from clipfinder.transcript import Segment
from clipfinder.clips import Clip


def clip(i=0, start=10.0, end=40.0, score=8.0, cat="chaos", **kw):
    return Clip(start, end, score, f"title {i}", f"hook {i}", cat, "r",
                id=kw.get("id", f"vod@{start:.1f}"), text=kw.get("text", f"words {i}"),
                features=kw.get("features", {}))


def test_retention_uses_posted_length_with_padding():
    # A 12 s clip is posted as 15 s (1.5 s padding each side). Watching 12 s on
    # average is 80% retention, not the 100% the content length gave.
    segs = [Segment(10, 22, "words")]
    c = clip(1, start=10, end=22, features=None)
    c.features = fb.clip_features(c, segs, pad=3.0)
    assert c.features["posted_length"] == 15.0
    assert "posted_length" not in fb.FEATURES          # bookkeeping, not a ranker input
    lib = {}
    e = fb.record_post(lib, c, streamer="a", video="v", platform="tiktok", views=100, avg_watch=12)
    assert e["duration"] == 15.0
    assert fb.totals(e)["retention"] == pytest.approx(0.8)


def test_retention_for_old_clips_assumes_default_padding():
    old = clip(1, start=10, end=22, features={"claude_score": 8})   # no posted_length
    e = fb.record_post({}, old, streamer="a", video="v", platform="tiktok", views=1, avg_watch=12)
    assert e["duration"] == 12 + fb.DEFAULT_PAD


def test_custom_padding_is_recorded():
    c = clip(1, start=0, end=20)
    assert fb.clip_features(c, [], pad=5.0)["posted_length"] == 25.0
