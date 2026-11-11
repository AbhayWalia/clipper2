"""Regression tests: clips from different VODs with the same filename must not merge."""

from clipfinder import feedback as fb
from clipfinder.clips import Clip


def clip(i=0, start=10.0, end=40.0, score=8.0, cat="chaos", **kw):
    return Clip(start, end, score, f"title {i}", f"hook {i}", cat, "r",
                id=kw.get("id", f"vod@{start:.1f}"), text=kw.get("text", f"words {i}"),
                features=kw.get("features", {}))


def test_clip_ids_differ_for_different_videos_with_same_name(tmp_path):
    from clipfinder.clips import clip_id, source_fingerprint
    a, b = tmp_path / "a" / "stream.mp4", tmp_path / "b" / "stream.mp4"
    for p, data in ((a, b"first vod" * 1000), (b, b"second vod" * 1000)):
        p.parent.mkdir()
        p.write_bytes(data)
    assert clip_id(a, 612.4) != clip_id(b, 612.4)
    assert clip_id(a, 612.4).startswith("stream-") and clip_id(a, 612.4).endswith("@612.4")
    moved = tmp_path / "renamed.mp4"
    moved.write_bytes(a.read_bytes())
    assert source_fingerprint(moved) == source_fingerprint(a)   # stable across moves


def test_library_keeps_colliding_old_ids_apart():
    # Old clips.json files: id from the filename alone, so two VODs collide.
    lib = {}
    one = clip(1, start=612.4, id="stream@612.4", text="first stream")
    two = clip(2, start=612.4, id="stream@612.4", text="second stream")
    fb.record_post(lib, one, streamer="a", video="/vods/a/stream.mp4", platform="tiktok", views=100)
    fb.record_post(lib, two, streamer="b", video="/vods/b/stream.mp4", platform="tiktok", views=900)
    assert set(lib) == {"stream@612.4", "stream@612.4#2"}
    assert lib["stream@612.4"]["text"] == "first stream"
    assert fb.totals(lib["stream@612.4"])["views"] == 100       # not merged
    # Re-recording the first clip still updates its own entry.
    fb.record_post(lib, one, streamer="a", video="/vods/a/stream.mp4", platform="tiktok", views=150)
    assert fb.totals(lib["stream@612.4"])["views"] == 150 and len(lib) == 2
