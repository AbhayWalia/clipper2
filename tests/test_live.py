"""--live: the pieces that decide what to transcribe, when to ask Claude and what's new."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfinder import live
from clipfinder.clips import Clip


def chunk(i, run=0, length=10.0, offset=0.0):
    return live.Chunk(Path(f"r{run}_{i:06d}.ts"), offset + i * length, offset + (i + 1) * length, run)


def clip(start, end, score=8, title="c"):
    return Clip(start, end, score, title, "h", "funny", "r")


def test_chunk_list_is_read_with_the_run_offset(tmp_path):
    (tmp_path / "r1.csv").write_text("r1_000000.ts,0.000000,10.010000\nr1_000001.ts,10.01,20.0\nbroken\n")
    got = live.read_chunk_list(tmp_path / "r1.csv", 1, 300.0)
    assert [(c.file.name, c.start, c.end, c.run) for c in got] == [
        ("r1_000000.ts", 300.0, 310.01, 1), ("r1_000001.ts", 310.01, 320.0, 1)]
    assert live.read_chunk_list(tmp_path / "missing.csv", 0, 0) == []


def test_place_writes_per_second_values_at_their_time():
    s = []
    live.place(s, 3.0, [1, 2], 0)
    assert s == [0, 0, 0, 1, 2]
    live.place(s, 4.0, [9, 9, 9], 0)  # overlapping write replaces
    assert s == [0, 0, 0, 1, 9, 9, 9]


def test_next_block_waits_for_enough_recording():
    chunks = [chunk(0), chunk(1)]
    assert live.next_block(chunks, set(), final=False) == []        # 20 s: wait
    assert live.next_block(chunks, set(), final=True) == chunks     # stream over: take it
    chunks.append(chunk(2))
    assert live.next_block(chunks, set(), final=False) == chunks    # 30 s
    done = {c.file for c in chunks}
    assert live.next_block(chunks, done, final=True) == []


def test_next_block_never_spans_a_reconnect():
    chunks = [chunk(0), chunk(0, run=1, offset=45.0), chunk(1, run=1, offset=45.0),
              chunk(2, run=1, offset=45.0)]
    assert live.next_block(chunks, set(), final=False) == chunks[:1]  # cut off by the gap
    assert live.next_block(chunks, {chunks[0].file}, final=False) == chunks[1:]


def test_fresh_clips_waits_for_the_aftermath_and_skips_repeats():
    have = [clip(100, 130)]
    found = [clip(105, 128, score=9, title="same moment again"),
             clip(200, 230, title="settled"),
             clip(400, 430, title="too recent")]
    new = live.fresh_clips(found, have, limit=300)
    assert [c.title for c in new] == ["settled"]


def test_fresh_clips_keeps_the_better_of_two_overlapping_finds():
    new = live.fresh_clips([clip(10, 40, 7, "weaker"), clip(15, 42, 9, "stronger")], [], 100)
    assert [c.title for c in new] == ["stronger"]


def test_chunks_for_a_clip_stay_in_one_run():
    chunks = [chunk(i) for i in range(5)] + [chunk(0, run=1, offset=60.0)]
    assert [c.file.name for c in live.chunks_for(chunks, 12, 33)] == [
        "r0_000001.ts", "r0_000002.ts", "r0_000003.ts"]
    assert live.chunks_for(chunks, 500, 520) == []


def session(tmp_path, **kw):
    args = SimpleNamespace(no_sound_tags=True, audio_threshold=8.0, chat_offset=0.0,
                           keep_recording=False, **kw)
    s = live.LiveSession(args, tmp_path)
    return s


def test_claude_looks_every_few_minutes_or_after_a_spike(tmp_path):
    s = session(tmp_path)
    s.analysed = 100
    assert not s.should_look(final=False)
    assert s.should_look(final=True)
    s.analysed = live.EVERY
    assert s.should_look(final=False)

    s = session(tmp_path)
    s.levels = [-30.0] * 200 + [-5.0] * 3 + [-30.0] * 100   # a loud moment at 200 s
    s.last_look, s.accepted = 150, 90                       # the regular look isn't due yet
    s.analysed = 230
    assert not s.should_look(final=False)                   # wait for the aftermath
    s.analysed = 200 + live.SETTLE
    assert s.should_look(final=False)
    s.last_look = 250
    assert not s.should_look(final=False)                   # too soon after the last look


def test_prune_deletes_only_old_analysed_chunks(tmp_path):
    s = session(tmp_path)
    files = []
    for i in range(3):
        f = tmp_path / f"c{i}.ts"
        f.write_bytes(b"x")
        files.append(live.Chunk(f, i * 600.0, i * 600.0 + 10, 0))
    s.done = {files[0].file, files[1].file}
    s.analysed = 1300
    s.prune(files)
    assert [f.file.exists() for f in files] == [False, True, True]


def test_render_argv_carries_the_look_of_the_clips(tmp_path):
    from clipfinder.cli import parse_args
    args = parse_args(["--live", "https://twitch.tv/x", "--vertical", "--burn-captions",
                       "--streamer", "alice", "--out", "elsewhere"])
    argv = live.render_argv(args, tmp_path / "clips.json", [3, 4])
    assert argv[:6] == ["--clips-json", str(tmp_path / "clips.json"), "--out", str(tmp_path),
                        "--only", "3,4"]
    assert "--vertical" in argv and "--burn-captions" in argv and "--live" not in argv
    assert argv[argv.index("--streamer") + 1] == "alice"
    parse_args(argv)  # what the renderer gets must parse


def test_live_rejects_other_sources():
    from clipfinder.cli import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--live", "https://twitch.tv/x", "--url", "https://twitch.tv/videos/1"])


def test_saving_keeps_review_and_trims_made_in_the_web_ui(tmp_path):
    import json
    s = session(tmp_path, live="u", model="m", effort="medium", streamer=None)
    s.clips = [clip(10, 40, title="a"), clip(100, 130, title="b")]
    s.save()
    data = json.loads(s.recipe.read_text())
    data["clips"][0]["review"] = "approved"                       # what the web UI writes
    data["clips"][1].update(review="rejected", start=101.0, original_start=100.0, edited_at=1.0)
    s.recipe.write_text(json.dumps(data))
    s.clips.append(clip(200, 230, title="c"))                      # a new clip arrives
    s.save()
    got = json.loads(s.recipe.read_text())["clips"]
    assert [c.get("review") for c in got] == ["approved", "rejected", None]
    assert got[1]["start"] == 101.0 and got[1]["original_start"] == 100.0
    assert s.clips[1].start == 101.0                               # later cuts use the trim
