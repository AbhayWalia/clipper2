from clipfinder.clips import Clip, clean_clips, load_manifest, save_manifest, snap_to_segments
from clipfinder.llm import render_prompt
from clipfinder.render import fmt_srt_time, write_clip_srt
from clipfinder.transcript import (Segment, chunk_segments, load_json as _load_json,
                                   load_srt_vtt as _load_srt_vtt, load_txt as _load_txt,
                                   ts_to_seconds as _ts_to_seconds)


def clip(start, end, score=8.0):
    return Clip(start, end, score, "t", "h", "funny", "r")


# Ten 5-second sentences: 0-5, 5-10, ... 45-50
SEGS = [Segment(i * 5.0, i * 5.0 + 5.0, f"line {i}") for i in range(10)]


# --- parsing --------------------------------------------------------------

def test_ts_to_seconds():
    assert _ts_to_seconds("01:02:03,456") == 3723.456
    assert _ts_to_seconds("02:03.5") == 123.5
    assert _ts_to_seconds("83.2") == 83.2


def test_load_srt():
    text = "1\n00:00:01,000 --> 00:00:02,500\nhello <i>there</i>\n\n2\n00:00:03,000 --> 00:00:04,000\nbye\n"
    segs = _load_srt_vtt(text)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.0, 2.5, "hello there"), (3.0, 4.0, "bye")]


def test_load_vtt_short_timestamps():
    text = "WEBVTT\n\n00:01.000 --> 00:02.000\nhi\n"
    assert _load_srt_vtt(text)[0].start == 1.0


def test_load_json_skips_empty_text():
    segs = _load_json('{"segments": [{"start": 0, "end": 1, "text": " a "}, {"start": 1, "end": 2, "text": ""}]}')
    assert len(segs) == 1 and segs[0].text == "a"


def test_load_txt_fills_missing_ends():
    segs = _load_txt("[00:00:01] one\n[00:00:04] two\n[83.2 - 87.9] three")
    assert (segs[0].start, segs[0].end) == (1.0, 4.0)
    assert (segs[2].start, segs[2].end) == (83.2, 87.9)


# --- snapping & cleaning --------------------------------------------------

def test_snap_short_clip_never_inverts():
    # Old behaviour: start snapped to 10, end snapped to 10 -> zero/negative length.
    c = snap_to_segments(clip(8.0, 9.0), SEGS)
    assert c.end > c.start


def test_snap_moves_to_nearby_boundaries():
    c = snap_to_segments(clip(9.0, 21.0), SEGS)
    assert (c.start, c.end) == (10.0, 20.0)


def test_snap_never_moves_an_edge_far_inward():
    # Speech only at 34-37; the action (silent) is at 28-34. The start must stay put.
    segs = [Segment(0.0, 5.0, "earlier"), Segment(34.0, 37.0, "did you see that")]
    c = snap_to_segments(clip(28.0, 40.0), segs)
    assert c.start == 28.0 and c.end == 40.0   # 37 is 3 s inward: too far
    c = snap_to_segments(clip(15.0, 25.0), [Segment(18.0, 18.6, "BOOM!"), Segment(21, 24.5, "x")])
    assert c.start == 15.0 and c.end == 24.5   # start keeps the build-up; end moves out


def test_snap_leaves_edge_when_no_boundary_nearby():
    long_seg = [Segment(0.0, 100.0, "one long segment")]
    c = snap_to_segments(clip(40.0, 60.0), long_seg)
    assert (c.start, c.end) == (40.0, 60.0)


def test_clean_respects_max_len_including_padding():
    out = clean_clips([clip(0.0, 50.0)], SEGS, min_len=12, max_len=30,
                      min_score=0, top=0, video_len=None, pad=3.0)
    assert out[0].duration + 3.0 <= 30.0


def test_clean_min_len_including_padding():
    out = clean_clips([clip(10.0, 15.0)], SEGS, min_len=12, max_len=75,
                      min_score=0, top=0, video_len=None, pad=3.0)
    assert out[0].duration + 3.0 >= 12.0


def test_clean_overlap_keeps_higher_score_and_top_n():
    clips = [clip(0, 20, score=7), clip(10, 30, score=9), clip(35, 50, score=6)]
    out = clean_clips(clips, SEGS, 5, 75, min_score=0, top=0, video_len=None)
    assert [c.score for c in out] == [9, 6]
    out = clean_clips([clip(0, 20, 7), clip(25, 45, 9)], SEGS, 5, 75, 0, top=1, video_len=None)
    assert [c.score for c in out] == [9]


def test_clean_drops_low_scores_and_clamps_to_video():
    out = clean_clips([clip(0, 20, score=3), clip(30, 50, score=8)], SEGS, 5, 75,
                      min_score=6, top=0, video_len=45.0)
    assert len(out) == 1 and out[0].end == 45.0


# --- misc -----------------------------------------------------------------

def test_chunk_segments_overlap():
    chunks = chunk_segments(SEGS, window=20, overlap=5)
    assert chunks[0][0].start == 0.0
    assert chunks[1][0].start <= 15.0  # second window starts at 15
    assert chunks[-1][-1].end == 50.0


def test_fmt_srt_time():
    assert fmt_srt_time(0) == "00:00:00,000"
    assert fmt_srt_time(3723.456) == "01:02:03,456"
    assert fmt_srt_time(-1) == "00:00:00,000"


def test_render_prompt_tolerates_other_braces():
    assert render_prompt("len {min_len}-{max_len} json {x}", 12, 75) == "len 12-75 json {x}"


def test_clip_srt_includes_tail_padding(tmp_path):
    p = tmp_path / "c.srt"
    write_clip_srt(p, SEGS, t0=8.5, t1=21.5)  # clip 10-20 with 1.5 s pads
    text = p.read_text()
    assert "line 4" in text  # the line that starts in the tail padding is captioned
    assert "00:00:13,000" in text  # captions run to the padded end


def test_manifest_round_trip(tmp_path):
    p = tmp_path / "clips.json"
    save_manifest(p, [clip(1, 20)], video=tmp_path / "v.mp4", model="m")
    meta, clips = load_manifest(p)
    assert meta["video"].endswith("v.mp4") and meta["model"] == "m"
    assert clips[0].start == 1 and clips[0].end == 20
