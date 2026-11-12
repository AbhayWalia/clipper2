"""Animated captions, hook overlay, per-streamer layouts and audio/chat signals."""

import json
from pathlib import Path

import pytest

from clipfinder import llm
from clipfinder.captions import (CaptionStyle, ass_color, ass_time, build_ass, caption_events,
                                 group_words, words_in_window)
from clipfinder.layout import Layout, load_layouts, parse_box, save_layout, split_filter
from clipfinder.signals import (Signals, audio_events, chat_events, load_chat, merge_runs,
                                parse_loudness)
from clipfinder.transcript import Segment, Word, load_json


def seg_with_words(start, text, step=0.4):
    words = [Word(start + i * step, start + i * step + step * 0.9, w) for i, w in enumerate(text.split())]
    return Segment(words[0].start, words[-1].end, text, words)


# --- transcript words -----------------------------------------------------

def test_load_json_reads_word_timings():
    segs = load_json(json.dumps({"segments": [
        {"start": 0, "end": 1, "text": "hi there", "words": [
            {"start": 0, "end": 0.4, "word": " hi"}, {"start": 0.5, "end": 1, "word": "there"}]}]}))
    assert [(w.word, w.start) for w in segs[0].words] == [("hi", 0.0), ("there", 0.5)]


def test_timed_words_fallback_spreads_over_segment():
    words = Segment(10, 12, "aa bbbb").timed_words()
    assert words[0].start == 10 and abs(words[-1].end - 12) < 1e-9
    assert words[1].end - words[1].start > words[0].end - words[0].start  # longer word, more time


# --- captions -------------------------------------------------------------

def test_ass_helpers():
    assert ass_color("FFE500") == "&H0000E5FF"
    assert ass_time(3723.456) == "1:02:03.46"
    with pytest.raises(ValueError):
        ass_color("red")


def test_words_retimed_to_clip_and_grouped():
    segs = [seg_with_words(100, "no way they are chasing me"), seg_with_words(104, "wow.")]
    words = words_in_window(segs, 99, 110)
    assert words[0][0].start == pytest.approx(1.0)
    groups = group_words(words, max_words=3)
    assert [len(g) for g in groups] == [3, 3, 1]  # breaks at 3 words and at the new segment


def test_group_breaks_on_sentence_end_and_pause():
    w = [(Word(0, 0.3, "Stop."), 0), (Word(0.4, 0.6, "Go"), 0), (Word(2.0, 2.2, "now"), 0)]
    assert [[x.word for x in g] for g in group_words(w)] == [["Stop."], ["Go"], ["now"]]


def test_animated_events_highlight_each_word():
    g = [[Word(0, 0.4, "no"), Word(0.5, 0.9, "way")]]
    ev = caption_events(g, CaptionStyle())
    assert len(ev) == 2
    assert ev[0][2].startswith("{\\c&H0000E5FF") and ev[0][2].endswith(" WAY")
    assert "{\\c" in ev[1][2].split(" ", 1)[1]
    plain = caption_events(g, CaptionStyle(animated=False, uppercase=False))
    assert plain == [(0, 0.9, "no way")]


def test_build_ass_with_hook_and_seam(tmp_path):
    p = tmp_path / "c.ass"
    build_ass(p, [seg_with_words(5, "hello there friend")], 4, 10, (1080, 1920), CaptionStyle(),
              caption_y=0.35, hook="He had {FIVE} stars")
    text = p.read_text()
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text
    assert "Hook,,0,0,0,,He had (FIVE) stars" in text  # braces can't break ASS
    assert "\\pos(540,672)" in text  # 0.35 of 1920


def test_build_ass_hook_only(tmp_path):
    p = tmp_path / "c.ass"
    build_ass(p, [], 0, 10, (1920, 1080), CaptionStyle(), captions=False, hook="Wait")
    assert "Caption,," not in p.read_text().split("[Events]")[1]


# --- layouts --------------------------------------------------------------

def test_parse_box_fractions_and_pixels():
    assert parse_box("0.72,0.03,0.26,0.30") == (0.72, 0.03, 0.26, 0.30)
    assert parse_box("1382,32,500,324", (1920, 1080)) == pytest.approx(
        (0.7198, 0.0296, 0.2604, 0.3), abs=1e-4)
    with pytest.raises(ValueError):
        parse_box("1382,32,500,324")  # pixels without a frame size
    with pytest.raises(ValueError):
        parse_box("0.9,0,0.2,0.1")  # runs off the right edge


def test_layout_round_trip(tmp_path):
    p = tmp_path / "layouts.json"
    save_layout(p, "alice", Layout(facecam=(0.7, 0.0, 0.3, 0.3), cam_position="bottom"))
    save_layout(p, "bob", Layout(facecam=(0.0, 0.7, 0.25, 0.3)))
    got = load_layouts(p)
    assert set(got) == {"alice", "bob"}
    assert got["alice"].cam_position == "bottom" and got["alice"].seam() == pytest.approx(0.65)


def test_split_filter_even_sizes_and_order():
    f = split_filter(Layout(facecam=(0.7, 0, 0.3, 0.3), cam_height=0.33), 1080, 1920)
    assert "scale=1080:634" in f and "scale=1080:1286" in f  # 634 is even, 634+1286 = 1920
    assert "[cam][game]vstack" in f
    f = split_filter(Layout(facecam=(0.7, 0, 0.3, 0.3), cam_position="bottom"), 1080, 1920)
    assert "[game][cam]vstack" in f


# --- signals --------------------------------------------------------------

def test_parse_loudness_places_values_by_time():
    out = "\n".join([
        "frame:0    pts:0       pts_time:0", "lavfi.astats.Overall.RMS_level=-40.0",
        "frame:1    pts:8000    pts_time:1", "lavfi.astats.Overall.RMS_level=-inf",
        "frame:2    pts:24000   pts_time:3", "lavfi.astats.Overall.RMS_level=-10.0",
    ])
    lv = parse_loudness(out)
    assert lv[0] == -40.0 and lv[1] == -90.0 and lv[2] == -90.0 and lv[3] == pytest.approx(-10.0)


def test_audio_events_find_spike_not_steady_loudness():
    levels = [-30.0] * 300
    levels[150:153] = [-12.0, -10.0, -14.0]
    ev = audio_events(levels)
    assert len(ev) == 1 and ev[0].start == 150 and ev[0].end == 153 and ev[0].strength == 20
    assert audio_events([-5.0] * 300) == []  # constantly loud is not a spike


def test_merge_runs():
    assert merge_runs([(1, 5), (2, 7), (10, 3)], 1, 2) == [(1, 3, 7), (10, 11, 3)]


def test_load_chat_formats(tmp_path):
    td = tmp_path / "td.json"
    td.write_text(json.dumps({"comments": [{"content_offset_seconds": 5.5, "message": {"body": "KEKW"}}]}))
    cd = tmp_path / "cd.json"
    cd.write_text(json.dumps([{"time_in_seconds": 7, "message": "LUL"}]))
    tx = tmp_path / "chat.txt"
    tx.write_text("[0:01:02] some_user: no way\nnot a chat line\n")
    assert load_chat(td) == [(5.5, "KEKW")]
    assert load_chat(cd) == [(7.0, "LUL")]
    assert load_chat(tx) == [(62.0, "no way")]


def test_chat_burst_with_offset_and_top_words():
    msgs = [(float(t), "hi") for t in range(0, 600, 20)]
    msgs += [(300 + i * 0.3, "KEKW no way") for i in range(20)]
    msgs += [(301.0, "KEKW") for _ in range(3)]
    ev = chat_events(msgs, offset=10)
    assert len(ev) == 1 and ev[0].start == 310
    assert ev[0].detail.startswith('"KEKW" x23')


def test_signal_notes_mark_no_speech_and_feed_prompt():
    segs = [Segment(0, 5, "hello"), Segment(100, 105, "later")]
    sig = Signals(audio_events([-30.0] * 50 + [-10.0] + [-30.0] * 100), segs)
    notes = sig.notes(0, 200)
    assert "[50.0-51.0] audio spike, +20 dB above normal (no speech)" == notes
    cfg = llm.LLMConfig(system="s", notes=sig.notes)
    prompt = llm.section_prompt(segs, cfg)
    assert llm.SIGNALS_HEADER in prompt and "(no speech)" in prompt
    assert "audio spike" not in llm.section_prompt(segs, llm.LLMConfig(system="s"))


def test_signal_notes_keep_both_kinds_when_crowded():
    from clipfinder.signals import Event
    # 25 audio spikes (9-13 dB) would crowd out chat bursts (4-6x) in a single ranking.
    events = [Event(float(i * 10), float(i * 10 + 2), "audio", 9.0 + i % 5) for i in range(25)]
    events += [Event(50, 60, "chat", 6.0, "KEKW"), Event(120, 130, "chat", 4.0, "LUL")]
    notes = Signals(events, []).notes(0, 1000).splitlines()
    assert len(notes) == 20
    assert sum("chat burst" in n for n in notes) == 2           # both chat bursts survive
    assert sum("audio spike" in n for n in notes) == 18         # unused chat share goes to audio
    assert all("+13 dB" in n or "+12 dB" in n or "+11 dB" in n or "+10 dB" in n
               for n in notes if "audio" in n)                   # strongest audio kept
    starts = [float(n[1:n.index("-")]) for n in notes]
    assert starts == sorted(starts)                              # time order


def test_signal_notes_single_kind_uses_all_slots():
    from clipfinder.signals import Event
    events = [Event(float(i), float(i) + 1, "audio", float(i)) for i in range(30)]
    assert len(Signals(events, []).notes(0, 100).splitlines()) == 20
