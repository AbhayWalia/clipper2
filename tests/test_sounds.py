"""Sound tags: windowing, event detection, prompt notes and ranker features."""

import json
import shutil
import subprocess
import types

import numpy as np
import pytest

from clipfinder import sounds
from clipfinder.clips import Clip
from clipfinder.feedback import FEATURES, clip_features
from clipfinder.signals import Event, Signals
from clipfinder.transcript import Segment


def test_split_windows_one_per_second_across_blocks():
    sr = 10
    audio = np.arange(95, dtype=np.float32)  # 9.5 seconds
    first, used = sounds.split_windows(audio[:60], final=False, sr=sr, window=2.0)
    assert len(first) == 5 and used == 50        # windows at 0..4 s fit inside 6 s
    rest, _ = sounds.split_windows(np.concatenate([audio[used:60], audio[60:]]), final=True,
                                   sr=sr, window=2.0)
    windows = first + rest
    assert len(windows) == 10                     # one per started second
    assert windows[7][0] == 70 and len(windows[-1]) == 20 and windows[-1][-1] == 0  # padded


def test_sound_events_need_level_and_contrast():
    quiet = [0.0] * 100
    scores = {"explosion": quiet[:50] + [0.6, 0.7] + quiet[:48],
              "siren": [0.5] * 100,               # on the whole time: background, not an event
              "laughter": quiet[:20] + [0.2] + quiet[:79]}  # below the threshold
    [e] = sounds.sound_events(scores)
    assert (e.start, e.end, e.detail, e.strength) == (50.0, 52.0, "explosion", 0.7)


def test_notes_share_room_between_kinds():
    segs = [Segment(0, 5, "hi")]
    events = [Event(t, t + 1, "audio", 20.0) for t in range(0, 40, 2)] + \
             [Event(41, 42, "sound", 0.4, "gunfire"), Event(50, 51, "chat", 3.0)]
    notes = Signals(events, segs, per_range=6).notes(0, 100)
    assert "sounds like gunfire (40% likely) (no speech)" in notes
    assert "chat burst" in notes and notes.count("audio spike") == 4


def test_sound_features_for_ranker():
    clip = Clip(start=10, end=30, score=7, title="t", hook="h", category="chaos", reason="r")
    sig = Signals([Event(12, 14, "sound", 0.8, "explosion"), Event(50, 52, "sound", 0.9, "gunfire")],
                  [Segment(10, 30, "boom")])
    f = clip_features(clip, [Segment(10, 30, "boom")], sig)
    assert f["sound_explosion"] == 0.8 and f["sound_gunfire"] == 0.0
    assert all(f"sound_{g}" in FEATURES for g in sounds.GROUPS)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_tag_audio_streams_and_caches(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5.5",
                    str(wav)], check=True)

    class FakeTagger:  # loud windows -> "explosion"
        def inference(self, batch):
            probs = np.zeros((len(batch), 3))
            probs[:, 1] = np.abs(batch).max(axis=1)
            return probs, None
    model = (FakeTagger(), {"explosion": [1], "laughter": [0], "music": [2]})
    cache = tmp_path / "a.sounds.json"
    scores = sounds.tag_audio(wav, cache, model=model)
    assert len(scores["explosion"]) == 6 and scores["laughter"] == [0.0] * 6
    assert cache.exists() and sounds.tag_audio(wav, cache, model=None) == scores  # cached


def test_music_is_a_warning_not_a_highlight():
    scores = {"music": [0.0] * 10 + [0.6] * 20 + [0.1] * 10 + [0.5] * 3, "gunfire": [0.0] * 43}
    assert sounds.sound_events(scores) == []                 # never becomes a "sounds like" event
    ev = sounds.music_events(scores)
    assert [(e.start, e.end, e.kind) for e in ev] == [(10.0, 30.0, "music")]  # 3 s blip ignored
    assert sounds.music_share(ev, 0, 40) == 0.5
    assert sounds.music_share(ev, 30, 40) == 0.0


def test_old_caches_without_music_are_redone(tmp_path):
    cache = tmp_path / "a.sounds.json"
    cache.write_text('{"gunfire": [0.1]}')
    calls = []

    class FakeTagger:
        def inference(self, batch):
            calls.append(1)
            return np.zeros((len(batch), 2)), None
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=2", str(wav)], check=True)
    got = sounds.tag_audio(wav, cache, model=(FakeTagger(), {"gunfire": [0], "music": [1]}))
    assert calls and "music" in got and "music" in json.loads(cache.read_text())
