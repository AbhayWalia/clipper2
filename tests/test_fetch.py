"""--url fetching (yt-dlp / chat-downloader faked) and YouTube auto-caption parsing."""

import json
import sys
import types
from pathlib import Path

from clipfinder import fetch
from clipfinder.signals import load_chat
from clipfinder.transcript import load_srt_vtt

YT_AUTO = """WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.270 align:start position:0%

no<00:00:00.480><c> way</c><00:00:00.800><c> they're</c><00:00:01.120><c> shooting</c>

00:00:02.270 --> 00:00:02.280 align:start position:0%
no way they're shooting


00:00:02.280 --> 00:00:04.110 align:start position:0%
no way they're shooting
at<00:00:02.640><c> me</c><00:00:03.000><c> five</c><00:00:03.400><c> stars</c>
"""


def test_youtube_auto_captions_dedupe_and_word_times():
    segs = load_srt_vtt(YT_AUTO)
    assert [s.text for s in segs] == ["no way they're shooting", "at me five stars"]
    assert [(w.word, round(w.start, 2)) for w in segs[0].words] == [
        ("no", 0.16), ("way", 0.48), ("they're", 0.8), ("shooting", 1.12)]
    assert segs[1].words[-1].end == 4.11


def test_youtube_auto_captions_with_literal_space_lines():
    with_spaces = YT_AUTO.replace("position:0%\n\nno<", "position:0%\n \nno<")
    assert " \nno<" in with_spaces
    assert [s.text for s in load_srt_vtt(with_spaces)] == ["no way they're shooting", "at me five stars"]


def test_plain_vtt_keeps_repeated_lines():
    # Without inline timestamps it's normal captions: "Go!" twice is real speech.
    text = "WEBVTT\n\n00:01.000 --> 00:02.000\nGo!\n\n00:02.000 --> 00:03.000\nGo!\n"
    assert [s.text for s in load_srt_vtt(text)] == ["Go!", "Go!"]


def test_srt_indexes_not_in_text():
    text = "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:03,000 --> 00:00:04,000\nbye\n"
    assert [s.text for s in load_srt_vtt(text)] == ["hello", "bye"]


def test_ytdlp_options():
    o = fetch.ytdlp_options(Path("/x"), video=False, captions=True, max_height=720)
    assert o["skip_download"] and o["writeautomaticsub"] and "height<=720" in o["format"]
    assert o["merge_output_format"] == "mp4"


class FakeYDL:
    """Stands in for yt_dlp.YoutubeDL: 'downloads' by creating the files."""
    last_opts = None

    def __init__(self, opts):
        FakeYDL.last_opts = opts
        self.dest = Path(opts["outtmpl"]).parent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=True):
        (self.dest / "Twitch_v123.mp4").write_bytes(b"video")
        vtt = self.dest / "Twitch_v123.en.vtt"
        vtt.write_text(YT_AUTO)
        return {"id": "v123", "ext": "mp4", "title": "Heist night", "uploader": "Big Streamer!",
                "duration": 3600, "requested_subtitles": {"en": {"filepath": str(vtt)}}}

    def prepare_filename(self, info):
        return str(self.dest / "Twitch_v123.mp4")


def fake_modules(monkeypatch, chat_rows):
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))

    class FakeChat:
        def get_chat(self, url):
            return iter(chat_rows)
    monkeypatch.setitem(sys.modules, "chat_downloader",
                        types.SimpleNamespace(ChatDownloader=FakeChat))


def test_fetch_source_end_to_end(tmp_path, monkeypatch):
    fake_modules(monkeypatch, [{"time_in_seconds": 5.0, "message": "KEKW"},
                               {"message": "no time, skipped"},
                               {"time_in_seconds": 9.5, "message": "LUL"}])
    got = fetch.fetch_source("https://twitch.tv/videos/123", tmp_path)
    assert got.video == tmp_path / "Twitch_v123.mp4"
    assert got.captions == tmp_path / "Twitch_v123.en.vtt"
    assert got.title == "Heist night" and got.duration == 3600
    assert load_chat(got.chat) == [(5.0, "KEKW"), (9.5, "LUL")]   # our chat loader reads it
    assert fetch.streamer_slug(got.uploader) == "big_streamer"


def test_chat_is_reused_and_failures_dont_block(tmp_path, monkeypatch):
    out = tmp_path / "x.chat.json"
    out.write_text(json.dumps([{"time_in_seconds": 1, "message": "old"}]))
    assert fetch.download_chat("u", out) == out          # existing file reused

    class Boom:
        def get_chat(self, url):
            raise RuntimeError("chat API changed")
    monkeypatch.setitem(sys.modules, "chat_downloader", types.SimpleNamespace(ChatDownloader=Boom))
    assert fetch.download_chat("u", tmp_path / "y.chat.json") is None
