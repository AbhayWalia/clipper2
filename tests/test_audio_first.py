"""--audio-first: download the audio to find clips, then only the video around each clip."""

import json
import sys
import types
from pathlib import Path

import pytest

from clipfinder import fetch


def test_audio_only_options_download_the_audio_track():
    o = fetch.ytdlp_options(Path("/x"), video=False, audio_only=True)
    assert o["format"].startswith("ba") and not o["skip_download"]


def test_fetch_source_audio_only(tmp_path, monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            assert self.opts["format"].startswith("ba")
            (tmp_path / "Twitch_v1.m4a").write_bytes(b"audio")
            return {"id": "v1", "title": "t", "webpage_url": "https://twitch.tv/videos/1"}

        def prepare_filename(self, info):
            return str(tmp_path / "Twitch_v1.m4a")
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    got = fetch.fetch_source("twitch.tv/videos/1", tmp_path, chat=False, audio_only=True)
    assert got.audio == tmp_path / "Twitch_v1.m4a" and got.video is None
    assert got.url == "https://twitch.tv/videos/1"


def fake_section_ytdlp(monkeypatch, calls, fail_at=None):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            (s, e), = self.opts["download_ranges"]
            calls.append((s, e))
            if s == fail_at:
                raise RuntimeError("ERROR: HTTP Error 403: Forbidden")
            Path(self.opts["outtmpl"].replace(".%(ext)s", ".mp4")).write_bytes(b"v")
            return {}
    utils = types.SimpleNamespace(download_range_func=lambda chapters, ranges: ranges)
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL, utils=utils))
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", utils)


def test_download_sections_skips_failures_and_reuses_files(tmp_path, monkeypatch, capsys):
    calls = []
    fake_section_ytdlp(monkeypatch, calls, fail_at=200.0)
    segs = fetch.download_sections("u", tmp_path, [(10.0, 70.0), (200.0, 260.0)], jobs=1)
    assert segs == [{"file": fetch.section_name(10.0, 70.0), "start": 10.0, "end": 70.0}]
    assert (tmp_path / segs[0]["file"]).exists()
    assert "section 200-260s failed: HTTP Error 403" in capsys.readouterr().out
    calls.clear()
    assert fetch.download_sections("u", tmp_path, [(10.0, 70.0)]) == segs
    assert calls == []  # already on disk


def recipe(tmp_path, n=3):
    m = tmp_path / "clips.json"
    m.write_text(json.dumps({"video": None, "audio": str(tmp_path / "a.m4a"),
                             "source_url": "https://twitch.tv/videos/1", "clips": [
        {"start": 100.0 + i * 300, "end": 130.0 + i * 300, "score": 8, "title": f"clip {i}",
         "hook": "h", "category": "funny", "reason": "r"} for i in range(n)]}))
    return m


def test_fetch_sections_downloads_only_what_is_missing(tmp_path, monkeypatch):
    import clipfinder.cli as cli
    asked = []

    def fake(url, folder, windows, max_height):
        asked.extend(windows)
        return [{"file": fetch.section_name(s, e), "start": s, "end": e} for s, e in windows]
    monkeypatch.setattr(cli, "download_sections", fake)
    m = recipe(tmp_path)
    args = cli.parse_args(["--url", "u", "--audio-first"])
    args.segment_pad = fetch.SECTION_PAD  # main() sets this once it knows sections are in use
    have = [{"file": "segments/old.mp4", "start": 90.0, "end": 150.0}]
    segs = cli.fetch_sections(args, "u", [(98.5, 131.5), (398.5, 431.5)], have, tmp_path, m, None)
    assert asked == [(388.5, 441.5)]  # the first span is covered; the second gets 10 s each side
    assert segs[0] == have[0] and len(segs) == 2
    assert json.loads(m.read_text())["segments"] == segs


def test_rerender_of_an_audio_first_recipe_fetches_only_the_chosen_clips(tmp_path, monkeypatch):
    # "Render approved" in the web UI: clips.json has no video, only the source URL.
    import clipfinder.cli as cli
    import clipfinder.render as render
    windows, cut = [], []

    def fake_download(url, folder, wins, max_height):
        windows.extend(wins)
        out = []
        for s, e in wins:
            (folder / "segments").mkdir(exist_ok=True)
            (folder / fetch.section_name(s, e)).write_bytes(b"v")
            out.append({"file": fetch.section_name(s, e), "start": s, "end": e})
        return out
    monkeypatch.setattr(cli, "download_sections", fake_download)
    monkeypatch.setattr(cli, "keyframe_before", lambda video, t: t)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(render, "run_cut_jobs",
                        lambda jobs, workers: (cut.extend(jobs), ([j[1].name for j in jobs], []))[1])
    m = recipe(tmp_path)
    argv = ["--clips-json", str(m), "--only", "1,3"]
    cli.main(argv)
    assert windows == [(88.5, 141.5), (688.5, 741.5)]
    assert [j[1].name for j in cut] == ["01_clip_0.mp4", "03_clip_2.mp4"]
    src, _, start, end = cut[0][:4]
    assert src == tmp_path / fetch.section_name(88.5, 141.5) and (start, end) == (10.0, 43.0)

    windows.clear()
    cli.main(argv)  # sections are on disk and in clips.json now: nothing to download
    assert windows == []


def test_audio_first_needs_a_url():
    import clipfinder.cli as cli
    with pytest.raises(SystemExit):
        cli.parse_args(["--video", "v.mp4", "--audio-first"])


def test_web_ui_passes_audio_first_for_links_only():
    from clipfinder.web import build_command
    cmd, _ = build_command({"source": "url", "url": "u", "audio_first": True})
    assert "--audio-first" in cmd
    cmd, _ = build_command({"source": "video", "video": "v.mp4", "audio_first": True})
    assert "--audio-first" not in cmd
