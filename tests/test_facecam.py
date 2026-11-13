"""--facecam auto: find the streamer's cam once, then reuse it."""

import json

import pytest

from clipfinder import facecam


def test_answer_becomes_a_box_in_fractions():
    ans = {"found": True, "x": 0, "y": 30, "w": 210, "h": 120, "note": "top-left"}
    assert facecam.box_from_answer(ans, (768, 432)) == pytest.approx((0, 30 / 432, 210 / 768, 120 / 432))
    # off the edge: clipped to the frame
    ans = {"found": True, "x": 650, "y": 0, "w": 200, "h": 100, "note": ""}
    x, y, w, h = facecam.box_from_answer(ans, (768, 432))
    assert x + w == pytest.approx(1.0)
    assert facecam.box_from_answer({"found": False, "x": 0, "y": 0, "w": 0, "h": 0}, (768, 432)) is None
    assert facecam.box_from_answer({"found": True, "x": 0, "y": 0, "w": 10, "h": 10}, (768, 432)) is None


def test_frames_are_spread_across_the_video_and_sized_from_the_jpeg():
    assert facecam.frame_times(1000) == [150.0, 350.0, 550.0, 750.0]
    # A minimal JPEG header: SOI, then a SOF0 segment saying 432 high, 768 wide.
    sof = bytes([0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x01, 0xB0, 0x03, 0x00]) + bytes(12)
    assert facecam.jpeg_size(sof) == (768, 432)


def recipe(tmp_path, **extra):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    m = tmp_path / "clips.json"
    m.write_text(json.dumps({"video": str(video), "streamer": "alice", **extra, "clips": [
        {"start": 20.0, "end": 40.0, "score": 8, "title": "clip", "hook": "h",
         "category": "funny", "reason": "r"}]}))
    return m


@pytest.fixture
def cutting(tmp_path, monkeypatch):
    """Run the CLI's render path without ffmpeg; record the layout each clip got."""
    import clipfinder.cli as cli
    import clipfinder.render as render
    got = {"layouts": [], "asks": 0}
    monkeypatch.setattr(render, "run_cut_jobs",
                        lambda jobs, workers: (got["layouts"].extend(j[8] for j in jobs),
                                               ([j[1].name for j in jobs], []))[1])
    monkeypatch.setattr(cli, "keyframe_before", lambda v, t: t)
    monkeypatch.setattr(cli, "video_duration", lambda v: 600.0)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.chdir(tmp_path)  # layouts.json lands here
    return cli, got


def test_auto_finds_the_cam_once_and_saves_it_for_the_streamer(tmp_path, monkeypatch, cutting):
    cli, got = cutting

    def fake_find(sources, model):
        got["asks"] += 1
        assert len(sources) == 4 and sources[0][1] == 90.0   # frames across the 600 s video
        return (0.0, 0.07, 0.27, 0.27), "top-left"
    monkeypatch.setattr("clipfinder.facecam.find_facecam", fake_find)
    m = recipe(tmp_path)
    cli.main(["--clips-json", str(m), "--facecam", "auto"], quiet=True)
    layout = got["layouts"][0]
    assert layout.facecam == pytest.approx((0.0, 0.07, 0.27, 0.27)) and layout.cam_position == "top"
    assert json.loads(m.read_text())["layout"]["facecam"] == pytest.approx([0.0, 0.07, 0.27, 0.27])
    assert "alice" in json.loads((tmp_path / "layouts.json").read_text())

    cli.main(["--clips-json", str(m), "--facecam", "auto"], quiet=True)   # re-render
    assert got["asks"] == 1 and got["layouts"][1].facecam == layout.facecam

    cli.main(["--clips-json", str(m), "--facecam", "auto", "--no-facecam"], quiet=True)
    assert got["layouts"][2] is None


def test_no_cam_found_is_remembered_and_clips_stay_plain(tmp_path, monkeypatch, cutting):
    cli, got = cutting

    def fake_find(sources, model):
        got["asks"] += 1
        return None, "just gameplay"
    monkeypatch.setattr("clipfinder.facecam.find_facecam", fake_find)
    m = recipe(tmp_path)
    cli.main(["--clips-json", str(m), "--facecam", "auto"], quiet=True)
    cli.main(["--clips-json", str(m), "--facecam", "auto"], quiet=True)
    assert got["asks"] == 1 and got["layouts"] == [None, None]
    assert json.loads(m.read_text())["layout"] == {"facecam": None}
    assert not (tmp_path / "layouts.json").exists()


def test_a_failed_look_doesnt_stop_the_cut(tmp_path, monkeypatch, cutting, capsys):
    cli, got = cutting

    def broken(sources, model):
        raise RuntimeError("no API key")
    monkeypatch.setattr("clipfinder.facecam.find_facecam", broken)
    cli.main(["--clips-json", str(recipe(tmp_path)), "--facecam", "auto"], quiet=True)
    assert got["layouts"] == [None] and "cutting without it" in capsys.readouterr().out


def test_web_switch_turns_the_cam_on_and_off():
    from clipfinder.web import build_command
    on, _ = build_command({"source": "url", "url": "u", "facecam": True})
    off, _ = build_command({"source": "url", "url": "u", "facecam": False})
    assert on[on.index("--facecam") + 1] == "auto" and "--no-facecam" in off


def test_each_clip_can_have_its_cam_on_or_off(tmp_path, monkeypatch, cutting):
    # The web UI's per-clip button writes "facecam" into clips.json; it beats the run's setting.
    cli, got = cutting
    monkeypatch.setattr("clipfinder.facecam.find_facecam", lambda s, m: ((0.0, 0.07, 0.27, 0.27), ""))
    m = recipe(tmp_path)
    data = json.loads(m.read_text())
    data["clips"] = [dict(data["clips"][0], title=f"clip {i}", start=20.0 + 60 * i, end=40.0 + 60 * i)
                     for i in range(3)]
    data["clips"][0]["facecam"] = False   # cam taken out of clip 1
    data["clips"][2]["facecam"] = True    # ...and forced into clip 3
    m.write_text(json.dumps(data))

    cli.main(["--clips-json", str(m), "--facecam", "auto"], quiet=True)
    assert [lay is not None for lay in got["layouts"]] == [False, True, True]

    got["layouts"].clear()
    cli.main(["--clips-json", str(m), "--no-facecam"], quiet=True)   # run says no cam...
    assert [lay is not None for lay in got["layouts"]] == [False, False, True]  # ...clip 3 still wants it


def test_web_saves_a_clips_cam_choice(tmp_path, monkeypatch):
    from clipfinder import web
    monkeypatch.setattr(web, "ROOT", tmp_path)
    m = recipe(tmp_path)
    assert web.update_clip({"recipe": "clips.json", "n": 1, "facecam": False})["facecam"] is False
    clip = json.loads(m.read_text())["clips"][0]
    assert clip["facecam"] is False and clip["edited_at"] > 0
    web.update_clip({"recipe": "clips.json", "n": 1, "facecam": None})
    assert "facecam" not in json.loads(m.read_text())["clips"][0]
    with pytest.raises(ValueError):
        web.update_clip({"recipe": "clips.json", "n": 1, "facecam": "yes"})
