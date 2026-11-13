"""Web UI: logging results, training, the cam layout editor, model options."""

import json

import pytest

from clipfinder import feedback as fb
from clipfinder import web


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    run = tmp_path / "clips"
    run.mkdir()
    (run / "clips.json").write_text(json.dumps({"video": "/vods/alice.mp4", "streamer": "alice", "clips": [
        {"start": 10.0 + 60 * i, "end": 40.0 + 60 * i, "score": 8, "title": f"Clip {i}", "hook": "h",
         "category": "chaos", "reason": "r", "id": f"alice-abc@{10 + 60 * i}.0",
         "features": {"claude_score": 8, "duration": 30}} for i in range(3)]}))
    (run / "01_clip_0.mp4").write_bytes(b"x")
    return tmp_path


def test_logging_results_shows_on_the_clip_and_in_the_summary(project):
    out = web.results.record(project, project / "clips/clips.json", 1,
                             {"platform": "TikTok", "views": "12,400", "avg_watch": "15.5", "likes": "900"})
    assert out["views"] == 12400 and out["with_stats"] == 1
    lib = json.loads((project / "library.json").read_text())
    post = lib["alice-abc@10.0"]["posts"]["tiktok"]
    assert post["views"] == 12400 and post["avg_watch"] == 15.5 and post["likes"] == 900
    clip = next(f for f in web.library() if f["name"] == "clips")["clips"][0]
    assert clip["results"]["views"] == 12400 and clip["has_id"]
    s = web.results.summary(project)
    assert s["with_stats"] == 1 and s["entries"][0]["title"] == "Clip 0" and s["ranker"] is None
    # updating replaces the numbers rather than adding a second post
    web.results.record(project, project / "clips/clips.json", 1, {"platform": "tiktok", "views": "20000"})
    assert web.results.summary(project)["entries"][0]["views"] == 20000


@pytest.mark.parametrize("form", [
    {"platform": "tiktok"},                                   # views are required
    {"platform": "tiktok", "views": "-5"},
    {"platform": "tiktok", "views": "lots"},
    {"platform": "<script>", "views": "5"},
])
def test_bad_results_are_refused(project, form):
    with pytest.raises(ValueError):
        web.results.record(project, project / "clips/clips.json", 1, form)


def test_csv_import_and_training_need_enough_data(project):
    csv = "clip,views,avg_watch\nalice-abc@70.0,5000,9\nnope,10,1\n"
    log = web.results.import_csv(project, csv, "youtube", [project / "clips/clips.json"])["log"]
    assert "Imported 1 row(s), skipped 1" in log
    assert "youtube" in json.loads((project / "library.json").read_text())["alice-abc@70.0"]["posts"]
    with pytest.raises(ValueError, match=f"at least {fb.MIN_TRAIN}"):
        web.results.train(project)


def test_drawn_cam_layout_is_saved_for_the_video_and_the_streamer(project):
    out = web.save_folder_layout({"recipe": "clips/clips.json", "facecam": [0.0, 0.07, 0.27, 0.27],
                                  "cam_height": 0.4, "cam_position": "bottom"})
    assert out["streamer"] == "alice"
    assert json.loads((project / "clips/clips.json").read_text())["layout"]["cam_position"] == "bottom"
    saved = json.loads((project / "layouts.json").read_text())["alice"]
    assert saved["facecam"] == [0.0, 0.07, 0.27, 0.27] and saved["cam_height"] == 0.4
    lay = next(f for f in web.library() if f["name"] == "clips")["layout"]
    assert lay["from"] == "streamer" and lay["cam_position"] == "bottom"
    with pytest.raises(ValueError):
        web.save_folder_layout({"recipe": "clips/clips.json", "facecam": [0.9, 0.9, 0.5, 0.5]})  # off the frame
    with pytest.raises(ValueError):
        web.save_folder_layout({"recipe": "../x/clips.json", "facecam": [0, 0, 0.2, 0.2]})


def test_frame_needs_some_video_left(project):
    with pytest.raises(ValueError, match="no video left"):
        web.frame_for(project / "clips/clips.json")


def test_model_first_pass_and_transcription_reach_the_command():
    cmd, _ = web.build_command({"source": "video", "video": "v.mp4", "model": "claude-opus-5-5",
                                "first_pass": "claude-haiku-5-5", "whisper_model": "turbo"})
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--triage-model") + 1] == "claude-haiku-5-5"
    assert cmd[cmd.index("--whisper-model") + 1] == "turbo"
    cmd, _ = web.build_command({"source": "rerender", "clips_json": "c/clips.json", "model": "claude-opus-5-5"})
    assert "--model" not in cmd   # re-rendering doesn't call Claude
    with pytest.raises(ValueError):
        web.build_command({"source": "url", "url": "u", "model": "gpt-9"})
