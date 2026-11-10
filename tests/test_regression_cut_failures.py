"""Regression tests: one failed clip cut must not crash the whole run."""

from pathlib import Path

import pytest

V = Path("in.mp4")


def test_run_cut_jobs_keeps_going_after_a_failure(capsys):
    # One ffmpeg failure used to raise out of the pool and end the run with a traceback.
    import subprocess
    from clipfinder.render import run_cut_jobs

    def fake_cut(video, out, *rest):
        if out.name.startswith("03"):
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
    jobs = [(V, Path(f"0{i}_clip.mp4"), 0, 5, "copy", False, None) for i in range(1, 6)]
    saved, failed = run_cut_jobs(jobs, workers=2, cut=fake_cut)
    assert sorted(saved) == ["01_clip.mp4", "02_clip.mp4", "04_clip.mp4", "05_clip.mp4"]
    assert [name for name, _ in failed] == ["03_clip.mp4"] and "exit status 1" in failed[0][1]
    assert "FAILED 03_clip.mp4" in capsys.readouterr().out


def test_cli_reports_failed_clips_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    import json
    import clipfinder.cli as cli
    import clipfinder.render as render
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"video": str(video), "clips": [
        {"start": i * 20.0, "end": i * 20.0 + 10, "score": 8, "title": f"clip {i}", "hook": "h",
         "category": "funny", "reason": "r"} for i in range(3)]}))

    def fake_cut(video, out, *rest):
        if "clip_1" in out.name:
            raise RuntimeError("bad crop box")
    real = render.run_cut_jobs
    monkeypatch.setattr(render, "run_cut_jobs", lambda jobs, workers: real(jobs, workers, cut=fake_cut))
    monkeypatch.setattr(render, "keyframe_before", lambda video, t: t)
    monkeypatch.setattr(cli, "keyframe_before", lambda video, t: t, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)  # no ffmpeg needed
    with pytest.raises(SystemExit) as e:
        cli.main(["--clips-json", str(manifest), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert e.value.code == 1
    assert "2 of 3 clips saved" in out and "02_clip_1.mp4: bad crop box" in out
