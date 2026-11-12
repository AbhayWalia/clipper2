"""Regression tests: --dry-run messages must match what actually happened."""


def _clips_json(tmp_path):
    import json
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    manifest = tmp_path / "old" / "clips.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"video": str(video), "clips": [
        {"start": 0, "end": 10, "score": 8, "title": "t", "hook": "h", "category": "funny",
         "reason": "r"}]}))
    return manifest


def test_dry_run_with_clips_json_points_at_the_real_file(tmp_path, monkeypatch, capsys):
    # Used to claim "clip list saved to new/clips.json", a file that was never written.
    import clipfinder.cli as cli
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    manifest = _clips_json(tmp_path)
    cli.main(["--clips-json", str(manifest), "--out", str(tmp_path / "new"), "--dry-run"])
    out = capsys.readouterr().out
    assert f"The clip list is {manifest}" in out and "saved to" not in out
    assert not (tmp_path / "new" / "clips.json").exists()


def test_dry_run_says_segments_were_skipped(tmp_path, monkeypatch, capsys):
    import clipfinder.cli as cli
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    manifest = _clips_json(tmp_path)
    cli.main(["--clips-json", str(manifest), "--dry-run", "--save-segments"])
    assert "--save-segments was skipped because this is a dry run" in capsys.readouterr().out
