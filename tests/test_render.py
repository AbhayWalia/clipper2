from pathlib import Path

from clipfinder.render import build_cut_cmd, choose_mode, pick_keyframe
from clipfinder.transcribe import extract_audio_cmd

V, O = Path("in.mp4"), Path("out.mp4")


def test_choose_mode():
    assert choose_mode(False, False, False, False) == "copy"
    assert choose_mode(True, False, False, False) == "encode"
    assert choose_mode(False, True, False, False) == "encode"
    assert choose_mode(False, False, False, True) == "encode"
    assert choose_mode(True, True, True, False) == "preview"


def test_copy_cmd_does_not_reencode():
    cmd = build_cut_cmd(V, O, 10, 40, "copy")
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-vf" not in cmd and "libx264" not in cmd


def graph(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_encode_vertical_full_res():
    cmd = build_cut_cmd(V, O, 10, 40, "encode", vertical=True, encoder="libx264")
    assert "crop=ih*9/16:ih,scale=1080:1920" in graph(cmd)
    assert cmd[cmd.index("-map") + 1] == "[v]"


def test_preview_is_480p():
    assert "scale=-2:480" in graph(build_cut_cmd(V, O, 0, 5, "preview"))
    assert "scale=270:480" in graph(build_cut_cmd(V, O, 0, 5, "preview", vertical=True))


def test_layout_and_ass_in_graph():
    from clipfinder.layout import Layout
    lay = Layout(facecam=(0.72, 0.03, 0.26, 0.30))
    g = graph(build_cut_cmd(V, O, 0, 5, "encode", ass=Path("/tmp/a b.ass"), layout=lay))
    assert "vstack=inputs=2" in g and "crop=w=iw*0.2600" in g
    assert g.endswith("subtitles='/tmp/a b.ass'[v]")


def test_pick_keyframe():
    assert pick_keyframe("8.008000\n10.010000\n12.012000\n", 11.5) == 10.01
    assert pick_keyframe("", 11.5) == 11.5
    assert pick_keyframe("20.0\n", 11.5) == 11.5


def test_extract_audio_is_16k_mono_no_video():
    cmd = extract_audio_cmd(V, Path("a.flac"))
    assert "-vn" in cmd and cmd[cmd.index("-ar") + 1] == "16000" and cmd[cmd.index("-ac") + 1] == "1"


def test_whisper_hallucination_filter():
    from clipfinder.transcribe import looks_hallucinated
    assert looks_hallucinated(0.85, -0.85)          # "you" over silence (measured)
    assert not looks_hallucinated(0.01, -0.32)      # real speech, incl. a shouted "BOOM!"
    assert not looks_hallucinated(0.9, -0.2)        # confident text: keep
    assert not looks_hallucinated(None, None)


# --- clip segments ----------------------------------------------------------

def test_segment_windows_pads_and_merges_overlaps():
    from clipfinder.render import segment_windows
    # Clips at 100-150 and 170-200 overlap once padded by 30s; 500-540 stays apart.
    wins = segment_windows([(500, 540), (100, 150), (170, 200)], 30)
    assert wins == [(70, 230), (470, 570)]


def test_segment_windows_clamped_to_video():
    from clipfinder.render import segment_windows
    assert segment_windows([(10, 40), (980, 995)], 30, video_len=1000) == [(0.0, 70), (950, 1000)]


def test_segment_for_prefers_covering_then_most_overlap():
    from clipfinder.render import segment_for
    segs = [{"file": "a", "start": 70, "end": 230}, {"file": "b", "start": 470, "end": 570}]
    assert segment_for(segs, 100, 150)["file"] == "a"
    assert segment_for(segs, 200, 260)["file"] == "a"   # nudged past the end: best overlap
    assert segment_for(segs, 300, 350) is None


def test_update_manifest_keeps_other_fields(tmp_path):
    from clipfinder.clips import Clip, load_manifest, save_manifest, update_manifest
    p = tmp_path / "clips.json"
    save_manifest(p, [Clip(1, 20, 8, "t", "h", "funny", "r")], video="v.mp4")
    update_manifest(p, segments=[{"file": "segments/seg_01.mp4", "start": 0.0, "end": 51.5}])
    data, clips = load_manifest(p)
    assert data["video"] == "v.mp4" and len(clips) == 1
    assert data["segments"][0]["file"] == "segments/seg_01.mp4"


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
