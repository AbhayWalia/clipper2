"""Web UI helpers: command building, review/trim edits, prompt file, run stats."""

import json
import types

import pytest

from clipfinder import stats, web


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "PROMPT_FILE", tmp_path / "prompt.txt")
    (tmp_path / "clips").mkdir()
    recipe = tmp_path / "clips" / "clips.json"
    recipe.write_text(json.dumps({"clips": [
        {"start": 10.0, "end": 40.0, "score": 8, "title": "Big crash", "hook": "h",
         "category": "chaos", "reason": "r"},
        {"start": 100.0, "end": 130.0, "score": 7, "title": "Cop chase", "hook": "h",
         "category": "chaos", "reason": "r"},
    ]}))
    return tmp_path


def test_review_and_trim_are_saved(project):
    web.update_clip({"recipe": "clips/clips.json", "n": 2, "review": "approved"})
    out = web.update_clip({"recipe": "clips/clips.json", "n": 2, "nudge_end": -1})
    assert out == {"start": 100.0, "end": 129.0, "review": "approved"}
    clip = json.loads((project / "clips/clips.json").read_text())["clips"][1]
    assert clip["original_end"] == 130.0 and clip["edited_at"] > 0
    web.update_clip({"recipe": "clips/clips.json", "n": 2, "review": None})
    assert "review" not in json.loads((project / "clips/clips.json").read_text())["clips"][1]


@pytest.mark.parametrize("form", [
    {"recipe": "../elsewhere/clips.json", "n": 1, "review": "approved"},   # outside the project
    {"recipe": "clips/clips.json", "n": 3, "review": "approved"},          # no such clip
    {"recipe": "clips/clips.json", "n": 1, "review": "maybe"},
    {"recipe": "clips/clips.json", "n": 1, "nudge_start": 29.5},           # under 1 s left
    {"recipe": "clips/clips.json", "n": 1, "nudge_end": 100},              # too large
])
def test_bad_edits_are_refused(project, form):
    with pytest.raises(ValueError):
        web.update_clip(form)


def test_approved_clips_listed_and_matched_by_name(project):
    (project / "clips/02_cop_chase.mp4").write_bytes(b"v")
    (project / "clips/01_old_unrelated_video.mp4").write_bytes(b"v")   # from an older run
    web.update_clip({"recipe": "clips/clips.json", "n": 2, "review": "approved"})
    [folder] = web.library()
    by_file = {c["file"]: c for c in folder["clips"]}
    assert by_file["clips/02_cop_chase.mp4"]["matched"] is True
    assert by_file["clips/01_old_unrelated_video.mp4"]["matched"] is False
    assert folder["approved"] == [2] and folder["uncut"] == 1


def test_prompt_file_and_only_reach_the_command(project):
    cmd, _ = web.build_command({"source": "video", "video": "a.mp4"})
    assert "--prompt-file" not in cmd
    web.save_prompt({"text": "Find GTA 6 clips between {min_len} and {max_len} seconds. " * 2})
    cmd, _ = web.build_command({"source": "video", "video": "a.mp4"})
    assert cmd[cmd.index("--prompt-file") + 1] == "prompt.txt"
    cmd, _ = web.build_command({"source": "rerender", "clips_json": "clips/clips.json", "only": [3, 1]})
    assert cmd[cmd.index("--only") + 1] == "1,3" and "--prompt-file" not in cmd
    with pytest.raises(ValueError):
        web.build_command({"source": "rerender", "clips_json": "clips/clips.json", "only": ["1;x"]})
    web.save_prompt({"reset": True})
    assert not (project / "prompt.txt").exists()


def msg(model, inp, out, read=0, write=0):
    return types.SimpleNamespace(model=model, usage=types.SimpleNamespace(
        input_tokens=inp, output_tokens=out, cache_read_input_tokens=read,
        cache_creation_input_tokens=write))


def test_run_cost():
    s = stats.RunStats()
    s.add(msg("claude-sonnet-5-5", 1_000_000, 100_000))            # $2 + $1
    s.add(msg("claude-sonnet-5-5", 1_000_000, 0), batch=True)      # half of $2
    s.add(msg("claude-haiku-5-5", 0, 0, read=1_000_000, write=1_000_000))  # $0.01 + $0.125
    assert s.cost() == pytest.approx(4.135)
    assert "$4.13" in s.summary(10) or "$4.14" in s.summary(10)
    s.add(msg("some-future-model", 10, 10))
    assert s.cost() is None
    with s.stage("cut"):
        pass
    d = s.to_dict("find", 3)
    assert d["requests"] == 4 and "cut" in d["seconds"] and d["cost_usd"] is None
