"""Per-user folders in the bucket and recording runs in Supabase. No network: the bucket and
Supabase's REST API are faked."""

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

import clipfinder.cli as cli
import clipfinder.render as render
from clipfinder.db import Database, DatabaseError
from clipfinder.storage import StorageError, new_run_name, run_path, user_id

USER = "3f1c9a2e-7b4d-4e8a-9c1f-0a2b3c4d5e6f"


# -- naming ----------------------------------------------------------------------------

def test_run_names_are_unique_and_safe():
    a, b = new_run_name("My Stream (part 2)"), new_run_name("My Stream (part 2)")
    assert a != b
    assert re.fullmatch(r"My-Stream-part-2-[0-9a-f]{6}", a)


def test_user_ids_must_be_one_safe_path_segment():
    assert user_id(USER) == USER
    for bad in ("", "../other", "a/b", "x" * 65, "name with spaces"):
        with pytest.raises(StorageError):
            user_id(bad)
    assert run_path(USER, "vod-abc123") == f"{USER}/vod-abc123"
    assert run_path(None, "vod-abc123") == "vod-abc123"


# -- the REST calls ----------------------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_record_run_upserts_the_run_before_its_clips(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout, context):
        sent.append(req)
        return FakeResponse(b"")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    db = Database("https://proj.supabase.co/", "service-key")
    db.record_run({"id": f"{USER}/vod-1"}, [{"run_id": f"{USER}/vod-1", "idx": 1}])

    assert [r.full_url for r in sent] == [
        "https://proj.supabase.co/rest/v1/runs?on_conflict=id",
        "https://proj.supabase.co/rest/v1/clips?on_conflict=run_id,idx"]
    req = sent[0]
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer service-key"
    assert req.get_header("Apikey") == "service-key"
    assert "merge-duplicates" in req.get_header("Prefer")
    assert json.loads(req.data) == [{"id": f"{USER}/vod-1"}]


def test_project_url_may_include_rest_v1():
    # The dashboard shows the API URL as ".../rest/v1/"; that used to make "/rest/v1/rest/v1".
    for given in ("https://p.supabase.co", "https://p.supabase.co/",
                  "https://p.supabase.co/rest/v1", "https://p.supabase.co/rest/v1/"):
        assert Database(given, "k").url == "https://p.supabase.co"


def test_no_clips_means_no_clips_request(monkeypatch):
    sent = []
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout, context: sent.append(req) or FakeResponse(b""))
    Database("https://p.supabase.co", "k").record_run({"id": "u/r"}, [])
    assert len(sent) == 1


def test_a_refused_write_explains_why(monkeypatch):
    def refuse(req, timeout, context):
        raise urllib.error.HTTPError(req.full_url, 409, "Conflict", {},
                                     io.BytesIO(b'{"message":"violates foreign key"}'))
    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(DatabaseError, match="409.*violates foreign key"):
        Database("https://p.supabase.co", "k").upsert("runs", [{"id": "x"}], "id")


def test_database_is_optional(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert Database.from_env() is None
    monkeypatch.setenv("SUPABASE_URL", "https://p.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    assert Database.from_env().url == "https://p.supabase.co"


# -- the whole --upload --user run ---------------------------------------------------------

class FakeStore:
    def __init__(self):
        self.uploads, self.fetches = [], []

    def upload_run(self, stream, files, recipe, transcript, video, jobs):
        self.uploads.append((stream, files))
        return [f"clips/{stream}/{rel}" for rel in files.values()]

    def fetch_run(self, stream, dest, jobs):
        self.fetches.append((stream, dest))
        return []

    def sizes(self, prefix):
        return {}


class FakeDatabase:
    def __init__(self, fail=False):
        self.runs, self.fail = [], fail

    def record_run(self, run, clips):
        if self.fail:
            raise DatabaseError("Couldn't reach Supabase")
        self.runs.append((run, clips))


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """A clips.json with three clips, ffmpeg faked to write each clip, a fake bucket and database."""
    video = tmp_path / "Big Heist.mp4"
    video.write_bytes(b"x")
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"video": str(video), "runs": [{"cost_usd": 0.05}], "clips": [
        {"start": i * 20.0, "end": i * 20.0 + 10, "score": 8, "title": f"clip {i}", "hook": "h",
         "category": "funny", "reason": "r", **({"review": "approved"} if i == 0 else {})}
        for i in range(3)]}))

    def fake_cut(video, out, *rest):
        out.write_bytes(b"0123456789")
    real = render.run_cut_jobs
    monkeypatch.setattr(render, "run_cut_jobs", lambda jobs, workers: real(jobs, workers, cut=fake_cut))
    monkeypatch.setattr(cli, "keyframe_before", lambda video, t: t, raising=False)
    monkeypatch.setattr(cli, "video_duration", lambda video: 900.0)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.delenv("CLIPPER_USER", raising=False)
    store, db = FakeStore(), FakeDatabase()
    monkeypatch.setattr(cli.Store, "from_env", classmethod(lambda cls: store))
    monkeypatch.setattr(cli.Database, "from_env", classmethod(lambda cls: db))
    return {"manifest": manifest, "out": tmp_path / "out", "store": store, "db": db}


def run(env, *extra):
    cli.main(["--clips-json", str(env["manifest"]), "--out", str(env["out"]), "--upload", *extra])


def test_upload_goes_to_the_users_folder_and_is_recorded(run_env, capsys):
    run(run_env, "--user", USER)
    path, files = run_env["store"].uploads[0]
    assert re.fullmatch(rf"{USER}/Big-Heist-[0-9a-f]{{6}}", path)

    [(run_row, clip_rows)] = run_env["db"].runs
    assert run_row["id"] == path and run_row["user_id"] == USER
    assert run_row["video_name"] == "Big Heist.mp4" and run_row["video_minutes"] == 15.0
    assert run_row["clip_count"] == 3 and run_row["cost_usd"] == 0.05
    assert [c["idx"] for c in clip_rows] == [1, 2, 3]
    assert clip_rows[0]["file_key"] == f"clips/{path}/01_clip_0.mp4"
    assert clip_rows[0]["review"] == "approved" and clip_rows[1]["review"] is None
    assert clip_rows[0]["size_bytes"] == 10 and clip_rows[0]["run_id"] == path
    assert "recorded run" in capsys.readouterr().out

    # The run name and user are saved, so a re-render uploads to the same place.
    saved = json.loads(run_env["manifest"].read_text())
    assert saved["user"] == USER and path.endswith(saved["stream"])
    run(run_env)
    assert run_env["store"].uploads[1][0] == path


def test_user_can_come_from_env(run_env, monkeypatch):
    monkeypatch.setenv("CLIPPER_USER", USER)
    run(run_env)
    assert run_env["store"].uploads[0][0].startswith(f"{USER}/")


def test_without_a_user_nothing_is_recorded(run_env):
    run(run_env)
    path = run_env["store"].uploads[0][0]
    assert "/" not in path and run_env["db"].runs == []


def test_upload_still_succeeds_without_supabase_settings(run_env, monkeypatch, capsys):
    monkeypatch.setattr(cli.Database, "from_env", classmethod(lambda cls: None))
    run(run_env, "--user", USER)
    assert run_env["store"].uploads
    assert "not recorded in Supabase" in capsys.readouterr().out


def test_a_database_error_doesnt_fail_the_uploaded_run(run_env, monkeypatch, capsys):
    monkeypatch.setattr(cli.Database, "from_env", classmethod(lambda cls: FakeDatabase(fail=True)))
    run(run_env, "--user", USER)  # no SystemExit
    assert run_env["store"].uploads
    assert "uploaded, but not recorded in Supabase" in capsys.readouterr().out


def test_a_bad_user_id_stops_before_uploading(run_env):
    with pytest.raises(SystemExit, match="isn't a valid user id"):
        run(run_env, "--user", "../someone-else")
    assert run_env["store"].uploads == []


def test_fetch_reads_from_the_users_folder(run_env, tmp_path):
    cli.main(["--fetch", "Big-Heist-abc123", "--user", USER, "--out", str(tmp_path / "f")])
    assert run_env["store"].fetches == [(f"{USER}/Big-Heist-abc123", tmp_path / "f" / "Big-Heist-abc123")]
