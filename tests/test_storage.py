"""Cloud storage against an in-memory S3 (moto). Skipped if boto3/moto aren't installed."""
import json

import pytest

pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from clipfinder.storage import Store, StorageError, load_dotenv, stream_name  # noqa: E402


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("CLIPPER_BUCKET", "clipper")
    monkeypatch.delenv("CLIPPER_S3_ENDPOINT", raising=False)
    with moto.mock_aws():
        s = Store.from_env()
        s.client.create_bucket(Bucket="clipper")
        yield s


def run_folder(tmp_path):
    out = tmp_path / "out"
    (out / "segments").mkdir(parents=True)
    (out / "01_a.mp4").write_bytes(b"clip")
    (out / "01_a.srt").write_text("1\n")
    (out / "segments" / "seg_01.mp4").write_bytes(b"segment")
    (out / "clips.json").write_text(json.dumps({"clips": []}))
    (out / "99_other_stream.mp4").write_bytes(b"not this run")
    t = tmp_path / "vod.transcript.json"
    t.write_text("{}")
    files = {out / "01_a.mp4": "01_a.mp4", out / "01_a.srt": "01_a.srt",
             out / "segments" / "seg_01.mp4": "segments/seg_01.mp4", out / "clips.json": "clips.json"}
    return out, t, files


def test_upload_lays_out_by_prefix_and_skips_other_files(store, tmp_path):
    out, t, files = run_folder(tmp_path)
    vod = tmp_path / "vod.mp4"
    vod.write_bytes(b"vod")
    store.upload_run("vod", files, out / "clips.json", t, vod)
    assert set(store.sizes("")) == {
        "clips/vod/01_a.mp4", "clips/vod/01_a.srt", "clips/vod/segments/seg_01.mp4",
        "clips/vod/clips.json", "recipes/vod/clips.json", "recipes/vod/vod.transcript.json",
        "vods/vod.mp4"}
    head = store.client.head_object(Bucket="clipper", Key="clips/vod/01_a.mp4")
    assert head["ContentType"] == "video/mp4"


def test_unchanged_segments_are_not_reuploaded(store, tmp_path):
    out, t, files = run_folder(tmp_path)
    store.upload_run("vod", files, out / "clips.json", t)
    again = store.upload_run("vod", files, out / "clips.json", t)
    assert "clips/vod/segments/seg_01.mp4" not in again
    assert "clips/vod/01_a.mp4" in again   # clips may have been re-rendered: always sent


def test_fetch_round_trip_brings_transcript(store, tmp_path):
    out, t, files = run_folder(tmp_path)
    store.upload_run("vod", files, out / "clips.json", t)
    dest = tmp_path / "fetched"
    store.fetch_run("vod", dest)
    assert (dest / "segments" / "seg_01.mp4").read_bytes() == b"segment"
    assert (dest / "clips.json").exists() and (dest / "vod.transcript.json").exists()
    with pytest.raises(StorageError):
        store.fetch_run("missing", dest)


def test_link_is_signed_and_capped_at_7_days(store):
    url = store.link("clips/vod/01_a.mp4", days=30)
    assert "X-Amz-Expires=604800" in url and "X-Amz-Signature" in url


def test_missing_bucket_setting(monkeypatch):
    monkeypatch.delenv("CLIPPER_BUCKET", raising=False)
    with pytest.raises(StorageError):
        Store.from_env()


def test_stream_name_and_dotenv(tmp_path, monkeypatch):
    assert stream_name("My Stream (part 2)") == "My-Stream-part-2"
    (tmp_path / ".env").write_text('# keys\nexport CLIPPER_BUCKET="b1"  # comment\nAWS_REGION=auto\n')
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.delenv("CLIPPER_BUCKET", raising=False)
    monkeypatch.setenv("AWS_REGION", "kept")
    load_dotenv(sub)
    import os
    assert os.environ["CLIPPER_BUCKET"] == "b1" and os.environ["AWS_REGION"] == "kept"
    monkeypatch.delenv("CLIPPER_BUCKET")
