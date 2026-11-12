"""Cloud storage: upload a run's output to S3 or Cloudflare R2, and fetch it back.

R2 speaks the S3 API, so one client covers both; R2 only needs an endpoint. Settings come
from the environment (or a .env file):

    CLIPPER_BUCKET         bucket name (required)
    CLIPPER_S3_ENDPOINT    https://ACCOUNT_ID.r2.cloudflarestorage.com for R2; empty for AWS S3
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    AWS_REGION             "auto" for R2 (the default when an endpoint is set), e.g. us-east-1 for S3

Bucket layout, which storage/lifecycle.json expires by prefix:

    vods/<stream>.mp4        source video                      deleted after 1 day
    clips/<stream>/...       the output folder (clips, srt,     deleted after 30 days
                             previews/, segments/, clips.json)
    recipes/<stream>/...     clips.json + transcript           kept
"""

from __future__ import annotations

import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

MAX_LINK_DAYS = 7  # the longest a signed link can last, on S3 and R2 alike

CONTENT_TYPES = {".mp4": "video/mp4", ".srt": "application/x-subrip", ".ass": "text/plain",
                 ".json": "application/json", ".flac": "audio/flac", ".png": "image/png"}


class StorageError(RuntimeError):
    pass


def load_dotenv(start: Path | None = None) -> None:
    """Read KEY=value lines from the nearest .env (this folder or a parent) into the
    environment, without overriding anything already set."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        env = folder / ".env"
        if env.is_file():
            break
    else:
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass
class Store:
    client: object
    bucket: str

    @classmethod
    def from_env(cls) -> "Store":
        bucket = os.environ.get("CLIPPER_BUCKET")
        if not bucket:
            raise StorageError("Set CLIPPER_BUCKET (and CLIPPER_S3_ENDPOINT for R2) in your "
                               "environment or .env. See the README's storage section.")
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise StorageError("Install boto3 for cloud storage:  pip install 'boto3>=1.36'")
        endpoint = os.environ.get("CLIPPER_S3_ENDPOINT") or None
        region = os.environ.get("AWS_REGION") or ("auto" if endpoint else None)
        # Newer boto3 adds integrity checksums to every request, which R2 has rejected
        # at times; sending them only when an operation requires it works on both.
        # SigV4 always: boto3 still signs links with SigV2 in some regions, which R2 rejects.
        config = Config(signature_version="s3v4",
                        request_checksum_calculation="when_required",
                        response_checksum_validation="when_required",
                        retries={"max_attempts": 5, "mode": "standard"})
        return cls(boto3.client("s3", endpoint_url=endpoint, region_name=region, config=config),
                   bucket)

    # -- basics ------------------------------------------------------------

    def put(self, path: Path, key: str) -> None:
        ctype = CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
        self.client.upload_file(str(path), self.bucket, key,
                                ExtraArgs={"ContentType": ctype or "application/octet-stream"})

    def sizes(self, prefix: str) -> dict[str, int]:
        """Every key under `prefix` with its size."""
        out = {}
        for page in self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=prefix):
            out.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
        return out

    def link(self, key: str, days: float = MAX_LINK_DAYS) -> str:
        seconds = int(min(days, MAX_LINK_DAYS) * 86400)
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=seconds)

    # -- a run's output ----------------------------------------------------

    def upload_run(self, stream: str, files: dict[Path, str], recipe: Path | None = None,
                   transcript: Path | None = None, video: Path | None = None,
                   jobs: int = 4) -> list[str]:
        """Upload `files` ({local path: path inside the run folder}) to clips/<stream>/,
        clips.json and the transcript to recipes/<stream>/, and the source video to vods/.
        Segments already in the bucket at the same size are skipped (they're copies of the
        source and never change), which also keeps their 30-day clock from the first upload."""
        prefix = f"clips/{stream}/"
        existing = self.sizes(prefix + "segments/")
        uploads: list[tuple[Path, str]] = []
        for path, rel in sorted(files.items()):
            if not path.is_file():
                continue
            key = prefix + rel
            if existing.get(key) == path.stat().st_size:
                continue
            uploads.append((path, key))
        if recipe and recipe.exists():
            uploads.append((recipe, f"recipes/{stream}/clips.json"))
        if transcript and transcript.exists():
            uploads.append((transcript, f"recipes/{stream}/{transcript.name}"))
        if video and video.exists():
            uploads.append((video, f"vods/{stream}{video.suffix.lower()}"))

        with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
            for f in [pool.submit(self.put, p, k) for p, k in uploads]:
                f.result()
        return [k for _, k in uploads]

    def fetch_run(self, stream: str, dest: Path, jobs: int = 4) -> list[Path]:
        """Download clips/<stream>/ into dest, plus the transcript from recipes/<stream>/ so
        captions can be burned in again. Returns the files written."""
        prefix = f"clips/{stream}/"
        keys = {k: dest / k[len(prefix):] for k in self.sizes(prefix) if not k.endswith("/")}
        if not keys:
            raise StorageError(f"Nothing under {prefix} in {self.bucket}. It may have expired "
                               f"(clips are kept 30 days), or the stream name is different.")
        for key in self.sizes(f"recipes/{stream}/"):
            name = key.rsplit("/", 1)[-1]
            if name != "clips.json" and not (dest / name).exists():
                keys[key] = dest / name

        def get(key: str, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, key, str(path))

        with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
            for f in [pool.submit(get, k, p) for k, p in keys.items()]:
                f.result()
        return sorted(keys.values())


def stream_name(text: str) -> str:
    """A safe key prefix from a file stem or user input: letters, digits, - and _."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    if not name:
        raise StorageError(f"Can't make a stream name from {text!r}; pass --stream NAME.")
    return name
