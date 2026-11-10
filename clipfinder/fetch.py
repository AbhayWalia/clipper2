"""Fetch a stream from a URL (Twitch VOD, YouTube, ...): the video, the platform's
own captions when it has them (free, so no Whisper needed), and the chat replay.

Uses yt-dlp for video/captions and chat-downloader for chat; both are optional
installs (`pip install yt-dlp chat-downloader`). Files already downloaded are
reused, so re-running on the same URL is cheap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CAPTION_LANGS = ["en", "en-orig", "en-US", "en-GB"]


@dataclass
class Fetched:
    video: Path | None
    captions: Path | None
    chat: Path | None
    title: str = ""
    uploader: str = ""
    duration: float | None = None


def ytdlp_options(dest: Path, video: bool = True, captions: bool = True,
                  max_height: int = 1080) -> dict:
    h = max_height
    return {
        "outtmpl": str(dest / "%(extractor_key)s_%(id)s.%(ext)s"),
        # Prefer mp4/m4a so the merge needs no re-encode; fall back to anything.
        "format": (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
                   f"bv*[height<={h}]+ba/b[height<={h}]/b"),
        "merge_output_format": "mp4",
        "skip_download": not video,
        "writesubtitles": captions,          # creator-made captions first...
        "writeautomaticsub": captions,       # ...else the platform's automatic ones
        "subtitleslangs": CAPTION_LANGS,
        "subtitlesformat": "vtt/srt/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }


def _caption_file(info: dict) -> Path | None:
    subs = info.get("requested_subtitles") or {}
    for lang in CAPTION_LANGS + list(subs):
        p = (subs.get(lang) or {}).get("filepath")
        if p and Path(p).exists() and Path(p).suffix.lower() in (".vtt", ".srt"):
            return Path(p)
    return None


def download_chat(url: str, out: Path, progress_every: int = 5000) -> Path | None:
    """Chat replay as chat-downloader JSON ([{time_in_seconds, message}, ...])."""
    if out.exists():
        return out
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        print("  (chat skipped: pip install chat-downloader)")
        return None
    rows = []
    try:
        for m in ChatDownloader().get_chat(url):
            t = m.get("time_in_seconds")
            if t is None:
                continue
            rows.append({"time_in_seconds": float(t), "message": m.get("message", "")})
            if progress_every and len(rows) % progress_every == 0:
                print(f"  ...{len(rows):,} chat messages")
    except Exception as e:  # chat is a bonus; never block clipping on it
        print(f"  chat download failed ({e}); continuing without chat")
        return None
    if not rows:
        print("  no chat replay found")
        return None
    out.write_text(json.dumps(rows), encoding="utf-8")
    return out


def fetch_source(url: str, dest: Path, video: bool = True, captions: bool = True,
                 chat: bool = True, max_height: int = 1080) -> Fetched:
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("--url needs yt-dlp:  pip install yt-dlp")
    dest.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL(ytdlp_options(dest, video, captions, max_height)) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":
            raise SystemExit("That URL is a playlist/channel; give a single VOD or video URL.")
        stem = Path(ydl.prepare_filename(info)).with_suffix("")
    video_path = None
    if video:
        for ext in (".mp4", ".mkv", ".webm"):
            if stem.with_suffix(ext).exists():
                video_path = stem.with_suffix(ext)
                break
    chat_path = download_chat(url, stem.with_suffix(".chat.json")) if chat else None
    return Fetched(video=video_path, captions=_caption_file(info), chat=chat_path,
                   title=info.get("title") or "", uploader=info.get("uploader") or "",
                   duration=info.get("duration"))


def streamer_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
