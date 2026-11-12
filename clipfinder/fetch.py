"""Fetch a stream from a URL (Twitch VOD, YouTube, ...): the video, the platform's
own captions when it has them (free, so no Whisper needed), and the chat replay.

Uses yt-dlp for video/captions and chat-downloader for chat; both are optional
installs (`pip install yt-dlp chat-downloader`). Files already downloaded are
reused, so re-running on the same URL is cheap.

Audio first (`--audio-first`): download only the audio (tens of MB per hour instead of
GBs), find the clips from it, then download just the video around each clip with
`download_sections`. Section downloads start exactly at the requested time (ffmpeg
hides the frames before it), so their times line up with the audio's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CAPTION_LANGS = ["en", "en-orig", "en-US", "en-GB"]


SECTION_JOBS = 8  # YouTube sends each section slowly, but parallel ones don't slow each other
SECTION_PAD = 10.0  # extra footage either side of a clip; a later trim past it fetches more


@dataclass
class Fetched:
    video: Path | None
    captions: Path | None
    chat: Path | None
    title: str = ""
    uploader: str = ""
    duration: float | None = None
    audio: Path | None = None
    url: str = ""  # the page URL yt-dlp resolved, for downloading sections later


def video_format(max_height: int = 1080) -> str:
    h = max_height
    # Prefer mp4/m4a so the merge needs no re-encode; fall back to anything.
    return (f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
            f"bv*[height<={h}]+ba/b[height<={h}]/b")


def ytdlp_options(dest: Path, video: bool = True, captions: bool = True,
                  max_height: int = 1080, audio_only: bool = False) -> dict:
    return {
        "outtmpl": str(dest / "%(extractor_key)s_%(id)s.%(ext)s"),
        "format": "ba[ext=m4a]/ba/b" if audio_only else video_format(max_height),
        "merge_output_format": "mp4",
        "skip_download": not (video or audio_only),
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


def _download(yt_dlp, url: str, opts: dict) -> tuple[dict, Path]:
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info.get("_type") == "playlist":
            raise SystemExit("That URL is a playlist/channel; give a single VOD or video URL.")
        return info, Path(ydl.prepare_filename(info)).with_suffix("")


def fetch_source(url: str, dest: Path, video: bool = True, captions: bool = True,
                 chat: bool = True, max_height: int = 1080, audio_only: bool = False) -> Fetched:
    """With audio_only, download the audio track instead of the video."""
    video = video and not audio_only
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("--url needs yt-dlp:  pip install yt-dlp")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        info, stem = _download(yt_dlp, url, ytdlp_options(dest, video, captions, max_height,
                                                         audio_only))
    except yt_dlp.utils.DownloadError as e:
        # Captions are only a shortcut past Whisper. YouTube often rate-limits them
        # (HTTP 429), so retry without them rather than failing the whole run.
        if not captions or "subtitles" not in str(e):
            raise
        print(f"  captions download failed ({str(e).removeprefix('ERROR: ')}); "
              "continuing without them, Whisper will transcribe instead")
        info, stem = _download(yt_dlp, url, ytdlp_options(dest, video, False, max_height, audio_only))
    video_path = None
    if video:
        for ext in (".mp4", ".mkv", ".webm"):
            if stem.with_suffix(ext).exists():
                video_path = stem.with_suffix(ext)
                break
    audio_path = None
    if audio_only:
        for ext in (".m4a", ".webm", ".opus", ".mp3", ".mp4", ".aac"):
            if stem.with_suffix(ext).exists():
                audio_path = stem.with_suffix(ext)
                break
    chat_path = download_chat(url, stem.with_suffix(".chat.json")) if chat else None
    return Fetched(video=video_path, captions=_caption_file(info), chat=chat_path,
                   title=info.get("title") or "", uploader=info.get("uploader") or "",
                   duration=info.get("duration"), audio=audio_path,
                   url=info.get("webpage_url") or url)


def section_name(start: float, end: float) -> str:
    return f"segments/sec_{start:08.1f}-{end:08.1f}.mp4"


def download_sections(url: str, folder: Path, windows: list[tuple[float, float]],
                      max_height: int = 1080, jobs: int = SECTION_JOBS) -> list[dict]:
    """Download the video for each (start, end) window into folder/segments/.

    Returns segment entries for clips.json (paths relative to `folder`), the same shape
    `render.save_segments` makes, so the renderer can't tell them apart. A window that
    fails is reported and left out; clips in it are then skipped."""
    from concurrent.futures import ThreadPoolExecutor
    try:
        import yt_dlp
        from yt_dlp.utils import download_range_func
    except ImportError:
        raise SystemExit("Downloading clip sections needs yt-dlp:  pip install yt-dlp")
    (folder / "segments").mkdir(parents=True, exist_ok=True)

    def one(win: tuple[float, float]) -> dict | None:
        s, e = win
        out = folder / section_name(s, e)
        entry = {"file": section_name(s, e), "start": round(s, 3), "end": round(e, 3)}
        if out.exists():
            return entry
        opts = {"outtmpl": str(out.with_suffix("")) + ".%(ext)s",
                "format": video_format(max_height), "merge_output_format": "mp4",
                "download_ranges": download_range_func(None, [(s, e)]),
                "quiet": True, "no_warnings": True, "noprogress": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as ex:  # one bad section shouldn't sink the rest
            print(f"  section {s:.0f}-{e:.0f}s failed: {str(ex).removeprefix('ERROR: ')}")
            return None
        if not out.exists():  # the site only had a non-mp4 format; keep whatever arrived
            got = sorted(out.parent.glob(out.stem + ".*"))
            if not got:
                print(f"  section {s:.0f}-{e:.0f}s: download finished but no file was found")
                return None
            entry["file"] = f"segments/{got[0].name}"
        return entry

    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        return [seg for seg in pool.map(one, windows) if seg]


def streamer_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
