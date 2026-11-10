"""Cutting clips with ffmpeg: stream copy, full re-encode, or low-res previews."""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

from .transcript import Segment

PREVIEW_HEIGHT = 480

LIBASS_HELP = """Burning in captions or a hook needs an ffmpeg built with libass (the
"subtitles" filter). Homebrew's standard ffmpeg doesn't include it. Fix either way:
  pip install imageio-ffmpeg          (bundled ffmpeg with libass; used automatically)
  brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass"""


@functools.cache
def has_filter(binary: str, name: str) -> bool:
    try:
        out = subprocess.run([binary, "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
    except OSError:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


@functools.cache
def ffmpeg_for(need_libass: bool) -> str:
    """The ffmpeg to run: the one on PATH, unless text has to be burned in and it
    lacks libass, in which case the bundled imageio-ffmpeg binary (which has it)."""
    if not need_libass or has_filter("ffmpeg", "subtitles"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if has_filter(exe, "subtitles"):
            return exe
    except Exception:
        pass
    raise RuntimeError(LIBASS_HELP)


def video_duration(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def keyframe_before(video: Path, t: float, search: float = 15.0) -> float:
    """Time of the last video keyframe at or before t (t itself if none found).

    Stream copy can only start on a keyframe, so we start the cut there and
    time the captions from the same point to keep them in sync.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
             "-read_intervals", f"{max(t - search, 0):.3f}%{t + 0.05:.3f}",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True).stdout
    except Exception:
        return t
    return pick_keyframe(out, t)


def pick_keyframe(ffprobe_csv: str, t: float) -> float:
    times = []
    for line in ffprobe_csv.split():
        try:
            times.append(float(line.strip(",")))
        except ValueError:
            pass
    before = [k for k in times if k <= t + 1e-3]
    return max(before) if before else t


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:n] or "clip"


def fmt_srt_time(t: float) -> str:
    # Round to whole milliseconds first, then split, so 59.9996 carries to 00:01:00,000
    # instead of wrapping the milliseconds and jumping back to 00:00:59,000.
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_clip_srt(path: Path, segs: list[Segment], t0: float, t1: float) -> None:
    """Captions for the cut range [t0, t1] (padding included), re-timed to start at 0."""
    lines, n = [], 1
    for s in segs:
        if s.end <= t0 or s.start >= t1:
            continue
        a, b = max(s.start, t0) - t0, min(s.end, t1) - t0
        lines.append(f"{n}\n{fmt_srt_time(a)} --> {fmt_srt_time(b)}\n{s.text}\n")
        n += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def pick_encoder() -> str:
    """Use the Mac's hardware H.264 encoder when ffmpeg has it."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return "h264_videotoolbox" if "h264_videotoolbox" in out else "libx264"


ENCODER_ARGS = {
    "h264_videotoolbox": ["-c:v", "h264_videotoolbox", "-q:v", "65"],
    "libx264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"],
}
PREVIEW_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"]

VERTICAL_SIZE = (1080, 1920)
PREVIEW_SIZE = (270, 480)


def choose_mode(vertical: bool, overlay: bool, preview: bool, reencode: bool) -> str:
    """'copy' when nothing needs re-encoding (almost free), else 'encode'/'preview'.
    `overlay` = anything burned in (captions, hook)."""
    if preview:
        return "preview"
    if vertical or overlay or reencode:
        return "encode"
    return "copy"


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def video_graph(mode: str, vertical: bool, layout=None, ass: Path | None = None) -> str:
    """filter_complex producing [v]: crop/stack to 9:16, scale, burn in the ASS file."""
    if vertical or layout is not None:
        w, h = PREVIEW_SIZE if mode == "preview" else VERTICAL_SIZE
        if layout is not None:
            from .layout import split_filter
            chain = split_filter(layout, w, h, out="base")
        else:
            chain = f"[0:v]crop=ih*9/16:ih,scale={w}:{h},setsar=1[base]"
    elif mode == "preview":
        chain = "[0:v]scale=-2:480,setsar=1[base]"
    else:
        chain = "[0:v]null[base]"
    if ass:
        return chain + f";[base]subtitles='{escape_filter_path(ass)}'[v]"
    return chain + ";[base]null[v]"


def build_cut_cmd(video: Path, out: Path, start: float, end: float, mode: str,
                  vertical: bool = False, ass: Path | None = None,
                  encoder: str = "libx264", layout=None) -> list[str]:
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}"]
    if mode == "copy":
        return cmd + ["-c", "copy", "-avoid_negative_ts", "make_zero",
                      "-movflags", "+faststart", str(out)]
    cmd += ["-filter_complex", video_graph(mode, vertical, layout, ass),
            "-map", "[v]", "-map", "0:a?"]
    cmd += PREVIEW_ARGS if mode == "preview" else ENCODER_ARGS[encoder]
    abitrate = "96k" if mode == "preview" else "160k"
    return cmd + ["-c:a", "aac", "-b:a", abitrate, "-movflags", "+faststart", str(out)]


def cut_clip(video: Path, out: Path, start: float, end: float, mode: str,
             vertical: bool, ass: Path | None, encoder: str = "libx264", layout=None) -> None:
    cmd = build_cut_cmd(video, out, start, end, mode, vertical, ass, encoder, layout)
    cmd[0] = ffmpeg_for(ass is not None)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if mode == "copy":  # e.g. a container/codec that can't be stream-copied to .mp4
            print(f"  stream copy failed on {out.name}, re-encoding instead")
            return cut_clip(video, out, start, end, "encode", vertical, ass, encoder, layout)
        if mode != "encode" or encoder == "libx264":
            raise
        print(f"  {encoder} failed on {out.name}, retrying with libx264")
        cut_clip(video, out, start, end, mode, vertical, ass, "libx264", layout)


# --------------------------------------------------------------------------- #
# Clip segments: keep the footage around each clip instead of the whole VOD.
# --------------------------------------------------------------------------- #

SEGMENT_PAD = 30.0


def segment_windows(spans: list[tuple[float, float]], pad: float,
                    video_len: float | None = None) -> list[tuple[float, float]]:
    """Pad each clip span by `pad` seconds on both sides and merge the ones that
    overlap, so footage shared by nearby clips is only stored once."""
    wins = sorted((max(s - pad, 0.0), min(e + pad, video_len) if video_len else e + pad)
                  for s, e in spans)
    merged: list[tuple[float, float]] = []
    for s, e in wins:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def segment_for(segments: list[dict], start: float, end: float) -> dict | None:
    """The saved segment covering [start, end]; if none covers it all (an edge was
    nudged too far), the one overlapping it most. None if nothing overlaps."""
    best, best_overlap = None, 0.0
    for seg in segments:
        if seg["start"] <= start + 1e-3 and end <= seg["end"] + 1e-3:
            return seg
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best, best_overlap = seg, overlap
    return best


def save_segments(video: Path, folder: Path, windows: list[tuple[float, float]],
                  jobs: int = 4) -> list[dict]:
    """Stream-copy each window into folder/segments/ (no re-encode, so it's fast and
    keeps full quality). Returns the manifest entries, with paths relative to `folder`
    and `start` moved back to the keyframe the copy really starts on."""
    from concurrent.futures import ThreadPoolExecutor
    (folder / "segments").mkdir(parents=True, exist_ok=True)
    segments = []
    for i, (s, e) in enumerate(windows, 1):
        segments.append({"file": f"segments/seg_{i:02d}.mp4",
                         "start": round(keyframe_before(video, s), 3), "end": round(e, 3)})
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        for f in [pool.submit(cut_clip, video, folder / seg["file"], seg["start"], seg["end"],
                              "copy", False, None) for seg in segments]:
            f.result()
    return segments


def video_size(video: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
            capture_output=True, text=True, check=True).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return None
