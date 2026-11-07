#!/usr/bin/env python3
"""
GTA 6 transcript clipper.

Give it a video and its transcript. It asks Claude to find the best clip-worthy
moments in the transcript, then cuts those moments out of the video with ffmpeg.

Usage:
    python clipper.py --video stream.mp4 --transcript stream.srt
    python clipper.py --video stream.mp4 --transcript stream.srt --vertical --top 10
    python clipper.py --video stream.mp4 --transcribe          # make the transcript first
    python clipper.py --video stream.mp4 --transcript stream.srt --dry-run   # find clips, don't cut

Supported transcript formats: .srt, .vtt, Whisper .json, or .txt lines like
"[01:23:45] text" / "[83.2 - 87.9] text".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"


# --------------------------------------------------------------------------- #
# Transcript loading
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    start: float
    end: float
    text: str


def _ts_to_seconds(ts: str) -> float:
    """'01:02:03,456' / '01:02:03.456' / '02:03' / '123.4' -> seconds."""
    ts = ts.strip().replace(",", ".")
    if ":" not in ts:
        return float(ts)
    parts = [float(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


_CUE_RE = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})"
)


def _load_srt_vtt(text: str) -> list[Segment]:
    segs: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _CUE_RE.search(line)
            if m:
                body = " ".join(l.strip() for l in lines[i + 1:] if l.strip())
                body = re.sub(r"<[^>]+>", "", body)  # strip vtt/html tags
                if body:
                    segs.append(Segment(_ts_to_seconds(m.group(1)),
                                        _ts_to_seconds(m.group(2)), body))
                break
    return segs


def _load_json(text: str) -> list[Segment]:
    data = json.loads(text)
    items = data.get("segments", data) if isinstance(data, dict) else data
    return [Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip())
            for s in items if str(s.get("text", "")).strip()]


_TXT_RE = re.compile(
    r"^\s*\[\s*([\d:.,]+)\s*(?:-|–|-->)?\s*([\d:.,]+)?\s*\]\s*(.+)$"
)


def _load_txt(text: str) -> list[Segment]:
    raw = []
    for line in text.splitlines():
        m = _TXT_RE.match(line)
        if m:
            start = _ts_to_seconds(m.group(1))
            end = _ts_to_seconds(m.group(2)) if m.group(2) else None
            raw.append([start, end, m.group(3).strip()])
    if not raw:
        raise ValueError("No timestamped lines found. Use lines like '[00:01:23] text'.")
    for i, r in enumerate(raw):  # fill missing end times from the next line
        if r[1] is None:
            r[1] = raw[i + 1][0] if i + 1 < len(raw) else r[0] + 5
    return [Segment(*r) for r in raw]


def load_transcript(path: Path) -> list[Segment]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext in (".srt", ".vtt"):
        segs = _load_srt_vtt(text)
    elif ext == ".json":
        segs = _load_json(text)
    else:
        segs = _load_txt(text)
    if not segs:
        raise ValueError(f"Couldn't read any lines from {path}")
    return sorted(segs, key=lambda s: s.start)


MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _transcribe_mlx(video: Path, model_size: str) -> list[dict]:
    """Whisper on the Apple Silicon GPU."""
    import mlx_whisper
    repo = MLX_MODELS.get(model_size, model_size)
    result = mlx_whisper.transcribe(str(video), path_or_hf_repo=repo, verbose=False,
                                    condition_on_previous_text=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result["segments"] if s["text"].strip()]


def _transcribe_cpu(video: Path, model_size: str) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("--transcribe needs mlx-whisper (Apple Silicon) or faster-whisper:\n"
                 "  pip install mlx-whisper   or   pip install faster-whisper")
    model = WhisperModel(model_size, compute_type="auto")
    segments, _ = model.transcribe(str(video), vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def transcribe(video: Path, model_size: str = "small") -> Path:
    """Make a Whisper JSON transcript next to the video (GPU if available)."""
    use_mlx = importlib.util.find_spec("mlx_whisper") is not None
    print(f"Transcribing {video.name} with Whisper '{model_size}' "
          f"on the {'GPU' if use_mlx else 'CPU'}...")
    out = _transcribe_mlx(video, model_size) if use_mlx else _transcribe_cpu(video, model_size)
    path = video.with_suffix(".transcript.json")
    path.write_text(json.dumps({"segments": out}, indent=1), encoding="utf-8")
    print(f"Saved transcript -> {path}")
    return path


# --------------------------------------------------------------------------- #
# Finding clips with Claude
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an expert short-form video editor who specialises in \
GTA 6 gaming content for TikTok, YouTube Shorts and Instagram Reels.

You will get part of a transcript from a GTA 6 stream or gameplay video. Each line \
is "[start-end] text" with times in seconds. Find the moments that would make the \
best standalone short clips.

What makes a great GTA clip:
- Chaos and escalation: police chases, wanted level climbing, explosions, crashes, \
heists going wrong, getting "wasted" or "busted" at the worst moment.
- Big streamer reactions: screaming, laughing, disbelief, rage, "NO WAY", "bro what".
- Funny moments: absurd physics, NPC weirdness, failed stunts, banter, roleplay bits.
- Skill/clutch moments: insane escapes, stunt jumps, perfect getaways.
- Story/meme moments: shocking plot beats, quotable lines, hot takes about the game.

Rules for each clip:
- It must make sense on its own, with a brief setup and a payoff. Start a few \
seconds before the action begins; end right after the payoff/reaction lands.
- Usually 15 to 60 seconds; never under {min_len}s or over {max_len}s.
- Use the transcript's own timestamps for start and end.
- Clips must not overlap each other.
- Score honestly from 1 to 10 for viral potential. Skip dull stretches entirely; \
returning zero clips for a boring section is fine.
- Remember the transcript only captures speech. Infer game action from what the \
player says (e.g. "they're shooting at me", "five stars!", sudden yelling)."""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Start time in seconds"},
                    "end": {"type": "number", "description": "End time in seconds"},
                    "score": {"type": "number", "description": "Viral potential 1-10"},
                    "title": {"type": "string", "description": "Catchy short title, under 60 chars"},
                    "hook": {"type": "string", "description": "On-screen hook text for the first 2 seconds"},
                    "category": {"type": "string", "enum": ["chaos", "funny", "reaction", "skill", "story", "other"]},
                    "reason": {"type": "string", "description": "One sentence on why this works"},
                },
                "required": ["start", "end", "score", "title", "hook", "category", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


@dataclass
class Clip:
    start: float
    end: float
    score: float
    title: str
    hook: str
    category: str
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def chunk_segments(segs: list[Segment], window: float, overlap: float) -> list[list[Segment]]:
    """Split the transcript into overlapping time windows so long streams fit."""
    chunks, t, last = [], segs[0].start, segs[-1].end
    while t < last:
        chunk = [s for s in segs if s.end > t and s.start < t + window]
        if chunk:
            chunks.append(chunk)
        t += window - overlap
    return chunks


def format_chunk(chunk: list[Segment]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in chunk)


def find_clips_in_chunk(client, model: str, chunk: list[Segment],
                        min_len: float, max_len: float, context: str) -> list[Clip]:
    user = ""
    if context:
        user += f"Context about this video: {context}\n\n"
    user += ("Transcript section:\n\n" + format_chunk(chunk) +
             "\n\nFind the best clips in this section.")
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT.format(min_len=int(min_len), max_len=int(max_len)),
        output_config={"format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        print(f"  Warning: skipped this section (stop reason: {resp.stop_reason})")
        return []
    text = next((b.text for b in resp.content if b.type == "text"), "")
    if not text:
        return []
    return [Clip(**{k: c[k] for k in Clip.__dataclass_fields__})
            for c in json.loads(text)["clips"]]


# --------------------------------------------------------------------------- #
# Cleaning up the clip list
# --------------------------------------------------------------------------- #

def snap_to_segments(clip: Clip, segs: list[Segment]) -> Clip:
    """Move start/end to real sentence boundaries so clips don't cut mid-word."""
    starts = [s.start for s in segs]
    ends = [s.end for s in segs]
    clip.start = min(starts, key=lambda x: abs(x - clip.start))
    clip.end = min(ends, key=lambda x: abs(x - clip.end))
    return clip


def clean_clips(clips: list[Clip], segs: list[Segment], min_len: float,
                max_len: float, min_score: float, top: int,
                video_len: float | None) -> list[Clip]:
    out = []
    for c in clips:
        if c.end <= c.start:
            continue
        c = snap_to_segments(c, segs)
        if c.duration > max_len:
            c.end = c.start + max_len
        if c.duration < min_len:
            c.end = c.start + min_len
        if video_len:
            c.end = min(c.end, video_len)
        if c.score >= min_score and c.duration > 0:
            out.append(c)

    # Remove overlaps (from chunk overlap or the model): keep the higher score.
    out.sort(key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for c in out:
        if all(c.end <= k.start or c.start >= k.end for k in kept):
            kept.append(c)
    kept = kept[:top] if top else kept
    return sorted(kept, key=lambda c: c.start)


# --------------------------------------------------------------------------- #
# Cutting the video
# --------------------------------------------------------------------------- #

def video_duration(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:n] or "clip"


def fmt_srt_time(t: float) -> str:
    t = max(t, 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"


def write_clip_srt(path: Path, clip: Clip, segs: list[Segment], pad_before: float) -> None:
    """Captions for just this clip, re-timed to start at 0."""
    t0 = clip.start - pad_before
    lines, n = [], 1
    for s in segs:
        if s.end <= t0 or s.start >= clip.end:
            continue
        a, b = max(s.start, t0) - t0, min(s.end, clip.end) - t0
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


def cut_clip(video: Path, out: Path, start: float, end: float,
             vertical: bool, captions: Path | None, encoder: str = "libx264") -> None:
    filters = []
    if vertical:
        # Center-crop to 9:16, then scale to 1080x1920.
        filters.append("crop=ih*9/16:ih,scale=1080:1920")
    if captions:
        style = ("FontName=Arial,FontSize=14,Bold=1,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=60")
        sub_path = str(captions).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        filters.append(f"subtitles='{sub_path}':force_style='{style}'")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ENCODER_ARGS[encoder]
    cmd += ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if encoder == "libx264":
            raise
        print(f"  {encoder} failed on {out.name}, retrying with libx264")
        cut_clip(video, out, start, end, vertical, captions, "libx264")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Find and cut GTA 6 clips from a transcript.")
    p.add_argument("--video", required=True, type=Path, help="Source video file")
    p.add_argument("--transcript", type=Path, help=".srt / .vtt / .json / .txt transcript")
    p.add_argument("--transcribe", action="store_true", help="Make a transcript with Whisper first")
    p.add_argument("--whisper-model", default="small", help="Whisper size for --transcribe")
    p.add_argument("--out", type=Path, default=Path("clips"), help="Output folder")
    p.add_argument("--model", default=os.environ.get("CLIPPER_MODEL", DEFAULT_MODEL))
    p.add_argument("--top", type=int, default=15, help="Max number of clips (0 = no limit)")
    p.add_argument("--min-score", type=float, default=6, help="Drop clips scored below this")
    p.add_argument("--min-len", type=float, default=12)
    p.add_argument("--max-len", type=float, default=75)
    p.add_argument("--pad-before", type=float, default=1.5, help="Seconds of lead-in")
    p.add_argument("--pad-after", type=float, default=1.5, help="Seconds of tail")
    p.add_argument("--window", type=float, default=900, help="Transcript chunk size (s)")
    p.add_argument("--overlap", type=float, default=90, help="Chunk overlap (s)")
    p.add_argument("--context", default="", help="Notes for the AI, e.g. streamer name, what happened")
    p.add_argument("--vertical", action="store_true", help="Crop clips to 9:16")
    p.add_argument("--burn-captions", action="store_true", help="Burn captions into clips")
    p.add_argument("--dry-run", action="store_true", help="Find clips but don't cut video")
    p.add_argument("--clips-json", type=Path, help="Skip the AI: cut clips from an existing clips.json")
    p.add_argument("--jobs", type=int, default=4, help="Clips to cut at the same time")
    p.add_argument("--encoder", choices=["auto", "h264_videotoolbox", "libx264"], default="auto",
                   help="auto uses the Mac's hardware encoder when available")
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg isn't installed or isn't on your PATH.")

    if args.transcribe:
        args.transcript = transcribe(args.video, args.whisper_model)
    if not args.transcript and not args.clips_json:
        sys.exit("Give --transcript, --transcribe or --clips-json.")

    args.out.mkdir(parents=True, exist_ok=True)
    vlen = video_duration(args.video)

    if args.clips_json:
        data = json.loads(args.clips_json.read_text())
        clips = [Clip(**{k: c[k] for k in Clip.__dataclass_fields__}) for c in data["clips"]]
        segs = load_transcript(args.transcript) if args.transcript else []
    else:
        segs = load_transcript(args.transcript)
        print(f"Loaded {len(segs)} transcript lines "
              f"({segs[0].start/60:.0f}-{segs[-1].end/60:.0f} min).")

        try:
            import anthropic
        except ImportError:
            sys.exit("Install the Anthropic SDK:  pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set your ANTHROPIC_API_KEY environment variable first.")
        client = anthropic.Anthropic()

        chunks = chunk_segments(segs, args.window, args.overlap)
        raw: list[Clip] = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Analysing section {i}/{len(chunks)} "
                  f"({chunk[0].start/60:.0f}-{chunk[-1].end/60:.0f} min)...")
            found = find_clips_in_chunk(client, args.model, chunk,
                                        args.min_len, args.max_len, args.context)
            print(f"  -> {len(found)} candidate clips")
            raw.extend(found)

        clips = clean_clips(raw, segs, args.min_len, args.max_len,
                            args.min_score, args.top, vlen)
        manifest = {"video": str(args.video), "model": args.model,
                    "clips": [asdict(c) for c in clips]}
        (args.out / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(clips)} clips selected:")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. [{c.score:.0f}/10] {fmt_srt_time(c.start)[:8]} "
              f"({c.duration:.0f}s) {c.title}")

    if args.dry_run:
        print(f"\nDry run: clip list saved to {args.out/'clips.json'}. "
              f"Cut later with --clips-json {args.out/'clips.json'}")
        return

    encoder = pick_encoder() if args.encoder == "auto" else args.encoder
    print(f"\nCutting clips with {encoder} ({args.jobs} at a time)...")
    jobs = []
    for i, c in enumerate(clips, 1):
        start = max(c.start - args.pad_before, 0)
        end = c.end + args.pad_after
        if vlen:
            end = min(end, vlen)
        base = args.out / f"{i:02d}_{slugify(c.title)}"
        srt = base.with_suffix(".srt")
        if segs:
            write_clip_srt(srt, c, segs, c.start - start)
        jobs.append((args.video, base.with_suffix(".mp4"), start, end, args.vertical,
                     srt if (args.burn_captions and segs) else None, encoder))
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        futures = {pool.submit(cut_clip, *job): job[1].name for job in jobs}
        for done in as_completed(futures):
            done.result()
            print(f"  saved {futures[done]}")

    print(f"\nDone. {len(clips)} clips in {args.out}/ (details in clips.json)")


if __name__ == "__main__":
    main()
