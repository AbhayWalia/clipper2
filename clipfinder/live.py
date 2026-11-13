"""Clip a live stream while it's on.

    python clipper.py --live https://www.twitch.tv/NAME --vertical --burn-captions

ffmpeg records the stream in 10-second chunks. Every ~30 s of new recording is
transcribed with Whisper and scanned for loudness spikes and sounds, while chat is read
live. Every few minutes, or sooner once chat or the audio spikes, Claude reads the last
few minutes and picks moments; new clips are cut from the recorded chunks straight away.

A clip only counts once there's a minute of stream after it (the reaction is often the
best part), so clips appear about 2-4 minutes after the moment. Everything for a session
goes in OUT/live_<name>_<date>/: clips.json, the transcript, the clips, and the recent
recording in chunks/ (older chunks are deleted unless --keep-recording).

The clock everywhere (transcript, signals, clips.json) is seconds since recording began.
"""

from __future__ import annotations

import csv
import json
import queue
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CHUNK = 10          # seconds per recorded chunk
BLOCK = 30.0        # transcribe once this much new recording is in
EVERY = 180.0       # ask Claude at least this often (seconds of stream)
SETTLE = 60.0       # a clip must end this long before the newest analysed second
CONTEXT = 120.0     # how far back each look reaches before the last one's cut-off
MIN_GAP = 60.0      # spike-triggered looks no closer together than this
STALL = 120.0       # no new chunk for this long (wall clock) = the stream is over
STALL_ADS = 420.0   # ...with streamlink, which pauses during Twitch ad breaks (up to a few minutes)
KEEP = 900.0        # recording kept behind the newest analysed second
POLL = 2.0
MIN_LOOK = 60.0     # don't ask Claude about less new stream than this (not worth a call)
# Output budget per live look. Normal looks use a few hundred tokens; this caps what a
# runaway response can cost (one hit the default 16,000 on a 51 s stretch: $0.16, no clips).
LOOK_MAX_TOKENS = 8000
MAX_RESTARTS = 5
SPIKE_SOUNDS = {"screaming", "laughter", "explosion", "cheering", "crash"}
UI_FIELDS = ("review", "original_start", "original_end", "edited_at")  # set by the web UI


@dataclass
class Chunk:
    file: Path
    start: float  # session clock
    end: float
    run: int


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #

def resolve_stream(source: str, max_height: int = 1080) -> tuple[str, dict]:
    """The live stream's media URL (HLS) and yt-dlp's info for it."""
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("--live needs yt-dlp:  pip install yt-dlp")
    opts = {"format": f"best[height<={max_height}]/best", "quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).removeprefix("ERROR: ")
        if "offline" in msg.lower() or "not currently live" in msg.lower():
            raise SystemExit(f"{source} isn't live right now.")
        raise SystemExit(f"Couldn't open {source}: {msg}")
    if not info.get("is_live"):
        raise SystemExit(f"{source} isn't a live stream right now (for a finished VOD use --url).")
    if not info.get("url"):
        raise SystemExit(f"yt-dlp found no stream URL for {source}.")
    return info["url"], info


def has_streamlink() -> bool:
    import importlib.util
    return importlib.util.find_spec("streamlink") is not None


def streamlink_cmd(source: str, max_height: int = 1080) -> list[str]:
    """streamlink writing the stream to stdout. Its Twitch plugin leaves out the ads Twitch
    stitches into the stream for logged-out viewers (they can't be decoded as part of it)."""
    import sys
    quality = "best" if max_height >= 1080 else f"{max_height}p60,{max_height}p,best"
    return [sys.executable, "-m", "streamlink", "--stdout", "--loglevel", "info", source, quality]


def record_cmd(src: str, folder: Path, run: int, mode: str,
               readrate: float | None = None) -> list[str]:
    """ffmpeg cutting the input into chunks. mode: "file" (replay a file at stream speed, or
    faster, for trying live mode out), "url" (an HLS URL) or "pipe" (streamlink's output)."""
    if mode == "file":
        inp = ["-readrate", str(readrate)] if readrate else ["-re"]
    elif mode == "url":
        inp = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"]
    else:
        inp, src = [], "pipe:0"
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", *inp, "-i", src,
            "-map", "0:v:0?", "-map", "0:a:0", "-c", "copy",
            "-f", "segment", "-segment_time", str(CHUNK), "-segment_list_type", "csv",
            "-segment_list", str(folder / f"r{run}.csv"), str(folder / f"r{run}_%06d.ts")]


def read_chunk_list(path: Path, run: int, offset: float) -> list[Chunk]:
    """Finished chunks from ffmpeg's segment list (file,start,end per line)."""
    if not path.exists():
        return []
    out = []
    for row in csv.reader(path.read_text().splitlines()):
        if len(row) >= 3:
            try:
                out.append(Chunk(path.parent / row[0], offset + float(row[1]),
                                 offset + float(row[2]), run))
            except ValueError:
                continue
    return out


class Recorder:
    """ffmpeg writing the stream into chunks (fed by streamlink for Twitch when it's
    installed, so ads are left out); reconnects if the connection drops."""

    def __init__(self, source: str, folder: Path, max_height: int = 1080,
                 readrate: float | None = None):
        self.source, self.folder, self.max_height, self.readrate = source, folder, max_height, readrate
        self.file_input = Path(source).exists()
        site = source.lower()
        self.twitch = "twitch.tv" in site
        # streamlink for Twitch (it leaves the ads out) and Kick (yt-dlp's Kick support is
        # thinner); yt-dlp + ffmpeg for everything else, or if streamlink isn't installed.
        self.streamlink = (not self.file_input and (self.twitch or "kick.com" in site)
                           and has_streamlink())
        self.stall = STALL_ADS if self.streamlink and self.twitch else STALL
        self.info: dict = {}
        # (run number, session time its first frame belongs at); None until the run's
        # first chunk shows up, since streamlink may hold back pre-roll ads for a while.
        self.runs: list[list] = []
        self.proc: subprocess.Popen | None = None
        self.feeder: subprocess.Popen | None = None
        self.log = folder / "recording.log"  # ffmpeg/streamlink messages, read when they stop
        self.t0 = 0.0  # wall clock at session start
        self.restarts = 0
        self.stopped = False

    def start(self) -> None:
        folder = self.folder
        folder.mkdir(parents=True, exist_ok=True)
        if self.file_input:
            src, mode = self.source, "file"
        else:  # yt-dlp also tells us whether it's live, and the title and channel
            src, self.info = resolve_stream(self.source, self.max_height)
            mode = "pipe" if self.streamlink else "url"
        if not self.runs:
            self.t0 = time.time()
        run = len(self.runs)
        self.runs.append([run, None, time.time() - self.t0])
        # Messages go to a file: a pipe nobody reads fills up and stalls the recording.
        log = open(self.log, "a")
        stdin = subprocess.DEVNULL
        if mode == "pipe":
            self.feeder = subprocess.Popen(streamlink_cmd(self.source, self.max_height),
                                           stdout=subprocess.PIPE, stderr=log)
            stdin = self.feeder.stdout
        # stdout to DEVNULL: ffmpeg mustn't hold the web UI's log pipe open after we exit.
        self.proc = subprocess.Popen(record_cmd(src, folder, run, mode, self.readrate),
                                     stdin=stdin, stdout=subprocess.DEVNULL, stderr=log)
        if self.feeder:
            self.feeder.stdout.close()  # ffmpeg owns it now; streamlink sees it if ffmpeg dies
        log.close()

    def chunks(self) -> list[Chunk]:
        out = []
        for r in self.runs:
            run, offset, launched = r
            if offset is None:
                got = read_chunk_list(self.folder / f"r{run}.csv", run, 0.0)
                if not got:
                    continue
                # Place the run by when its first chunk turned up, not when it was launched
                # (pre-roll ads can delay the first frame), so chat stays lined up.
                r[1] = offset = max(launched, time.time() - self.t0 - got[0].end)
            out += read_chunk_list(self.folder / f"r{run}.csv", run, offset)
        return out

    def last_message(self) -> str:
        try:
            lines = [l for l in self.log.read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            return ""
        return lines[-1][-200:] if lines else ""

    def _end_feeder(self) -> None:
        if self.feeder and self.feeder.poll() is None:
            self.feeder.terminate()
            try:
                self.feeder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.feeder.kill()

    def recording(self) -> bool:
        """True while recording; reconnects (a few times) if it stopped but the channel is
        still live."""
        if self.proc is None or self.stopped:
            return False
        if self.proc.poll() is None:
            return True
        self._end_feeder()
        if self.file_input:
            return False  # the file ended
        why = self.last_message() or "the recording stopped"
        if self.restarts >= MAX_RESTARTS:
            print(f"  recording stopped ({why}); giving up after {MAX_RESTARTS} reconnects")
            return False
        self.restarts += 1
        try:
            self.start()  # says it isn't live (and we stop) if the stream really ended
        except SystemExit as e:
            print(f"  {e}")
            return False
        print(f"  recording dropped ({why}); reconnected")
        return True

    def stop(self) -> None:
        self.stopped = True
        if self.feeder and self.feeder.poll() is None:
            self._end_feeder()  # ffmpeg sees the end of its input and finishes the last chunk
        elif self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # lets ffmpeg finish the last chunk
        if self.proc:
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class LiveChat(threading.Thread):
    """Reads live chat in the background as (session seconds, message)."""

    def __init__(self, url: str, t0: float):
        super().__init__(daemon=True)
        self.url, self.t0 = url, t0
        self.messages: list[tuple[float, str]] = []

    def run(self) -> None:
        try:
            from chat_downloader import ChatDownloader
        except ImportError:
            print("  (live chat skipped: pip install chat-downloader)")
            return
        try:
            for m in ChatDownloader().get_chat(self.url):
                ts = m.get("timestamp")  # microseconds since the epoch
                wall = ts / 1e6 if ts else time.time()
                self.messages.append((wall - self.t0, str(m.get("message", ""))))
        except Exception as e:  # chat is a bonus; never stop clipping over it
            print(f"  live chat stopped ({e}); carrying on without it")


# --------------------------------------------------------------------------- #
# Helpers (pure, tested)
# --------------------------------------------------------------------------- #

def place(series: list, t0: float, values: list, fill) -> None:
    """Write per-second `values` into `series` starting at second t0, padding any gap."""
    i = max(int(round(t0)), 0)
    if len(series) < i:
        series.extend([fill] * (i - len(series)))
    series[i:i + len(values)] = values


def next_block(chunks: list[Chunk], done: set[Path], final: bool) -> list[Chunk]:
    """The next run of unprocessed, back-to-back chunks from one recording run: once it
    holds BLOCK seconds, or whatever is left when the stream is over."""
    todo = [c for c in chunks if c.file not in done]
    if not todo:
        return []
    block = [todo[0]]
    for c in todo[1:]:
        if c.run != block[-1].run or c.start - block[-1].end > 0.5:
            final = True  # a gap ends this block whatever its length
            break
        block.append(c)
        if block[-1].end - block[0].start >= BLOCK:
            return block
    return block if final or block[-1].end - block[0].start >= BLOCK else []


def overlaps(a, b, share: float = 0.5) -> bool:
    """Two clips cover the same moment: they share at least `share` of the shorter one."""
    inter = min(a.end, b.end) - max(a.start, b.start)
    return inter > 0 and inter >= share * min(a.end - a.start, b.end - b.start)


def fresh_clips(found: list, have: list, limit: float) -> list:
    """Clips that have settled (end by `limit`) and aren't already clipped."""
    out = []
    for c in sorted(found, key=lambda c: -c.score):
        if c.end <= limit + 1e-6 and not any(overlaps(c, h) for h in have + out):
            out.append(c)
    return sorted(out, key=lambda c: c.start)


def chunks_for(chunks: list[Chunk], start: float, end: float) -> list[Chunk]:
    """Back-to-back chunks of one run covering [start, end] (as much as was recorded)."""
    inside = [c for c in chunks if c.end > start and c.start < end]
    if not inside:
        return []
    run = next((c.run for c in inside if c.start <= start < c.end), inside[0].run)
    return [c for c in inside if c.run == run]


RENDER_FLAGS = ["vertical", "burn_captions", "hook_overlay", "preview", "reencode", "no_facecam"]
RENDER_VALUES = ["streamer", "layouts", "facecam", "gameplay", "cam_height", "cam_position",
                 "caption_style", "caption_font", "caption_color", "highlight_color",
                 "caption_case", "hook_seconds", "pad_before", "pad_after", "min_len",
                 "max_len", "jobs", "encoder"]


def render_argv(args, recipe: Path, numbers: list[int]) -> list[str]:
    """clipper.py arguments that cut these clips from the session's clips.json."""
    out = ["--clips-json", str(recipe), "--out", str(recipe.parent),
           "--only", ",".join(map(str, numbers))]
    for k in RENDER_FLAGS:
        if getattr(args, k, False):
            out.append("--" + k.replace("_", "-"))
    for k in RENDER_VALUES:
        v = getattr(args, k, None)
        if v is not None:
            out += ["--" + k.replace("_", "-"), str(v)]
    return out


def fmt_clock(t: float) -> str:
    t = int(max(t, 0))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #

class LiveSession:
    def __init__(self, args, folder: Path):
        from .sounds import GROUPS
        self.args, self.folder = args, folder
        self.chunk_dir = folder / "chunks"
        self.recipe = folder / "clips.json"
        self.transcript = folder / "transcript.json"
        self.done: set[Path] = set()
        self.raw_segments: list[dict] = []
        self.levels: list[float] = []
        self.sounds: dict[str, list[float]] = {g: [] for g in GROUPS}
        self.sound_model = None
        self.clips: list = []
        self.segments: list[dict] = []
        self.analysed = 0.0       # newest second transcribed and scanned
        self.accepted = 0.0       # clips ending before this are settled
        self.last_look = 0.0      # `analysed` when Claude last looked
        self.chat: LiveChat | None = None
        self.status = "starting"
        # Cutting runs in the background so transcription keeps up with the stream.
        self.renders: queue.Queue[list[int]] = queue.Queue()
        self.renderer: threading.Thread | None = None

    # --- analysis -----------------------------------------------------------
    def load_sound_model(self) -> None:
        if self.args.no_sound_tags:
            return
        try:
            from .sounds import load_model
            self.sound_model = load_model()
        except ImportError:
            print("  (sound tags skipped: pip install panns-inference)")
        except Exception as e:
            print(f"  (sound tags skipped: {e})")

    def analyse(self, block: list[Chunk]) -> None:
        from .signals import measure_loudness
        from .transcribe import transcribe_audio
        t0, t1 = block[0].start, block[-1].end
        audio = self.folder / "block.flac"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", "concat:" + "|".join(str(c.file) for c in block),
                        "-vn", "-ac", "1", "-ar", "32000", "-c:a", "flac", str(audio)], check=True)
        segs = transcribe_audio(audio, self.args.whisper_model, progress=False)
        for s in segs:
            s["start"], s["end"] = s["start"] + t0, s["end"] + t0
            for w in s.get("words", []):
                w["start"], w["end"] = w["start"] + t0, w["end"] + t0
        self.raw_segments += segs
        self.transcript.write_text(json.dumps({"segments": self.raw_segments}), encoding="utf-8")
        if not self.args.no_audio_signals:
            try:
                place(self.levels, t0, measure_loudness(audio), -90.0)
            except Exception as e:
                print(f"  loudness scan failed ({e})")
        if self.sound_model:
            from .sounds import tag_audio
            try:
                for g, vals in tag_audio(audio, None, self.sound_model).items():
                    place(self.sounds.setdefault(g, []), t0, vals, 0.0)
            except Exception as e:
                print(f"  sound tagging failed ({e})")
        audio.unlink(missing_ok=True)
        self.done.update(c.file for c in block)
        self.analysed = max(self.analysed, t1)
        self.save()  # also the heartbeat the web UI watches
        print(f"[{fmt_clock(t1)}] transcribed {t1 - t0:.0f}s: {len(segs)} line(s)")

    def events(self) -> list:
        from .signals import audio_events, chat_events
        from .sounds import music_events, sound_events
        ev = audio_events(self.levels, self.args.audio_threshold) if self.levels else []
        ev += sound_events(self.sounds) + music_events(self.sounds)
        if self.chat and self.chat.messages:
            ev += chat_events(list(self.chat.messages), offset=self.args.chat_offset)
        return ev

    def spike_since(self, t: float) -> float | None:
        """Start of the first chat burst, loudness spike or telling sound after t."""
        hits = [e.start for e in self.events() if e.start >= t and
                (e.kind in ("chat", "audio") or e.detail in SPIKE_SOUNDS)]
        return min(hits) if hits else None

    def should_look(self, final: bool) -> bool:
        if self.analysed - self.accepted < MIN_LOOK:
            return False
        if final or self.analysed - self.last_look >= EVERY:
            return True
        spike = self.spike_since(self.accepted)
        return (spike is not None and self.analysed - spike >= SETTLE
                and self.analysed - self.last_look >= MIN_GAP)

    # --- Claude and cutting ---------------------------------------------------
    def look(self, final: bool) -> list[int]:
        """Ask Claude about the newest stretch; returns the numbers of new clips."""
        from .cli import find_clips
        from .signals import Signals
        from .transcript import load_transcript, segments_between
        t0 = max(0.0, self.accepted - CONTEXT)
        t1 = self.analysed
        limit = t1 if final else t1 - SETTLE
        self.last_look = t1
        segs = segments_between(load_transcript(self.transcript), t0, t1) if self.raw_segments else []
        if not segs:
            self.accepted = max(self.accepted, limit)
            return []
        ev = [e for e in self.events() if e.end > t0 and e.start < t1]
        print(f"[{fmt_clock(t1)}] asking Claude about {fmt_clock(t0)}-{fmt_clock(t1)}"
              + (f" ({len(ev)} signal(s))" if ev else "") + "...")
        try:
            found = find_clips(self.args, segs, None, signals=Signals(ev, segs) if ev else None,
                               max_tokens=LOOK_MAX_TOKENS)
        except SystemExit as e:  # every call failed; try this stretch again next time
            print(f"  {e}")
            return []
        new = fresh_clips(found, self.clips, limit)
        self.accepted = max(self.accepted, limit)
        numbers = []
        for c in new:
            self.clips.append(c)
            numbers.append(len(self.clips))
            print(f"  new clip {len(self.clips):02d}: [{c.score:.0f}/10] {fmt_clock(c.start)} "
                  f"({c.duration:.0f}s) {c.title}")
        if not new:
            print("  nothing new worth clipping yet")
        self.save()
        return numbers

    def cut(self, numbers: list[int], chunks: list[Chunk]) -> None:
        """Copy each clip's footage out of the chunks (fast, so chunks can be deleted
        later), then queue the actual cut for the background renderer."""
        from .fetch import section_name
        ready = []
        for n in numbers:
            c = self.clips[n - 1]
            s = max(c.start - self.args.pad_before - 2, 0.0)
            e = c.end + self.args.pad_after + 2
            part = chunks_for(chunks, s, e)
            if not part:
                print(f"  clip {n:02d}: its recording is gone, can't cut it")
                continue
            entry = {"file": section_name(part[0].start, part[-1].end).replace("sec_", "live_"),
                     "start": round(part[0].start, 3), "end": round(part[-1].end, 3)}
            (self.folder / "segments").mkdir(exist_ok=True)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-i", "concat:" + "|".join(str(x.file) for x in part),
                            "-c", "copy", str(self.folder / entry["file"])], check=True)
            self.segments.append(entry)
            ready.append(n)
        if not ready:
            return
        self.save()
        if self.renderer is None:
            self.renderer = threading.Thread(target=self._render_loop, daemon=True)
            self.renderer.start()
        self.renders.put(ready)

    def _render_loop(self) -> None:
        from .cli import main as clipper_main
        while True:
            numbers = self.renders.get()
            try:
                clipper_main(render_argv(self.args, self.recipe, numbers), quiet=True)
            except SystemExit as e:
                if e.code not in (0, None):
                    print(f"  cutting stopped: {e}")
            except Exception as e:  # keep clipping the stream even if one cut breaks
                print(f"  cutting clip(s) {numbers} failed: {e}")
            finally:
                self.renders.task_done()

    def wait_for_cuts(self) -> None:
        if self.renders.unfinished_tasks:
            print("Waiting for the last clips to finish cutting...")
        self.renders.join()

    def prune(self, chunks: list[Chunk]) -> None:
        if self.args.keep_recording:
            return
        for c in chunks:
            if c.file in self.done and c.end < self.analysed - KEEP:
                c.file.unlink(missing_ok=True)

    def save(self) -> None:
        """Write clips.json, keeping what the web UI changed in it meanwhile (Keep/Reject,
        trims), since you can review clips while the stream is still on."""
        from dataclasses import asdict
        from .clips import write_json
        disk: dict = {}
        if self.recipe.exists():
            try:
                disk = json.loads(self.recipe.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        old = disk.get("clips") or []
        clips = []
        for i, c in enumerate(self.clips):
            o = old[i] if i < len(old) else {}
            if o.get("edited_at"):  # trimmed in the UI
                c.start, c.end = o["start"], o["end"]
            d = asdict(c)
            d.update({k: o[k] for k in UI_FIELDS if k in o})
            clips.append(d)
        write_json(self.recipe, {
            "video": None, "transcript": str(self.transcript.resolve()), "model": self.args.model,
            "effort": self.args.effort, "streamer": self.args.streamer, "segments": self.segments,
            "runs": disk.get("runs") or [],
            **({"layout": disk["layout"]} if "layout" in disk else {}),  # found by the renderer
            "live": {"source": self.args.live, "status": self.status,
                     "analysed": round(self.analysed, 1),
                     "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "clips": clips})

    # --- the loop -------------------------------------------------------------
    def step(self, rec: Recorder, final: bool) -> None:
        chunks = rec.chunks()
        from .stats import STATS
        while block := next_block(chunks, self.done, final):
            with STATS.stage("transcribe"):
                self.analyse(block)
            if not final:
                break  # keep up: one block per tick, look at Claude in between
        if self.should_look(final):
            try:
                with STATS.stage("find clips"):
                    numbers = self.look(final)
            except Exception as e:  # a hiccup mustn't end a long session; next look retries
                print(f"  checking for clips failed ({type(e).__name__}: {e}); will try again")
                numbers = []
            if numbers and not self.args.dry_run:
                self.cut(numbers, rec.chunks())
        self.prune(chunks)


def session_name(source: str, info: dict) -> str:
    who = info.get("uploader") or info.get("channel") or Path(source.rstrip("/")).stem
    slug = re.sub(r"[^a-z0-9_]+", "_", who.lower()).strip("_") or "stream"
    return f"live_{slug}_{datetime.now().strftime('%Y%m%d-%H%M')}"


def run_live(args) -> None:
    from .cli import report_run
    from .fetch import streamer_slug
    import os
    import shutil
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg isn't installed or isn't on your PATH.")
    if not os.environ.get("ANTHROPIC_API_KEY"):  # fail now, not at the first look minutes in
        raise SystemExit("Live mode needs ANTHROPIC_API_KEY (set it, or put it in .env).")
    # Ctrl+C (or Stop in the web UI) must reach us even if we were started with it ignored,
    # as programs launched in the background are: it's how a session wraps up properly.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    file_input = Path(args.live).exists()
    info = {} if file_input else resolve_stream(args.live, args.max_height)[1]
    folder = args.out / session_name(args.live, info)
    if not args.streamer and info.get("uploader"):
        args.streamer = streamer_slug(info["uploader"])
    if not args.context and info.get("title"):
        args.context = f"Live stream title: {info['title']}"
    # find_clips uses these for clip ids and the feedback features
    args.video = folder / "recording"

    session = LiveSession(args, folder)
    rec = Recorder(args.live, session.chunk_dir, args.max_height, args.live_readrate)
    print(f"Clipping {args.live} live into {folder}/ (Ctrl+C to stop)")
    if rec.streamlink and rec.twitch:
        print("  recording with streamlink, which leaves out Twitch's ads")
    elif "twitch.tv" in args.live.lower():
        print("  note: Twitch's ads will be recorded too (they show up as broken video). "
              "pip install streamlink to leave them out")
    session.load_sound_model()
    rec.start()
    session.status = "recording"
    if "kick.com" in args.live.lower():
        print("  (live chat isn't available for Kick; clipping without chat bursts)")
    elif not file_input and not args.no_chat:
        session.chat = LiveChat(args.live, rec.t0)
        session.chat.start()
    session.save()

    last_new, seen = time.time(), 0
    try:
        while True:
            time.sleep(POLL)
            alive = rec.recording()
            n = len(rec.chunks())
            if n > seen:
                last_new, seen = time.time(), n
            elif alive and time.time() - last_new > rec.stall:
                print(f"  no new video for {rec.stall:.0f}s; treating the stream as over")
                rec.stop()
                alive = False
            if not alive:
                break
            session.step(rec, final=False)
    except KeyboardInterrupt:
        print("\nStopping: finishing the last few minutes...")
        rec.stop()
    session.status = "finishing"
    try:
        session.step(rec, final=True)
        session.wait_for_cuts()
    except KeyboardInterrupt:
        print("Stopped before the last stretch was checked and cut.")
    session.status = "ended"
    session.save()
    print(f"\nDone. {len(session.clips)} clip(s) in {folder}/")
    report_run(session.recipe, "live", len(session.clips))
