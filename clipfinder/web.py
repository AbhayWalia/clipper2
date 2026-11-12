"""Local web UI: start clipper runs from a form, watch the log, browse the clips.

    python ui.py            # then open http://localhost:8000

Standard library only. It runs `clipper.py` as a subprocess with the options from
the form, so the web UI and the command line always behave the same.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .render import slugify

ROOT = Path.cwd().resolve()
STATIC = Path(__file__).parent / "static"
CLIPPER = Path(__file__).resolve().parent.parent / "clipper.py"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
CLIP_NAME = re.compile(r"^(\d+)_.*\.mp4$")
PROMPT_FILE = ROOT / "prompt.txt"  # the clip-finding prompt, editable from the UI
RECIPE_LOCK = threading.Lock()


def load_env() -> None:
    """Read ANTHROPIC_API_KEY etc. from ./.env or ../.env if they aren't set already."""
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# --------------------------------------------------------------------------- #
# Runs: one clipper subprocess at a time
# --------------------------------------------------------------------------- #

class Job:
    def __init__(self, cmd: list[str], out: str):
        self.cmd, self.out = cmd, out
        self.lines: list[str] = []
        self.started = time.time()
        self.ended: float | None = None
        self.code: int | None = None
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            # ffmpeg/whisper progress bars redraw with \r; keep the last state.
            self.lines.append(line.rstrip("\n").split("\r")[-1])
        self.code = self.proc.wait()
        self.ended = time.time()

    @property
    def running(self) -> bool:
        return self.code is None

    def state(self, since: int = 0) -> dict:
        return {"running": self.running, "code": self.code, "out": self.out,
                "cmd": " ".join(self.cmd[1:]), "started": self.started, "ended": self.ended,
                "lines": self.lines[since:], "total": len(self.lines)}


JOB: Job | None = None
JOB_LOCK = threading.Lock()


def build_command(f: dict) -> tuple[list[str], str]:
    """Turn the form into clipper.py arguments. Only known fields are passed through."""
    def text(key: str) -> str:
        return str(f.get(key) or "").strip()

    out = text("out") or "clips"
    args = ["--out", out]
    source = text("source")
    if source == "url":
        if not text("url"):
            raise ValueError("Paste a Twitch or YouTube link.")
        args += ["--url", text("url")]
    elif source == "video":
        if not text("video"):
            raise ValueError("Pick a video file.")
        args += ["--video", text("video")]
        if text("transcript"):
            args += ["--transcript", text("transcript")]
        else:
            args.append("--transcribe")
    elif source == "rerender":
        if not text("clips_json"):
            raise ValueError("Pick a clips.json to re-render.")
        args += ["--clips-json", text("clips_json")]
        args[1] = str(Path(text("clips_json")).parent)  # render next to its recipe
        out = args[1]
    else:
        raise ValueError("Choose a source.")

    for key, flag in (("streamer", "--streamer"), ("context", "--context"),
                      ("effort", "--effort"), ("chat", "--chat")):
        if text(key):
            args += [flag, text(key)]
    for key, flag in (("top", "--top"), ("min_score", "--min-score"),
                      ("min_len", "--min-len"), ("max_len", "--max-len"),
                      ("pad_before", "--pad-before"), ("pad_after", "--pad-after")):
        if text(key):
            float(text(key))  # reject non-numbers before they reach the shell
            args += [flag, text(key)]
    if source != "rerender" and PROMPT_FILE.is_file():
        args += ["--prompt-file", PROMPT_FILE.name]
    only = f.get("only")
    if only:
        if not (isinstance(only, list) and all(isinstance(n, int) and n > 0 for n in only)):
            raise ValueError("Bad clip selection.")
        args += ["--only", ",".join(map(str, sorted(only)))]
    for key, flag in (("vertical", "--vertical"), ("burn_captions", "--burn-captions"),
                      ("hook_overlay", "--hook-overlay"), ("preview", "--preview"),
                      ("dry_run", "--dry-run"), ("batch", "--batch")):
        if f.get(key) is True:
            args.append(flag)
    if f.get("sound_tags") is False:
        args.append("--no-sound-tags")
    return [sys.executable, str(CLIPPER), *args], out


# --------------------------------------------------------------------------- #
# Library: output folders and their clips
# --------------------------------------------------------------------------- #

def safe_path(rel: str) -> Path | None:
    p = (ROOT / rel).resolve()
    return p if p.is_relative_to(ROOT) else None


def folder_clips(folder: Path) -> list[dict]:
    """Clips in a folder (and its previews/), matched to clips.json by their number."""
    meta: dict = {}
    recipe = folder / "clips.json"
    if recipe.is_file():
        try:
            meta = json.loads(recipe.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    info = meta.get("clips") or []
    out, seen = [], set()
    for sub in (folder, folder / "previews"):
        if not sub.is_dir():
            continue
        for mp4 in sorted(sub.glob("*.mp4")):
            m = CLIP_NAME.match(mp4.name)
            if not m or mp4.stem in seen:  # full-quality version already listed
                continue
            seen.add(mp4.stem)
            i = int(m.group(1))
            c = info[i - 1] if 0 < i <= len(info) else {}
            # Only trust the match if the name is what cli.py would give this clip;
            # otherwise it's an older video left in the folder by a previous run.
            if c and mp4.stem != f"{i:02d}_{slugify(c.get('title', ''))}":
                c = {}
            title = c.get("title") or mp4.stem.split("_", 1)[1].replace("_", " ").capitalize()
            out.append({
                "n": i, "file": mp4.relative_to(ROOT).as_posix(), "title": title,
                "preview": sub.name == "previews", "size": mp4.stat().st_size,
                "score": c.get("score"), "hook": c.get("hook"), "category": c.get("category"),
                "reason": c.get("reason"), "start": c.get("start"), "end": c.get("end"),
                "srt": mp4.with_suffix(".srt").is_file(), "matched": bool(c),
                "review": c.get("review"),
                # Trimmed after this video was made: it no longer matches the new edges.
                "stale": bool(c.get("edited_at")) and c["edited_at"] > mp4.stat().st_mtime,
            })
    return out


def uncut(folder: Path) -> int:
    """Clips listed in clips.json that have no full-quality video yet."""
    try:
        info = json.loads((folder / "clips.json").read_text(encoding="utf-8")).get("clips") or []
    except (OSError, ValueError):
        return 0
    return sum(not (folder / f"{i:02d}_{slugify(c.get('title', ''))}.mp4").is_file()
               for i, c in enumerate(info, 1))


def library() -> list[dict]:
    folders = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        clips = folder_clips(d)
        recipe = d / "clips.json"
        if not clips and not recipe.is_file():
            continue
        runs = []
        if recipe.is_file():
            try:
                runs = json.loads(recipe.read_text(encoding="utf-8")).get("runs") or []
            except (OSError, ValueError):
                pass
        costs = [r["cost_usd"] for r in runs if r.get("cost_usd") is not None]
        folders.append({"name": d.name, "clips": clips, "uncut": uncut(d) if recipe.is_file() else 0,
                        "approved": sorted(c["n"] for c in clips if c["review"] == "approved"),
                        "rejected": sum(c["review"] == "rejected" for c in clips),
                        "cost": round(sum(costs), 4) if costs else None,
                        "last_run": runs[-1] if runs else None,
                        "recipe": recipe.relative_to(ROOT).as_posix() if recipe.is_file() else None,
                        "modified": max([(ROOT / c["file"]).stat().st_mtime for c in clips]
                                        + ([recipe.stat().st_mtime] if recipe.is_file() else []))})
    folders.sort(key=lambda f: f["modified"], reverse=True)
    return folders


def update_clip(form: dict) -> dict:
    """Approve/reject a clip or nudge its edges, saved straight into clips.json."""
    recipe = safe_path(str(form.get("recipe") or ""))
    if not recipe or recipe.name != "clips.json" or not recipe.is_file():
        raise ValueError("Clip list not found.")
    n = form.get("n")
    with RECIPE_LOCK:
        data = json.loads(recipe.read_text(encoding="utf-8"))
        clips = data.get("clips") or []
        if not isinstance(n, int) or not 0 < n <= len(clips):
            raise ValueError("Clip not found.")
        c = clips[n - 1]
        if "review" in form:
            if form["review"] not in ("approved", "rejected", None):
                raise ValueError("Bad review value.")
            if form["review"]:
                c["review"] = form["review"]
            else:
                c.pop("review", None)
        for key in ("start", "end"):
            delta = form.get(f"nudge_{key}")
            if delta is None:
                continue
            delta = float(delta)
            if abs(delta) > 30:
                raise ValueError("Nudge too large.")
            c.setdefault(f"original_{key}", c[key])
            new = round(max(c[key] + delta, 0.0), 2)
            if (key == "start" and new >= c["end"] - 1) or (key == "end" and new <= c["start"] + 1):
                raise ValueError("A clip must stay at least 1 second long.")
            c[key] = new
            c["edited_at"] = time.time()
        recipe.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"start": c["start"], "end": c["end"], "review": c.get("review")}


def get_prompt() -> dict:
    from .llm import SYSTEM_PROMPT
    custom = PROMPT_FILE.is_file()
    return {"text": PROMPT_FILE.read_text(encoding="utf-8") if custom else SYSTEM_PROMPT,
            "custom": custom, "default": SYSTEM_PROMPT, "file": PROMPT_FILE.name}


def save_prompt(form: dict) -> dict:
    if form.get("reset"):
        PROMPT_FILE.unlink(missing_ok=True)
        return get_prompt()
    text = str(form.get("text") or "").strip()
    if len(text) < 50:
        raise ValueError("That prompt is too short to find clips with.")
    PROMPT_FILE.write_text(text + "\n", encoding="utf-8")
    return get_prompt()


def inputs() -> dict:
    videos, transcripts = [], []
    for p in sorted(ROOT.iterdir()):
        if p.suffix.lower() in VIDEO_EXT and p.is_file():
            videos.append({"path": p.name, "size": p.stat().st_size})
        elif p.suffix.lower() in {".srt", ".vtt"} or p.name.endswith(".transcript.json"):
            transcripts.append(p.name)
    layouts = ROOT / "layouts.json"
    try:
        streamers = sorted(json.loads(layouts.read_text(encoding="utf-8"))) if layouts.is_file() else []
    except (OSError, ValueError):
        streamers = []
    return {"videos": videos, "transcripts": transcripts, "streamers": streamers,
            "api_key": bool(os.environ.get("ANTHROPIC_API_KEY"))}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "clipfinder"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def send_json(self, data, status=HTTPStatus.OK) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self.send_file(STATIC / "index.html")
        if url.path == "/api/library":
            return self.send_json(library())
        if url.path == "/api/inputs":
            return self.send_json(inputs())
        if url.path == "/api/prompt":
            return self.send_json(get_prompt())
        if url.path == "/api/job":
            since = int((url.query.partition("since=")[2] or "0").split("&")[0] or 0)
            return self.send_json(JOB.state(since) if JOB else None)
        if url.path.startswith("/media/"):
            p = safe_path(unquote(url.path[len("/media/"):]))
            if p and p.is_file() and p.suffix.lower() in {".mp4", ".srt", ".json"}:
                return self.send_file(p, download="download" in url.query)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        global JOB
        length = int(self.headers.get("Content-Length") or 0)
        try:
            form = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send_json({"error": "Bad request"}, HTTPStatus.BAD_REQUEST)
        if self.path == "/api/run":
            with JOB_LOCK:
                if JOB and JOB.running:
                    return self.send_json({"error": "A run is already going."}, HTTPStatus.CONFLICT)
                try:
                    cmd, out = build_command(form)
                except ValueError as e:
                    return self.send_json({"error": str(e) or "Check the numbers."},
                                          HTTPStatus.BAD_REQUEST)
                JOB = Job(cmd, out)
            return self.send_json(JOB.state())
        if self.path in ("/api/clip", "/api/prompt"):
            try:
                return self.send_json(update_clip(form) if self.path == "/api/clip" else save_prompt(form))
            except (ValueError, TypeError, KeyError) as e:
                return self.send_json({"error": str(e) or "Bad request"}, HTTPStatus.BAD_REQUEST)
        if self.path == "/api/stop":
            if JOB and JOB.running:
                JOB.proc.terminate()
            return self.send_json({"ok": True})
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_file(self, path: Path, download: bool = False) -> None:
        """Serve a file, honouring Range requests so videos can seek (Safari needs this)."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(size - int(rng.group(2)), 0)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser stopped reading (normal when seeking)


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Web UI for the clip finder.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 unless you trust your network")
    args = p.parse_args(argv)
    load_env()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Clip finder UI running at http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if JOB and JOB.running:
            JOB.proc.terminate()


if __name__ == "__main__":
    main()
