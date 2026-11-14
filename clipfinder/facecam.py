"""Find the streamer's webcam in the stream automatically (`--facecam auto`).

A few frames from across the video go to Claude, which says whether there's a webcam
overlay of the streamer and where. One call, under a cent. (Snapping the box onto edges
that stay put across frames doesn't work: the streamer's room is still too, so its door
frames and walls look just like the overlay's border.) The result is saved in clips.json (and as the
streamer's layout in layouts.json), so it's only worked out once per streamer.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from .layout import Box, check_box

FRAME_WIDTH = 768
N_FRAMES = 4

SYSTEM = """You look at frames from a gaming livestream and find the streamer's webcam (facecam).

The facecam is a rectangle overlaid on the game showing the real person streaming, filmed by a
camera: usually in a corner, often with a border or frame around it. It is NOT a game character,
an avatar or a cartoon, a chat box, an alert or a logo. Some streams have no facecam.

The camera doesn't move during a stream, so give one box that fits it in all frames. Cover the
whole camera picture (the person and their room behind them) and stop at its edges: leave out
decorative borders, names and follower counts around it, and any game around it."""

SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean", "description": "true if there is a webcam of the streamer"},
        "x": {"type": "integer", "description": "left edge of the facecam, in pixels"},
        "y": {"type": "integer", "description": "top edge, in pixels"},
        "w": {"type": "integer", "description": "width, in pixels"},
        "h": {"type": "integer", "description": "height, in pixels"},
        "note": {"type": "string", "description": "a few words on what you saw"},
    },
    "required": ["found", "x", "y", "w", "h", "note"],
    "additionalProperties": False,
}


def frame_times(duration: float | None, n: int = N_FRAMES) -> list[float]:
    """Times spread across the video (not the very start, which is often a waiting screen)."""
    d = duration or 60.0
    return [round(d * (0.15 + 0.6 * i / max(n - 1, 1)), 2) for i in range(n)]


def grab_jpeg(video: Path, t: float, width: int = FRAME_WIDTH) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "mjpeg", "-"],
        capture_output=True, check=True).stdout


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Width and height from a JPEG's SOF header (no image library needed)."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker, length = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    raise ValueError("not a JPEG frame")


def box_from_answer(ans: dict, size: tuple[int, int]) -> Box | None:
    """Claude's pixel box -> fractions of the frame, or None if there's no usable facecam."""
    if not ans or not ans.get("found"):
        return None
    W, H = size
    x, y = max(ans["x"], 0), max(ans["y"], 0)
    w, h = min(ans["w"], W - x), min(ans["h"], H - y)
    if w < W * 0.05 or h < H * 0.05:  # too small to be a usable camera
        return None
    return check_box((x / W, y / H, w / W, h / H))


def ask_claude(frames: list[bytes], model: str, client=None) -> tuple[Box | None, str]:
    """Where the facecam is (as fractions) and Claude's note; (None, note) if there isn't one."""
    import json
    from .stats import STATS
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=4)
    W, H = jpeg_size(frames[0])
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(f).decode()}}
               for f in frames]
    content.append({"type": "text", "text":
                    f"These {len(frames)} frames are {W}x{H} pixels, from the same stream. "
                    "Is there a facecam? If so, give its box in pixels of these frames."})
    msg = client.messages.create(
        model=model, max_tokens=2000, system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}])
    STATS.add(msg)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    ans = json.loads(text) if msg.stop_reason != "refusal" and text else {}
    return box_from_answer(ans, (W, H)), ans.get("note", "")


def find_facecam(sources: list[tuple[Path, float]], model: str, client=None) -> tuple[Box | None, str]:
    """Look for the facecam in frames at (video, time) pairs. Returns (box or None, note)."""
    frames = [grab_jpeg(v, t) for v, t in sources]
    frames = [f for f in frames if f]
    if not frames:
        return None, "couldn't read any frames"
    return ask_claude(frames, model, client)
