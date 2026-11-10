"""Per-streamer 9:16 layouts: facecam on top, gameplay below (or the other way).

Boxes are stored as fractions of the source frame (x, y, w, h in 0-1), so one
layout works for 720p, 1080p and 1440p VODs from the same streamer.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

Box = tuple[float, float, float, float]  # x, y, w, h as fractions of the source frame

DEFAULT_LAYOUTS = Path("layouts.json")


@dataclass
class Layout:
    facecam: Box
    gameplay: Box = (0.0, 0.0, 1.0, 1.0)  # area to take gameplay from (centre-cropped to fit)
    cam_height: float = 0.35               # share of the 9:16 frame the facecam gets
    cam_position: str = "top"              # "top" or "bottom"

    def __post_init__(self):
        self.facecam = check_box(self.facecam, "facecam")
        self.gameplay = check_box(self.gameplay, "gameplay")
        if not 0.1 <= self.cam_height <= 0.7:
            raise ValueError("cam_height must be between 0.1 and 0.7")
        if self.cam_position not in ("top", "bottom"):
            raise ValueError('cam_position must be "top" or "bottom"')

    def seam(self) -> float:
        """Where facecam meets gameplay, as a share of output height."""
        return self.cam_height if self.cam_position == "top" else 1 - self.cam_height


def check_box(box, name: str = "box") -> Box:
    x, y, w, h = (float(v) for v in box)
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
            and x + w <= 1.0001 and y + h <= 1.0001):
        raise ValueError(f"{name} {box} must be x,y,w,h fractions inside the frame (0-1)")
    return (x, y, w, h)


def parse_box(text: str, frame: tuple[int, int] | None = None) -> Box:
    """'0.72,0.03,0.26,0.30' (fractions) or '1382,32,500,324' (pixels; needs frame size)."""
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise ValueError("A box is four numbers: x,y,w,h")
    if any(v > 1 for v in vals):
        if not frame:
            raise ValueError("Pixel boxes need the video's size; use fractions (0-1) instead")
        fw, fh = frame
        vals = [vals[0] / fw, vals[1] / fh, vals[2] / fw, vals[3] / fh]
    return check_box(vals)


def load_layouts(path: Path) -> dict[str, Layout]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: Layout(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()})
            for name, cfg in data.items()}


def save_layout(path: Path, name: str, layout: Layout) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(layout).items()}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _even(n: float) -> int:
    return max(int(round(n / 2)) * 2, 2)


def _crop(box: Box) -> str:
    x, y, w, h = box
    return f"crop=w=iw*{w:.4f}:h=ih*{h:.4f}:x=iw*{x:.4f}:y=ih*{y:.4f}"


def split_filter(layout: Layout, out_w: int, out_h: int, src: str = "0:v",
                 out: str = "base") -> str:
    """filter_complex chain that builds the stacked 9:16 frame as [out]."""
    cam_h = _even(out_h * layout.cam_height)
    game_h = out_h - cam_h
    fill = "force_original_aspect_ratio=increase"
    parts = [
        f"[{src}]split=2[camsrc][gamesrc]",
        f"[camsrc]{_crop(layout.facecam)},scale={out_w}:{cam_h}:{fill},crop={out_w}:{cam_h}[cam]",
        f"[gamesrc]{_crop(layout.gameplay)},scale={out_w}:{game_h}:{fill},crop={out_w}:{game_h}[game]",
        ("[cam][game]" if layout.cam_position == "top" else "[game][cam]")
        + f"vstack=inputs=2,setsar=1[{out}]",
    ]
    return ";".join(parts)


def grab_frame_cmd(video: Path, t: float, out: Path) -> list[str]:
    """A still with a 10% grid, for reading off a facecam box (each square = 0.1)."""
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
            "-frames:v", "1",
            "-vf", "drawgrid=w=iw/20:h=ih/20:t=1:c=yellow@0.5,"   # 0.05 steps
                   "drawgrid=w=iw/10:h=ih/10:t=5:c=black@0.9,"    # 0.1 steps: dark edge...
                   "drawgrid=w=iw/10:h=ih/10:t=2:c=white",        # ...white line, visible on anything
            str(out)]


def grab_frame(video: Path, t: float, out: Path) -> Path:
    subprocess.run(grab_frame_cmd(video, t, out), check=True)
    return out
