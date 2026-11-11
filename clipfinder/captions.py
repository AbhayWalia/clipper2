"""Burned-in text as an ASS subtitle file: TikTok-style word-by-word captions
(the spoken word lights up) plus an optional hook banner for the first seconds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .transcript import Segment, Word


@dataclass
class CaptionStyle:
    animated: bool = True          # highlight the current word; False = plain phrases
    font: str = "Arial"
    color: str = "FFFFFF"          # RRGGBB
    highlight: str = "FFE500"      # RRGGBB for the word being spoken
    uppercase: bool = True
    max_words: int = 3             # words on screen at once
    size_ratio: float = 0.047      # font size as a share of frame height
    hook_size_ratio: float = 0.036


def ass_color(rrggbb: str) -> str:
    """'FFE500' -> '&H0000E5FF' (ASS is &HAABBGGRR)."""
    h = rrggbb.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Colour must be RRGGBB, got {rrggbb!r}")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def words_in_window(segs: list[Segment], t0: float, t1: float) -> list[tuple[Word, int]]:
    """Words spoken in [t0, t1], re-timed to the clip (0 = t0), with the index of
    the segment they came from (so phrases don't run across sentences)."""
    out = []
    for i, s in enumerate(segs):
        if s.end <= t0 or s.start >= t1:
            continue
        for w in s.timed_words():
            if w.end <= t0 or w.start >= t1:
                continue
            out.append((Word(max(w.start, t0) - t0, min(w.end, t1) - t0, w.word), i))
    return out


def group_words(words: list[tuple[Word, int]], max_words: int = 3,
                max_gap: float = 0.6) -> list[list[Word]]:
    """Split into short on-screen phrases: at most max_words, broken at pauses,
    sentence ends and segment boundaries."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    last_seg = None
    for w, seg in words:
        if cur and (len(cur) >= max_words or seg != last_seg
                    or w.start - cur[-1].end > max_gap
                    or cur[-1].word.rstrip()[-1:] in ".?!"):
            groups.append(cur)
            cur = []
        cur.append(w)
        last_seg = seg
    if cur:
        groups.append(cur)
    return groups


def caption_events(groups: list[list[Word]], style: CaptionStyle) -> list[tuple[float, float, str]]:
    """(start, end, ASS text) events. Animated: one event per word, with that word
    highlighted and slightly enlarged; the phrase stays up until the next one."""
    hi = ass_color(style.highlight)
    events = []
    for gi, g in enumerate(groups):
        g_end = g[-1].end
        if gi + 1 < len(groups) and groups[gi + 1][0].start - g_end < 0.3:
            g_end = groups[gi + 1][0].start  # no flicker between close phrases
        tokens = [ass_escape(w.word.upper() if style.uppercase else w.word) for w in g]
        if not style.animated:
            events.append((g[0].start, g_end, " ".join(tokens)))
            continue
        for i in range(len(g)):
            a = g[i].start
            b = g[i + 1].start if i + 1 < len(g) else g_end
            if b <= a:
                continue
            text = " ".join(
                f"{{\\c{hi}\\fscx112\\fscy112}}{t}{{\\r}}" if j == i else t
                for j, t in enumerate(tokens))
            events.append((a, b, text))
    return events


def build_ass(path: Path, segs: list[Segment], t0: float, t1: float,
              size: tuple[int, int], style: CaptionStyle, captions: bool = True,
              caption_y: float | None = None, hook: str | None = None,
              hook_seconds: float = 3.0) -> None:
    """Write the ASS file for one clip covering source time [t0, t1].

    size is the output frame (e.g. 1080x1920); caption_y is where the caption
    line is centred, as a share of the frame height (default: lower third).
    """
    w, h = size
    fs = max(int(h * style.size_ratio), 18)
    hfs = max(int(h * style.hook_size_ratio), 16)
    outline = max(fs // 12, 2)
    y = int(h * (caption_y if caption_y is not None else (0.70 if h > w else 0.85)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style.font},{fs},{ass_color(style.color)},&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},2,5,{int(w*0.06)},{int(w*0.06)},0,1
Style: Hook,{style.font},{hfs},&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,{max(hfs//4, 4)},0,8,{int(w*0.08)},{int(w*0.08)},{int(h*0.06)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(hook_seconds, t1 - t0))},Hook,,0,0,0,,"
                     f"{ass_escape(hook)}")
    if captions:
        groups = group_words(words_in_window(segs, t0, t1), style.max_words)
        for a, b, text in caption_events(groups, style):
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Caption,,0,0,0,,"
                         f"{{\\pos({w // 2},{y})}}{text}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
