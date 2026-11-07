# GTA 6 Transcript Clipper

Give it a video and its transcript. Claude reads the transcript, picks the best
clip-worthy moments (chases, wasted/busted moments, big reactions, funny bits),
and ffmpeg cuts those moments out of the video.

## Setup

1. Install Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (it needs to be on your PATH).
2. `pip install -r requirements.txt`
3. Get an API key from https://console.anthropic.com and set it:
   - Mac/Linux: `export ANTHROPIC_API_KEY=sk-ant-...`
   - Windows (PowerShell): `$env:ANTHROPIC_API_KEY="sk-ant-..."`

## Usage

```bash
# Basic: find clips and cut them into ./clips
python clipper.py --video stream.mp4 --transcript stream.srt

# Vertical 9:16 with burned-in captions, top 10 clips
python clipper.py --video stream.mp4 --transcript stream.srt --vertical --burn-captions --top 10

# No transcript yet? Make one with Whisper (pip install faster-whisper)
python clipper.py --video stream.mp4 --transcribe

# Review first: find clips but don't cut
python clipper.py --video stream.mp4 --transcript stream.srt --dry-run
#   ...edit clips/clips.json (delete clips, tweak times)...
python clipper.py --video stream.mp4 --transcript stream.srt --clips-json clips/clips.json

# Give the AI context
python clipper.py --video stream.mp4 --transcript stream.srt --context "Streamer is doing the first heist mission"
```

## Transcript formats

- `.srt` / `.vtt` subtitle files (YouTube, Premiere, CapCut, etc.)
- Whisper `.json` (`{"segments": [{"start", "end", "text"}]}`)
- `.txt` with lines like `[00:01:23] text` or `[83.2 - 87.9] text`

## Output

For each clip in the output folder:
- `01_five_stars_in_10_seconds.mp4`: the clip
- `01_five_stars_in_10_seconds.srt`: captions re-timed to the clip
- `clips.json`: every clip's times, score, title, hook text and why it was picked

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--top` | 15 | Max clips to keep (0 = all) |
| `--min-score` | 6 | Drop clips the AI scored below this (1–10) |
| `--min-len` / `--max-len` | 12 / 75 | Clip length limits in seconds |
| `--pad-before` / `--pad-after` | 1.5 / 1.5 | Extra seconds around each clip |
| `--window` / `--overlap` | 900 / 90 | How the transcript is split for long streams |
| `--model` | claude-sonnet-5-5 | Any Claude model (or set `CLIPPER_MODEL`) |

## How it works

1. Loads the transcript and splits long streams into ~15-minute overlapping sections.
2. Sends each section to Claude with a GTA-specific editor prompt and gets back clips as structured data
   (start, end, score, title, hook, category, reason).
3. Snaps clip edges to sentence boundaries, enforces length limits, removes overlaps
   (keeping the higher score) and keeps the top N.
4. Cuts each clip with ffmpeg (frame-accurate re-encode), optionally cropping to 9:16 and burning captions.

## Ideas for what to add next

- Use Twitch chat activity and audio loudness spikes to boost clip scores (the transcript alone misses silent action).
- A facecam + gameplay split layout instead of a center crop.
- Detect GTA 6 HUD events (wanted stars, "Wasted") from the video frames.
