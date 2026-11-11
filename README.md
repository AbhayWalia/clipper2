# Stream Clipper

Give it a long gameplay video and its transcript. Claude reads the transcript, picks the best
clip-worthy moments (chases, wasted/busted moments, big reactions, funny bits), and ffmpeg cuts
them out as short clips for TikTok, Shorts and Reels. The default prompt is tuned for GTA; use
`--prompt-file` for other games.

## Setup

1. Install Python 3.10+ (3.10–3.12 recommended; the SDK's pydantic dependency doesn't load on
   3.15 yet) and [ffmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg`).
   Homebrew's ffmpeg can't burn in text (no libass); the `imageio-ffmpeg` package in
   requirements.txt bundles one that can, and it's used automatically for captions and hooks.
2. Make a venv and install: `python3.12 -m venv .venv && source .venv/bin/activate &&
   pip install -r requirements.txt` (plus `mlx-whisper` for `--transcribe` on Apple Silicon,
   and `yt-dlp chat-downloader` for `--url`).
3. Get an API key from https://console.anthropic.com and set it:
   - Mac/Linux: `export ANTHROPIC_API_KEY=sk-ant-...`
   - Windows (PowerShell): `$env:ANTHROPIC_API_KEY="sk-ant-..."`

## Usage

```bash
# Straight from a Twitch VOD or YouTube URL: downloads video, captions and chat
python clipper.py --url https://www.twitch.tv/videos/123456789 --streamer alice --burn-captions

# Basic: find clips and cut them into ./clips
python clipper.py --video stream.mp4 --transcript stream.srt

# No transcript yet? Whisper makes one (pip install mlx-whisper or faster-whisper).
# It's reused on later runs; --retranscribe forces a new one.
python clipper.py --video stream.mp4 --transcribe

# Cheapest workflow: find clips, render 480p previews, then render only the keepers
python clipper.py --video stream.mp4 --transcript stream.srt --preview
#   ...watch clips/previews/, delete the clips you don't want from clips/clips.json...
python clipper.py --clips-json clips/clips.json --vertical --burn-captions

# Not in a hurry? Batch API (half price) with a Haiku first pass that Sonnet rescores
python clipper.py --video stream.mp4 --transcript stream.srt --batch --triage-model claude-haiku-5-5

# Give the AI context
python clipper.py --video stream.mp4 --transcript stream.srt --context "Streamer is doing the first heist mission"

# The full Shorts look: facecam + gameplay split, word-by-word captions, hook banner,
# with chat helping to find moments the transcript can't see
python clipper.py --video stream.mp4 --transcribe --chat chat.json \
  --streamer alice --burn-captions --hook-overlay
```

## Per-streamer facecam layouts

Every streamer puts their facecam somewhere different, so each one gets a saved layout
in `layouts.json`. Boxes are fractions of the frame, so one layout works for any resolution.

1. Grab a still with a grid on it (white lines every 0.1 of the frame, yellow every 0.05):
   ```bash
   python clipper.py --video alice_vod.mp4 --grab-frame 600     # -> clips/frame_600s.png
   ```
2. Read off the facecam's left edge, top edge, width and height, and save them:
   ```bash
   python clipper.py --video alice_vod.mp4 --streamer alice --facecam 0.72,0.03,0.26,0.30 --save-layout
   ```
   Pixels work too (`--facecam 1382,32,500,324`) and are converted using the video's size.
3. From then on, `--streamer alice` gives a 9:16 clip with the facecam on top (35% of the
   frame) and gameplay below. The streamer is saved in `clips.json`, so re-renders keep it.

Optional, saved with the layout: `--gameplay x,y,w,h` takes gameplay from a smaller area
(for example to keep the facecam out of the gameplay half), `--cam-height 0.3` changes the
split, and `--cam-position bottom` puts the facecam under the gameplay. Without `--streamer`,
`--vertical` still does a plain centre crop.

## Learning from posted clips

Claude guesses what will do well; your posted clips show what actually did. The feedback
layer feeds those results back in three stages. Each stage switches itself on once there's
enough data, so nothing changes until then.

1. **Record results.** Every clip in `clips.json` gets an id, its transcript and the
   measurements the ranker learns from. After posting, record how it did (re-run as the
   numbers grow; several platforms per clip are added together):
   ```bash
   python feedback.py record clips/clips.json 3 --platform tiktok --views 48000 --avg-watch 17.5 --shares 120
   python feedback.py import stats.csv --clips-json clips/clips.json   # many at once
   python feedback.py list                                             # best first
   ```
   CSV columns: `clip` (number or id), `views`, and optionally `platform`, `url`,
   `avg_watch`, `shares`, `likes`, `comments`. Everything goes into `library.json`.
2. **Examples (from 5 clips with stats).** The channel's best performers and a few flops are
   added to Claude's system prompt, with the same streamer's clips first. The prompt is then
   cached, so the extra text costs a fraction of normal input price after the first section.
   Preview them with `python feedback.py examples --streamer alice`; turn them off with
   `--examples 0`.
3. **Re-ranker (from 30 clips; trustworthy from about 200).**
   ```bash
   python feedback.py train
   ```
   This fits a small model that predicts how much better than *that streamer's* typical clip
   a moment will do, from Claude's score, length, speech rate, audio spikes, chat bursts and
   category. It's tested with 5-fold cross-validation against Claude's score alone and only
   switches on (`ranker.json`, `"enabled": true`) if it ranks unseen clips better. When on,
   Claude still finds the candidates; the ranker picks the top N. Retrain as data grows;
   `--no-ranker` turns it off for a run.

Performance is measured relative to each streamer's usual views, so a big channel's clips
don't drown out a small one's. Stats should be at least a few days old (most short-form
views arrive in the first 48 hours), so compare clips at a similar age.

## Fetching from a URL

`--url` takes a Twitch VOD, YouTube video or anything else [yt-dlp](https://github.com/yt-dlp/yt-dlp)
supports (`pip install yt-dlp chat-downloader`). Into `OUT/source/` it downloads:

- **the video** (up to 1080p as mp4; `--max-height 720` for less, which also applies to saved segments),
- **the platform's captions**, when there are any. They're free, so Whisper is skipped. YouTube's
  automatic captions include per-word timing, so animated captions still line up. Twitch VODs
  usually have none, so Whisper runs. `--no-captions` forces Whisper.
- **the chat replay** with [chat-downloader](https://github.com/xenova/chat-downloader), used for
  chat bursts. `--no-chat` skips it. If chat can't be fetched, the run continues without it.

The stream title becomes `--context` and the channel name becomes `--streamer` (for the facecam
layout and the feedback library) unless you give your own. Files already downloaded are reused,
so re-running on the same URL is cheap. Only clip streams you have permission to use.

## Transcript formats

- `.srt` / `.vtt` subtitle files (YouTube, Twitch, Premiere, CapCut, etc.). Captions you already
  have are free; use them instead of Whisper when you can. YouTube's rolling automatic captions
  are de-duplicated, and their per-word timings are used.
- Whisper `.json` (`{"segments": [{"start", "end", "text"}]}`)
- `.txt` with lines like `[00:01:23] text` or `[83.2 - 87.9] text`

## Output

In the output folder:
- `01_five_stars_in_10_seconds.mp4`: the clip
- `01_five_stars_in_10_seconds.srt`: captions re-timed to the clip
- `clips.json`: every clip's times, score, title, hook text and why it was picked, plus the video
  and transcript it came from. This is the "recipe": `--clips-json` re-renders from it.
- `previews/`: 480p previews when you use `--preview`
- `segments/`: with `--save-segments`, the footage around each clip (30s either side, stream
  copied). `clips.json` lists them, so `--clips-json` can re-render after the VOD is deleted.

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--url` | none | Twitch VOD / YouTube URL: downloads video, captions and chat |
| `--no-captions` / `--no-chat` | off | With --url, use Whisper instead of the platform captions / skip chat |
| `--top` | 15 | Max clips to keep (0 = all) |
| `--min-score` | 6 | Drop clips the AI scored below this (1–10) |
| `--min-len` / `--max-len` | 12 / 75 | Clip length limits in seconds, padding included |
| `--pad-before` / `--pad-after` | 1.5 / 1.5 | Extra seconds around each clip |
| `--model` | claude-sonnet-5-5 | Main model (or set `CLIPPER_MODEL`) |
| `--effort` | medium | Claude reasoning effort: low / medium / high / xhigh / max |
| `--triage-model` | off | Cheap first pass (e.g. `claude-haiku-5-5`); `--model` rescores its shortlist |
| `--batch` / `--batch-id` | off | Message Batches API at 50% off; resume an interrupted run with `--batch-id` |
| `--refusal-model` | claude-opus-5-5 | Re-runs a section if `--model` declines it (`none` to disable) |
| `--no-server-fallback` | off | Don't send the API's server-side refusal fallback |
| `--prompt-file` | built-in GTA prompt | Your own system prompt; `{min_len}`/`{max_len}` are filled in |
| `--preview` | off | Render 480p previews instead of full clips |
| `--reencode` | off | Re-encode even when a stream copy would do |
| `--keep-audio` | off | Keep the 16 kHz audio extracted for Whisper |
| `--save-segments` / `--segment-pad` | off / 30 | Keep the footage around each clip in `segments/` so the VOD can be deleted |
| `--from-segments` | off | Re-render from saved segments even if the VOD still exists (automatic when it's gone) |
| `--upload` / `--upload-vod` | off | Upload the run to S3 or R2 (`clips/`, `recipes/`; `vods/` too). See *Running it as a service* |
| `--fetch STREAM` | none | Download a stream's run from the bucket into `clips/STREAM/` to re-render it |
| `--stream` / `--links` | file name / off | Name in the bucket; print 7-day download links after `--upload` |
| `--streamer` / `--layouts` | none / layouts.json | Use a saved facecam layout (see above) |
| `--burn-captions` | off | Word-by-word captions; the spoken word lights up |
| `--caption-style` | animated | `plain` shows phrases without the highlight |
| `--caption-color` / `--highlight-color` | FFFFFF / FFE500 | Caption colours (RRGGBB) |
| `--caption-font` / `--caption-case` | Arial / upper | Font and casing |
| `--hook-overlay` / `--hook-seconds` | off / 3 | Show the AI's hook text at the top of the clip |
| `--chat` / `--chat-offset` | none / 0 | Twitch chat replay, and seconds to line it up with the video |
| `--no-audio-signals` / `--audio-threshold` | on / 8 | Loudness-spike scan, and how many dB counts as a spike |
| `--api-jobs` / `--jobs` | 4 / 4 | Parallel Claude calls / parallel ffmpeg cuts |

## How it works

1. **Transcript.** Loads yours, or extracts a 16 kHz mono audio track (about 1 MB a minute) and
   runs Whisper on it: on the Apple Silicon GPU with mlx-whisper, otherwise faster-whisper on the
   CPU with voice-activity detection.
2. **Signals.** ffmpeg measures loudness every second; moments much louder than the two
   minutes around them become audio spikes (cached in `<video>.loudness.json`). With `--chat`,
   10-second windows where chat is much busier than usual become chat bursts, with the most
   repeated messages. Download chat with
   [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader) (`chatdownload`, JSON) or
   [chat-downloader](https://github.com/xenova/chat-downloader) (JSON); `[0:12:34] user: msg`
   text also works.
3. **Find.** Splits the transcript into 15-minute overlapping sections and sends them to Claude in
   parallel (or as one batch), with that section's audio spikes and chat bursts listed under the
   transcript, so silent action still gets found. Claude returns clips as structured data.
4. **Refusals.** Sonnet 5.5 can decline a section. The API's server-side fallback only retries
   some refusal categories, so a declined section is re-run once on `--refusal-model`; if that
   declines too, the section is skipped and the rest of the run continues.
5. **Clean.** Snaps clip edges to nearby sentence boundaries, enforces length limits (padding
   included), removes overlaps (keeping the higher score) and keeps the top N.
6. **Cut.** With no crop and no overlays, clips are stream-copied (no re-encode, nearly free). The
   cut starts on the keyframe just before the clip and the captions are timed to match. With
   a layout, `--vertical`, captions or a hook it re-encodes, using the Mac's hardware encoder
   when available. Captions and the hook are written to an `.ass` file next to each clip and
   burned in by ffmpeg. Whisper transcripts include per-word timings; for `.srt`/`.vtt` input the
   word timings are estimated from each line, which is close but not exact. Re-run with
   `--transcribe --retranscribe` for exact timings if you made the transcript before this update.

### Code layout

| Module | What's in it |
|---|---|
| `clipfinder/fetch.py` | `--url`: download video, captions and chat |
| `clipfinder/transcript.py` | Transcript parsers and chunking |
| `clipfinder/transcribe.py` | Audio extraction, Whisper, and filtering Whisper's invented filler |
| `clipfinder/llm.py` | Prompt, Claude calls, batch, refusal fallback, rescoring |
| `clipfinder/clips.py` | Clip snapping/cleaning and the `clips.json` manifest |
| `clipfinder/render.py` | ffmpeg: stream copy, re-encode, previews |
| `clipfinder/captions.py` | Word-by-word captions and the hook banner (ASS) |
| `clipfinder/layout.py` | Per-streamer facecam layouts and the split-screen filter |
| `clipfinder/signals.py` | Audio loudness spikes and chat bursts |
| `clipfinder/feedback.py` | Library of posted clips, prompt examples, re-ranker |
| `clipfinder/feedback_cli.py` | The `feedback.py` command |
| `clipfinder/cli.py` | Command-line options and the pipeline |

Tests: `pip install pytest && pytest` (no API key or ffmpeg needed).

## Costs

Claude costs cents per stream: a one-hour stream is roughly 10K input tokens. `--effort low`,
`--batch` and `--triage-model claude-haiku-5-5` cut that further. Before making Haiku triage your
default, compare it with a normal run on a few streams to check it isn't missing good clips.
Nearly all the real cost is video: transcription, encoding and storage.

## Running it as a service: storage

The tool only writes local files. If you host it, the source VOD is about 14× the size of all
the clips made from it, so what you keep of it sets the bill. The recommended setup keeps clips
re-editable for 30 days while storing about a tenth of the VOD:

1. **Keep the footage around each clip, not the VOD.** Run with `--save-segments`. It
   stream-copies each clip plus 30s either side into `segments/` (no re-encode; clips close
   together share one segment), about 1.4 GB per 4-hour stream instead of about 13 GB.
2. **Delete the VOD after a day.** Once segments exist, `--clips-json` re-renders from them on its
   own when the VOD is missing: nudge clip edges (within the 30s), change the layout, captions or
   hook, or render the keepers after `--preview`.
3. **Store everything on Cloudflare R2, in Standard storage.** R2 doesn't charge for downloads;
   on S3, download fees cost more than the storage itself. Avoid infrequent-access and archive
   tiers (S3 Standard-IA, R2 Infrequent Access, Glacier, Wasabi): their 30–90 day minimums and
   per-read fees cost more than they save on files kept 30 days and downloaded.
4. **Upload with `--upload`.** It lays the bucket out by how long things are kept, and
   [storage/lifecycle.json](storage/lifecycle.json) deletes them on schedule:

   | Prefix | What goes there | Deleted after |
   |---|---|---|
   | `vods/<stream>.mp4` | source video (only with `--upload-vod`) | 1 day |
   | `clips/<stream>/` | this run's clips, `.srt`, `previews/`, `segments/`, `clips.json` | 30 days |
   | `recipes/<stream>/` | a copy of `clips.json` and the transcript (about 50 KB) | never |

### Setting up R2 (or S3)

1. **R2:** in the Cloudflare dashboard, turn on R2, create a bucket, and under *Manage API
   tokens* create a token with *Object Read & Write* for that bucket.
   **S3:** create a bucket and an IAM user allowed `s3:GetObject`, `PutObject`, `DeleteObject`
   and `ListBucket` on it.
2. `pip install 'boto3>=1.36'`, then put the settings in a `.env` file (found in the current
   folder or any parent; it's git-ignored) or export them:

   ```
   CLIPPER_BUCKET=clipper
   CLIPPER_S3_ENDPOINT=https://ACCOUNT_ID.r2.cloudflarestorage.com   # leave out for AWS S3
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=auto                                                   # R2; for S3 e.g. us-east-1
   ```
3. Apply the lifecycle rules once (or add them in the R2 dashboard under *Object lifecycle rules*):

   ```bash
   aws s3api put-bucket-lifecycle-configuration --bucket clipper \
     --endpoint-url https://ACCOUNT_ID.r2.cloudflarestorage.com \
     --lifecycle-configuration file://storage/lifecycle.json
   ```

Then:

```bash
# Find clips, keep segments, upload everything, print download links
python clipper.py --video vods/alice_0412.mp4 --transcribe --save-segments --preview \
  --upload --upload-vod --links

# Within 30 days, from any machine: download the run and render the keepers from its segments
python clipper.py --fetch alice_0412                 # -> clips/alice_0412/
python clipper.py --clips-json clips/alice_0412/clips.json --streamer alice --burn-captions --upload
```

The stream's name in the bucket is the video's file name (letters, digits, `-` and `_`), or
`--stream NAME`; it's saved in `clips.json`, so re-uploads go to the same place. Only the files
a run made are uploaded, so a shared `--out` folder doesn't leak other streams' clips. Segments
already in the bucket aren't uploaded again. Download links last at most 7 days (the limit on
both S3 and R2), so for longer access have your app make a fresh link when someone downloads.

Lifecycle deletes run up to a day late. If you promise users exactly 30 days, expire their
download links in the app as well. If bucket versioning is on, also expire noncurrent versions,
or deleted files keep costing money.

At 100 four-hour streams a day this comes to roughly $130/month on R2, about $0.003 per clip
(estimates, not measured). Lowering clip resolution saves little: clips are about a third of that,
and TikTok, Shorts and Reels give 1080p uploads better quality. Two more savings:

- **Use proxies.** `--preview` makes 480p copies for review; only the approved clips are rendered
  at full quality.
- **Upload audio, not video,** for transcription: `--keep-audio` writes the 16 kHz FLAC track
  Whisper uses.

## Ideas for what to add next

- Detect GTA HUD events (wanted stars, "Wasted") from the video frames.
- Detect the facecam box automatically (face detection on a few frames) to speed up layout setup.
- A review page for nudging clip edges and approving clips before the full render.
