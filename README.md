# Stream Clipper

Give it a long gameplay video and its transcript. Claude reads the transcript, picks the best
clip-worthy moments (chases, wasted/busted moments, big reactions, funny bits), and ffmpeg cuts
them out as short clips for TikTok, Shorts and Reels. Loudness spikes, chat bursts and a sound
model (laughter, gunfire, explosions, sirens...) help it find moments nobody talks about. The
default prompt is tuned for GTA; edit it in the web UI or use `--prompt-file` for other games.

Use it from the command line (`python clipper.py`) or the web UI (`python ui.py`).

## Setup

1. Install Python 3.10+ (3.10–3.12 recommended; the SDK's pydantic dependency doesn't load on
   3.15 yet) and [ffmpeg](https://ffmpeg.org/download.html) (`brew install ffmpeg`).
   Homebrew's ffmpeg can't burn in text (no libass); the `imageio-ffmpeg` package in
   requirements.txt bundles one that can, and it's used automatically for captions and hooks.
2. Make a venv and install: `python3.12 -m venv .venv && source .venv/bin/activate &&
   pip install -r requirements.txt`. Optional extras:
   - `mlx-whisper` (Apple Silicon) or `faster-whisper`: transcribe videos with no captions
   - `yt-dlp chat-downloader`: `--url` downloads
   - `panns-inference`: sound detection (see below)
   - `streamlink`: leaves Twitch ads out of `--live` recordings
3. Get an API key from https://console.anthropic.com and set it:
   - Mac/Linux: `export ANTHROPIC_API_KEY=sk-ant-...`
   - Windows (PowerShell): `$env:ANTHROPIC_API_KEY="sk-ant-..."`
   - Web UI only: or put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in this folder
     (git-ignored); `ui.py` reads it.

## Usage

```bash
# Straight from a Twitch VOD or YouTube URL: downloads video, captions and chat
python clipper.py --url https://www.twitch.tv/videos/123456789 --streamer alice --burn-captions

# Long VOD? Download only the audio, then only the video around each clip (see below)
python clipper.py --url https://www.twitch.tv/videos/123456789 --audio-first --burn-captions

# Clip a stream while it's live; clips appear a few minutes after each moment (Ctrl+C to stop)
python clipper.py --live https://www.twitch.tv/alice --vertical --burn-captions

# Basic: find clips and cut them into ./clips
python clipper.py --video stream.mp4 --transcript stream.srt

# No transcript yet? Whisper makes one (pip install mlx-whisper or faster-whisper).
# It's reused on later runs; --retranscribe forces a new one.
python clipper.py --video stream.mp4 --transcribe

# Cheapest workflow: find clips, render 480p previews, then render only the keepers
python clipper.py --video stream.mp4 --transcript stream.srt --preview
#   ...review the previews (easiest in the web UI: Keep / Reject)...
python clipper.py --clips-json clips/clips.json --only 2,5,7 --vertical --burn-captions

# Not in a hurry? Batch API (half price) with a Haiku first pass that Sonnet rescores
python clipper.py --video stream.mp4 --transcript stream.srt --batch --triage-model claude-haiku-5-5

# Give the AI context
python clipper.py --video stream.mp4 --transcript stream.srt --context "Streamer is doing the first heist mission"

# The full Shorts look: facecam + gameplay split, word-by-word captions, hook banner,
# with chat helping to find moments the transcript can't see
python clipper.py --video stream.mp4 --transcribe --chat chat.json \
  --streamer alice --burn-captions --hook-overlay
```

## Web UI

Prefer clicking to flags? Run `python ui.py` from the venv and open http://localhost:8000
(`--port 9000` for another port). It only listens on your own machine.

- **New run:** start from a Twitch/YouTube link (with an **Audio first** switch, on by default),
  a video in the project folder, a channel that's **Live** now, or a saved `clips.json`
  (**Render** with new settings, no AI cost). The main options are switches; the rest (notes
  for Claude, clip count and length, padding, effort, Batch API, sound detection, output
  folder) are under More options. The log streams live and a run can be stopped.
- **Live:** paste a channel link and press Start live clipping. The run card shows how much of
  the stream has been checked and how many clips were found and cut; the session's folder gets
  a red dot, and clips appear in it as they're cut, ready to Keep/Reject while the stream is
  still on. **Stop & finish** checks the last few minutes and cuts the last clips (press again
  to force stop). A session that stops without finishing (e.g. the computer slept) is marked
  so you can cut what's left with Render.
- **Clips:** every output folder (and each live session in it) gets a tab and every clip a card with its score, category,
  hook, timestamps, real length and why Claude picked it, plus Download and Copy caption.
  Clips listed in `clips.json` but not cut yet get a **Cut them now** button.

It runs `clipper.py` under the hood, so the UI and the command line always behave the same.
It reads `ANTHROPIC_API_KEY` from `.env` or `../.env` if it isn't set.

- **Review:** on each clip, Keep or Reject it and nudge its start/end by a second. Choices are
  saved in `clips.json` (`review`, plus `original_start`/`original_end` when trimmed). **Render
  approved** cuts only the kept clips at full quality (`--only 2,5,7` on the command line).
- **Results:** after posting a clip, **Log results** on its card saves its views and average
  watch time per platform (update them as they grow); the card then shows them. The
  **Results** button lists every posted clip, best first, imports a CSV of stats, and trains
  the re-ranker once 30 clips have results. Same `library.json` / `ranker.json` as
  `python feedback.py` (see [Learning from posted clips](#learning-from-posted-clips)).
- **Cam layout:** if the streamer cam box is off, **Cam layout** in a folder's toolbar shows a
  frame from the stream with the box on it: drag or resize it, pick cam on top or bottom and
  its size, watch the 9:16 preview, then save (as the streamer's layout when the run knows
  the streamer) and optionally re-render the clips.
- **Models:** More options picks the Claude model (Sonnet 5.5, or Opus 5.5 at twice the
  price), an optional Haiku **first pass** (it reads the whole stream; the main model only
  re-scores its best finds, much cheaper on long streams), and the Whisper model used when a
  stream has no captions.
- **Prompt:** the Prompt button edits what Claude looks for. It's saved to `prompt.txt` and
  every new run uses it (`--prompt-file prompt.txt`); Reset goes back to the built-in one.
- **Cost and time:** every run prints its time per stage and Claude cost, and records them in
  `clips.json` under `runs`. The UI shows the total spent and cost per clip for each folder.
  Prices live in `clipfinder/stats.py`.

## Sound detection

The loudness scan finds loud moments; the sound model says what they are: laughter, screaming,
gunfire, explosions, sirens, crashes and skidding. Claude sees notes like
`[823.0-825.0] sounds like gunfire (65% likely) (no speech)` next to the transcript, so it can
find action nobody talks over. It runs PANNs locally (free, ~80x real time on an M-series Mac,
so about a minute per hour of video, cached in `VIDEO.sounds.json`).

`pip install panns-inference` to turn it on; the ~330 MB model downloads to `~/panns_data` the
first time. `--no-sound-tags` (or the switch in the web UI) turns it off.

**Music warning.** GTA's radio plays licensed songs: Twitch mutes them in VODs and deletes
clips with them, and YouTube/TikTok can claim the clip. The sound model also scores music, so
Claude is told where music plays (and to prefer moments without it), each clip records how
much of it has music (`"music": 0.6` in `clips.json`), and clips with 25% or more get a
**♪ Music** badge in the web UI. The model can't tell the radio from the game's own score, so
treat the badge as "check this before posting". Tested: a radio song mixed under a streamer
talking is caught at -6 to -18 dB; speech-only stream audio isn't flagged.

## Streamer cam (facecam) layouts

The Shorts look: the streamer's webcam on top, gameplay below, captions on the seam.

**Automatic:** `--facecam auto` (the **Streamer cam** switch in the web UI, on by default)
finds the cam for you. A few frames go to Claude, which says whether there's a webcam of
the streamer and where (one call, under a cent). If there is, clips get the split layout; if
not, a plain 9:16 crop. The answer is stored in `clips.json`, so re-renders and live clips
don't ask again, and saved as the streamer's layout in `layouts.json`, so their next streams
don't either. `--no-facecam` (switch off) leaves the cam out even if a layout is saved.
Each clip card in the web UI also has a **Streamer cam** button: it turns the cam on or off
for that clip alone and re-renders it (about 10 seconds). The choice is saved in
`clips.json` (`"facecam": true/false` on the clip) and beats the run's setting from then on.

```bash
python clipper.py --url https://www.twitch.tv/videos/123456789 --facecam auto --burn-captions
```

Works with every source: a video, a URL, `--audio-first` (it looks at the downloaded clip
sections) and `--live`. If the box is off, fix it in `layouts.json` or set it by hand:

**By hand:** every streamer puts their facecam somewhere different, so each one gets a saved
layout in `layouts.json`. Boxes are fractions of the frame, so one layout works for any
resolution.

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
  If YouTube rate-limits the caption download (HTTP 429), the run carries on and uses Whisper.
- **the chat replay** with [chat-downloader](https://github.com/xenova/chat-downloader), used for
  chat bursts. `--no-chat` skips it. If chat can't be fetched, the run continues without it.

The stream title becomes `--context` and the channel name becomes `--streamer` (for the facecam
layout and the feedback library) unless you give your own. Files already downloaded are reused,
so re-running on the same URL is cheap. Only clip streams you have permission to use.

### Audio first: skip downloading the whole VOD

Finding clips only needs sound: Whisper, loudness and sound detection all listen, and captions
and chat come separately. With `--audio-first`, the clipper downloads the audio track
(roughly 60-70 MB per hour, against several GB of 1080p video), finds the clips from it, and
then downloads only the video around each clip into `segments/`, 8 sections at a time.

- Sections line up exactly with the audio, so clip times and captions don't drift.
- `clips.json` records the source URL instead of a video. Re-renders (`--clips-json`, or
  **Render approved** in the web UI) download only what the chosen clips need. A trim past the
  downloaded footage fetches just the missing piece. Sections already on disk are reused.
- Each section has 10 s of extra footage either side (`--segment-pad` to change it).
- With **Find only** (`--dry-run`), no video is downloaded at all until you render.
- **When it's slower:** YouTube sends sections at under 2x playback speed (sections run in
  parallel, so 15 clips take about a minute), while it sends a whole video very fast. For short
  videos, downloading the whole thing is quicker. For multi-hour streams, audio first wins.
- It needs the source to stay online until you've rendered. `--grab-frame` needs the whole
  video, so set up facecam layouts from a normal download.

The web UI's **Link** tab has an **Audio first** switch, on by default.

## Live streams

`--live URL` clips a Twitch, Kick or YouTube stream while it's on:

```bash
python clipper.py --live https://www.twitch.tv/alice --vertical --burn-captions --whisper-model turbo
```

- ffmpeg records the stream in 10-second chunks. Every 30 s of new recording is transcribed
  with Whisper and scanned for loudness spikes and sounds; chat is read live.
- Every 3 minutes, Claude reads the newest few minutes (with 2 minutes of earlier context) and
  picks moments. A chat burst, loudness spike or a scream, laugh or explosion triggers a look
  sooner (at most once a minute).
- A moment only counts once there's a minute of stream after it, because the reaction is often
  the best part. So **clips appear about 2-4 minutes after the moment**.
- Clips are cut in the background, so transcription keeps up. Moments Claude sees twice (each
  look overlaps the last) are only clipped once.
- Ctrl+C stops recording, checks the last stretch, and waits for the last cuts. The run also
  ends by itself when the stream goes offline (or sends no video for 2 minutes), and reconnects
  if the connection drops.

Each session gets its own folder, `OUT/live_<channel>_<date-time>/`, with clips.json, the
transcript, the clips, and `chunks/` (the recording). Only the last 15 minutes of chunks are
kept (each clip's footage is copied to `segments/` first); `--keep-recording` keeps it all.
The folder shows up in the web UI like any other run, so you can review clips while the
stream is still on and re-render them with different settings later.

**Cost and limits:** about $0.03 per 7 minutes of stream with Sonnet in testing, so roughly
$1-2 for a 6-hour stream (`--triage-model claude-haiku-5-5` for less). The whole video is
recorded as it airs (about 3 GB/hour at 1080p60; `--max-height 720` roughly halves it), and
the clipper must run for the whole stream. It can't clip what aired before it started.
Only clip streams you have permission to use.

**Twitch ads:** Twitch inserts ads into the stream for viewers who aren't logged in (about
90 s when you join, and whenever the streamer runs ads). Recorded as-is, they come out as
broken video (ffmpeg prints errors like `error while decoding MB`). With
`pip install streamlink`, live mode records Twitch through streamlink, which leaves the ads
out; the recording just pauses for them and the clock stays in line with chat. Kick streams
are recorded through streamlink too (when installed); live chat isn't read for Kick. Without it,
the ads are recorded and a clip that runs into one shows broken video. The recorder's own
messages go to `chunks/recording.log`.

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
  and transcript it came from. This is the "recipe": `--clips-json` re-renders from it. It also
  holds your review choices and trims, and a `runs` list with each run's time and cost.
- `previews/`: 480p previews when you use `--preview`
- `segments/`: with `--save-segments`, the footage around each clip (30s either side, stream
  copied). `clips.json` lists them, so `--clips-json` can re-render after the VOD is deleted.

## Useful options

| Option | Default | What it does |
|---|---|---|
| `--url` | none | Twitch VOD / YouTube URL: downloads video, captions and chat |
| `--live` / `--keep-recording` | none / off | Clip a live stream as it happens / keep the whole recording |
| `--audio-first` | off | With --url, download the audio to find clips, then only the video around each clip |
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
| `--only` | all | Cut only these clip numbers from the list, e.g. `2,5,7` (files keep their numbers) |
| `--dry-run` | off | Find clips and save `clips.json` without cutting any video |
| `--reencode` | off | Re-encode even when a stream copy would do |
| `--keep-audio` | off | Keep the 16 kHz audio extracted for Whisper |
| `--save-segments` / `--segment-pad` | off / 30 (10 with --audio-first) | Keep the footage around each clip in `segments/` so the VOD can be deleted |
| `--from-segments` | off | Re-render from saved segments even if the VOD still exists (automatic when it's gone) |
| `--upload` / `--upload-vod` | off | Upload the run to S3 or R2 (`clips/`, `recipes/`; `vods/` too). See *Running it as a service* |
| `--fetch RUN` | none | Download a run from the bucket into `clips/RUN/` to re-render it (add `--user` for a user's run) |
| `--stream` / `--links` | file name + suffix / off | Run name in the bucket; print 7-day download links after `--upload` |
| `--user USER_ID` | `CLIPPER_USER` | Upload into that user's folder and record the run in Supabase |
| `--facecam auto` / `--no-facecam` | off | Find the streamer's cam and put it above the gameplay / never add it |
| `--streamer` / `--layouts` | none / layouts.json | Use a saved facecam layout (see above) |
| `--burn-captions` | off | Word-by-word captions; the spoken word lights up |
| `--caption-style` | animated | `plain` shows phrases without the highlight |
| `--caption-color` / `--highlight-color` | FFFFFF / FFE500 | Caption colours (RRGGBB) |
| `--caption-font` / `--caption-case` | Arial / upper | Font and casing |
| `--hook-overlay` / `--hook-seconds` | off / 3 | Show the AI's hook text at the top of the clip |
| `--chat` / `--chat-offset` | none / 0 | Twitch chat replay, and seconds to line it up with the video |
| `--no-audio-signals` / `--audio-threshold` | on / 8 | Loudness-spike scan, and how many dB counts as a spike |
| `--no-sound-tags` | on | Turn off sound detection (laughter, gunfire, explosions...) |
| `--api-jobs` / `--jobs` | 4 / 4 | Parallel Claude calls / parallel ffmpeg cuts |

## How it works

1. **Transcript.** Loads yours, or extracts a 16 kHz mono audio track (about 1 MB a minute) and
   runs Whisper on it: on the Apple Silicon GPU with mlx-whisper, otherwise faster-whisper on the
   CPU with voice-activity detection.
2. **Signals.** ffmpeg measures loudness every second; moments much louder than the two
   minutes around them become audio spikes (cached in `<video>.loudness.json`). The sound model
   labels what's happening (cached in `<video>.sounds.json`). With `--chat`,
   10-second windows where chat is much busier than usual become chat bursts, with the most
   repeated messages. Download chat with
   [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader) (`chatdownload`, JSON) or
   [chat-downloader](https://github.com/xenova/chat-downloader) (JSON); `[0:12:34] user: msg`
   text also works.
3. **Find.** Splits the transcript into 15-minute overlapping sections and sends them to Claude in
   parallel (or as one batch), with that section's audio spikes, sounds and chat bursts listed
   under the transcript, so silent action still gets found. Claude returns clips as structured data.
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
7. **Report.** Prints how long each stage took and what Claude cost (tokens priced per model,
   batch discount and caching included), and appends it to `runs` in `clips.json`.

### Code layout

| Module | What's in it |
|---|---|
| `clipfinder/live.py` | `--live`: recording in chunks, rolling transcription, Claude every few minutes |
| `clipfinder/fetch.py` | `--url`: download video (or audio first, then clip sections), captions and chat |
| `clipfinder/transcript.py` | Transcript parsers and chunking |
| `clipfinder/transcribe.py` | Audio extraction, Whisper, and filtering Whisper's invented filler |
| `clipfinder/llm.py` | Prompt, Claude calls, batch, refusal fallback, rescoring |
| `clipfinder/clips.py` | Clip snapping/cleaning and the `clips.json` manifest |
| `clipfinder/render.py` | ffmpeg: stream copy, re-encode, previews |
| `clipfinder/captions.py` | Word-by-word captions and the hook banner (ASS) |
| `clipfinder/layout.py` | Per-streamer facecam layouts and the split-screen filter |
| `clipfinder/facecam.py` | `--facecam auto`: finding the streamer's cam with Claude |
| `clipfinder/signals.py` | Audio loudness spikes and chat bursts, and the notes Claude sees |
| `clipfinder/sounds.py` | Sound detection with PANNs (laughter, gunfire, explosions...) |
| `clipfinder/stats.py` | Run timing, token usage and cost (prices per model live here) |
| `clipfinder/web.py`, `clipfinder/static/index.html` | The web UI (`ui.py`) |
| `clipfinder/feedback.py` | Library of posted clips, prompt examples, re-ranker |
| `clipfinder/feedback_cli.py` | The `feedback.py` command |
| `clipfinder/cli.py` | Command-line options and the pipeline |

Tests: `pip install pytest && pytest` (no API key needed; the one test that needs ffmpeg is
skipped without it).

## Costs

Claude costs cents per stream: a one-hour stream is roughly 10K input tokens. Every run prints
its actual cost and cost per clip, and the web UI shows the total per folder. `--effort low`,
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
   | `vods/<user>/<run>.mp4` | source video (only with `--upload-vod`) | 1 day |
   | `clips/<user>/<run>/` | this run's clips, `.srt`, `previews/`, `segments/`, `clips.json` | 30 days |
   | `recipes/<user>/<run>/` | a copy of `clips.json` and the transcript (about 50 KB) | never |

   `<user>/` is there only with `--user` (see *Users and Supabase* below). Each user's files sit
   in their own folder, and the retention prefix stays first so one lifecycle rule covers everyone.

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
# (the upload prints the run's name, e.g. alice_0412-4f9a2c)
python clipper.py --fetch alice_0412-4f9a2c          # -> clips/alice_0412-4f9a2c/
python clipper.py --clips-json clips/alice_0412-4f9a2c/clips.json --streamer alice --burn-captions --upload
```

A run's name in the bucket is the video's file name (letters, digits, `-` and `_`) plus a random
suffix, so two uploads of `stream.mp4` never overwrite each other; `--stream NAME` picks one
yourself. It's saved in `clips.json`, so re-uploads go to the same place. Only the files
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

### Users and Supabase

R2 holds the files; a database says who owns them. The tables are in
[supabase/migrations/](supabase/migrations/):

| Table | What it holds |
|---|---|
| `profiles` | one row per signed-up user (Supabase Auth makes the user; a trigger adds the profile) |
| `runs` | each uploaded run: owner, video, length, cost. Its `id` is its folder, `<user>/<run>` |
| `clips` | each clip in a run: title, score, times, R2 key, size, review choice |
| `credit_ledger` | every credit change (+6 free on sign-up, +purchases, -renders); `credit_balance` sums it |

`spend_credits(user, amount, run)` checks the balance and deducts in one step, so two renders at
the same moment can't both spend the last credit. Users can read only their own rows; only the
server writes, with the service-role key.

Set it up:

1. Create a project at https://supabase.com. In the SQL editor, run the migration file (or
   `supabase db push` with the Supabase CLI).
2. Add to `.env` (server only; the service-role key can read and write everything):

   ```
   SUPABASE_URL=https://PROJECT.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=...
   ```
3. Upload under a user: `--user` takes the user's id from Supabase Auth (Authentication →
   Users). The run and its clips are recorded after the upload:

   ```bash
   python clipper.py --video stream.mp4 --transcribe --preview --upload --user 3f1c9a2e-...
   # uploaded 12 file(s) to clips/3f1c9a2e-.../stream-4f9a2c/ ...
   # recorded run 3f1c9a2e-.../stream-4f9a2c and 9 clip(s) in Supabase
   python clipper.py --fetch stream-4f9a2c --user 3f1c9a2e-...
   ```

   `CLIPPER_USER` in `.env` sets a default. Without Supabase settings the upload still works and
   says the run wasn't recorded. A database error never undoes an upload; re-uploading records it.

### Accounts: sign-in and linked channels

The web UI has a placeholder account page at http://localhost:8000/account: sign up, sign in,
set a display name, link YouTube and Twitch channels, and delete the account. The server signs
in for the browser and keeps the session in HttpOnly cookies, so no key or token reaches the page.

1. Run [supabase/migrations/20261113000000_accounts.sql](supabase/migrations/) after the first
   migration. It adds `display_name` to `profiles` and a `linked_accounts` table. Signed-in
   users can read their links but never the OAuth tokens.
2. Add the publishable key: `SUPABASE_ANON_KEY=...` (Project Settings → API).
3. In Supabase, under Authentication → URL Configuration, add `http://localhost:8000/account` as a
   redirect URL so email confirmation links come back to the account page.
4. To link channels, create OAuth apps and add their keys. Register the redirect URL
   `<CLIPPER_PUBLIC_URL>/api/connect/youtube/callback` (or `/twitch/callback`) with each:

   ```
   GOOGLE_CLIENT_ID=...        # Google Cloud console: OAuth client, "Web application",
   GOOGLE_CLIENT_SECRET=...    # with the YouTube Data API v3 and YouTube Analytics API enabled
   TWITCH_CLIENT_ID=...        # dev.twitch.tv/console/apps
   TWITCH_CLIENT_SECRET=...
   CLIPPER_PUBLIC_URL=http://localhost:8000   # where the app is served (https in production)
   CLIPPER_SECRET=...          # any long random string; signs the linking requests
   ```

Deleting an account deletes the user's `clips/`, `recipes/` and `vods/` folders in the bucket
first, then the Supabase user, which removes their profile, runs, clips, credits and links. If
the files can't be deleted, the account is kept so nothing is left without an owner.

Folders keep users' files apart, but they don't protect them: the R2 keys can read everything.
Users never get those keys. Your app checks that the signed-in user owns a run, then makes a
short-lived link for its files.

## Ideas for what to add next

- A benchmark: score the finder against a VOD's most-viewed Twitch community clips, so prompt
  and model changes are measured instead of guessed.
- Send Claude a few frames from each candidate clip when rescoring, to catch visual moments.
- Detect GTA HUD events (wanted stars, "Wasted") from the video frames.
