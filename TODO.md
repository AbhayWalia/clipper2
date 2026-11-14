# TODO

What's left before the clipper is a working MVP, then what to improve once it is.
Ordered by priority within each section.

## Before MVP

### Validate on real streams
- [ ] Run 3–5 real VODs end to end (`--url` or `--video`), cut and watch every clip. So far
      it has run on synthetic test streams and one small real API call.
- [ ] Compare `--effort medium` (default) with `--effort high` on the same streams. Keep medium
      unless high finds clearly better clips.
- [ ] Compare `--triage-model claude-haiku-5-5` with a normal run. Only make it the default if
      it doesn't miss good clips.
- [ ] Check how often Sonnet declines GTA sections (look for "declined" in the output) and
      whether the Opus retry rescues them.
- [ ] Test `--url` chat download on a real Twitch VOD (chat-downloader is only tested against
      a stand-in) and tune `--chat-offset` if chat bursts land late.
- [ ] Test the CPU path (`faster-whisper`) on a non-Apple machine.

### Setup
- [x] Key in the workspace `.env` renamed to `ANTHROPIC_API_KEY`.
- [ ] **Load `.env` automatically:** nothing reads it yet, so run `set -a; . ../.env; set +a`
      before the clipper, or add `python-dotenv` loading in `clipfinder/cli.py`.
- [ ] Save a facecam layout for each streamer (`--grab-frame`, then `--facecam ... --save-layout`)
      and add a `--gameplay` box where the facecam would show up twice.
- [ ] Decide what to do about the old stream transcripts still in git history. They were
      untracked in this PR but remain in past commits; `git filter-repo` removes them, but it
      rewrites history and needs a force-push.

### Business
- [ ] Get permission from each streamer (or join their clipping program) before posting.
- [ ] Start recording posted-clip stats right away (`python feedback.py record` / `import`).
      Examples switch on at 5 clips, the re-ranker can train at 30 and is trustworthy around 200.
- [ ] Back up `library.json` (it's git-ignored on purpose: it holds stream transcript text).

## After MVP: room for improvement

### Less manual work
- [ ] **Pull stats automatically** from the YouTube Analytics API and TikTok instead of
      `feedback.py record`/`import`. Keeps the learning layer fed without effort. TikTok
      needs app approval.
- [ ] **Review page:** a small web UI to watch the 480p previews, nudge clip edges, approve
      or reject, then render the approved ones. Rejections are useful training data too.
- [ ] **Auto-posting** to YouTube Shorts / TikTok / Reels with the AI's title and hashtags.
      Mostly API approval paperwork.
- [ ] **Detect the facecam box automatically** (face detection on a few frames) so new
      streamers need no setup.
- [ ] **Download only what's needed:** fetch audio and chat first, pick clips, then download
      just the clip sections with yt-dlp (`--download-sections`) instead of the whole VOD.

### Better clips
- [ ] **On-screen game events:** detect GTA HUD changes (wanted stars, "Wasted"/"Busted")
      from sampled frames as another silent-moment signal.
- [ ] **Prompt presets per game** (`--game valorant`, `--game minecraft`, ...) instead of only
      `--prompt-file`.
- [ ] **Smarter re-ranker:** add posting time, platform, hashtags and hook wording as features;
      predict per platform; record rejected clips as negatives.
- [ ] **Reframing for streams without a facecam:** blurred-background fit or subject tracking
      instead of a plain centre crop.
- [ ] Caption styles per streamer (font, colours, position) saved with the layout.

### Running it as a service
- [ ] Job queue + workers (one stream per job) on serverless or spot GPUs instead of one Mac.
- [x] Upload/fetch runs to S3 or Cloudflare R2 (`--upload`, `--fetch`); lifecycle rules in
      `storage/lifecycle.json` (VODs 1 day, clips and segments 30 days, recipes kept).
- [ ] Apply `storage/lifecycle.json` to the real bucket and test `--upload` against it.
- [ ] Download links from the app (signed links last at most 7 days; make one per download).
- [x] Per-user folders in the bucket (`--user`) and Supabase tables for users, runs, clips and
      credits (`supabase/migrations/`); `--upload --user` records each run.
- [ ] Run the migration on the real Supabase project and test `--upload --user` against it.
- [ ] Hosted app: Supabase Auth sign-in, spend credits (`spend_credits`) before a render,
      Stripe purchases into `credit_ledger`, and download links only for the user's own runs.
- [ ] Per-run cost logging (token usage from each response, Whisper minutes, render time) so
      the cost per clip is measured, not estimated.
- [ ] Bundle a caption font: Arial isn't on most Linux servers.

### Code health
- [ ] `pyproject.toml` so it installs as a package with `clipper` / `clipper-feedback` commands.
- [ ] CI (GitHub Actions) running `pytest` on every push; the suite needs no API key or ffmpeg.
- [ ] An end-to-end test that renders a short synthetic clip when ffmpeg is available.
- [ ] Check Windows paths in the `subtitles` filter escaping.
