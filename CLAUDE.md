# CLAUDE.md

Guidance for AI coding agents (and humans) working on this repo.

## What this is

An AI clip finder for gaming streams: a long VOD plus its transcript goes in, and ranked
9:16 short clips come out (TikTok / Shorts / Reels). Claude picks the moments. ffmpeg cuts
them. `README.md` covers usage, `TODO.md` what's left.

## Layout

- `clipper.py` and `feedback.py` are thin entry points. All code lives in `clipfinder/`:
  - `transcript.py`: parsers (.srt/.vtt/Whisper .json/.txt) and chunking
  - `transcribe.py`: audio extraction and Whisper
  - `llm.py`: the prompt, Claude calls, batching, refusal fallback, rescoring
  - `clips.py`: the Clip record, snapping/cleaning, the `clips.json` manifest
  - `render.py`: the ffmpeg commands
  - `captions.py`: the ASS captions and hook overlay
  - `layout.py`: per-streamer facecam layouts
  - `signals.py`: audio and chat signals
  - `feedback.py` / `feedback_cli.py`: the library of posted clips, prompt examples and the re-ranker
  - `fetch.py`: `--url` downloads
  - `sounds.py`: optional PANNs sound tagging (laughter, gunfire, sirens...)
  - `stats.py`: token cost and stage timing per run
  - `storage.py`: `--upload`/`--fetch` for S3 and R2, per-user folders
  - `db.py`: recording runs and clips in Supabase (tables in `supabase/migrations/`)
  - `web.py` + `static/index.html`: the local web UI (`ui.py`)
  - `cli.py`: options and the pipeline
- Tests are in `tests/`, one file per area.

## Setup and commands

- Python **3.12** in `.venv` (3.15 can't load the SDK's pydantic yet):
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest`
- Tests: `.venv/bin/python -m pytest -q`. They need no API key, ffmpeg or network. Keep it
  that way: fake the Anthropic client, yt-dlp and chat-downloader the way `tests/test_llm.py`
  and `tests/test_fetch.py` do.
- Run the full suite before every commit. Every bug fix gets a test that fails without the fix.
- The API key is `ANTHROPIC_API_KEY`, kept in the workspace `.env` one level up. It isn't
  loaded automatically. Never print it, commit it or paste it into output.

## Rules

- **Never commit generated or third-party content:** `clips*/`, `*.transcript.json`,
  `*.loudness.json`, `*.16k.flac`, `library.json`, videos, `.venv/`. They're in `.gitignore`.
  Transcripts and the library contain other people's stream content.
- **Models:** the default is `claude-sonnet-5-5`, the refusal retry goes to `claude-opus-5-5`,
  and the cheap first pass uses `claude-haiku-5-5`. Use exact model IDs. Don't change defaults
  without measuring on real streams.
- **Claude API shape** (checked against the live API): structured output via
  `output_config.format`, effort in `output_config.effort`, server-side
  `fallbacks: "default"` via `client.beta.messages.create` with the
  `server-side-fallback-2026-07-01` beta (Sonnet/Opus/Fable only, never Haiku, never in
  batches). Prompt caching only pays off above 512 tokens.
- **ffmpeg:** Homebrew's ffmpeg has no libass. Anything that burns in text must get its binary
  from `render.ffmpeg_for(True)`, which falls back to the bundled `imageio-ffmpeg`.
- **Timing:** clip `start`/`end` in `clips.json` are the *content* range. Cuts add
  `--pad-before`/`--pad-after`. Length limits (`--min-len`/`--max-len`) apply to the padded
  clip. Snapping moves edges outward up to 3 s and inward at most 1 s; never trim the build-up.
- **Keep it dependency-light:** pure Python where reasonable (the re-ranker needs no numpy).
  New dependencies that only some features need are optional imports with a clear install
  message.
- **Don't break the CLI:** existing flags keep working, and old `clips.json` files must
  still load (`Clip.from_dict` tolerates missing optional fields).
- **User-facing text** (help, prints, README): plain, short sentences that say what happens and
  what to do next.

## Git and PRs

- One fix or feature per branch and PR, with a short description of the bug, the fix and how
  it was tested.
- Commits end with a `Co-Authored-By:` line for the agent that wrote them.
