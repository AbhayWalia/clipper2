"""Command-line entry point: transcript -> Claude -> clips.json -> ffmpeg."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from . import feedback as fb
from . import llm
from .captions import CaptionStyle, build_ass
from .clips import (clean_clips, clip_id, load_manifest, save_manifest, source_fingerprint,
                    update_manifest)
from .layout import DEFAULT_LAYOUTS, Layout, grab_frame, load_layouts, parse_box, save_layout
from .fetch import SECTION_PAD, download_sections
from .render import (SEGMENT_PAD, VERTICAL_SIZE, choose_mode, cut_clip, fmt_srt_time,
                     keyframe_before, pick_encoder, save_segments, segment_for, segment_windows,
                     slugify, video_duration, video_size, write_clip_srt)
from .stats import STATS
from .signals import Signals, audio_events, chat_events, load_chat, measure_loudness
from .storage import MAX_LINK_DAYS, Store, StorageError, load_dotenv, stream_name
from .transcribe import transcribe, transcript_path_for
from .transcript import chunk_segments, load_transcript


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and cut short-form clips from a gameplay stream.")
    io = p.add_argument_group("input / output")
    io.add_argument("--url", help="Twitch VOD / YouTube URL: downloads the video, its captions and chat")
    io.add_argument("--source-dir", type=Path, help="Where --url downloads go (default: OUT/source)")
    io.add_argument("--no-chat", action="store_true", help="With --url, don't download chat")
    io.add_argument("--live", metavar="URL",
                    help="Clip a live stream while it's on (Twitch or YouTube live URL). Records it in "
                         "chunks and cuts clips a few minutes after each moment. Stop with Ctrl+C")
    io.add_argument("--keep-recording", action="store_true",
                    help="With --live, keep the whole recording instead of only the last 15 minutes")
    io.add_argument("--live-readrate", type=float, help=argparse.SUPPRESS)  # replay a file as if live, faster
    io.add_argument("--audio-first", action="store_true",
                    help="With --url, download only the audio (plus captions and chat) to find clips, "
                         "then only the video around each clip. Far less to download for long VODs")
    io.add_argument("--no-captions", action="store_true",
                    help="With --url, ignore the platform's captions and use Whisper")
    io.add_argument("--max-height", type=int, default=1080,
                    help="With --url, max video height to download (saved segments use it too)")
    io.add_argument("--video", type=Path, help="Source video (optional with --clips-json if it's recorded there)")
    io.add_argument("--transcript", type=Path, help=".srt / .vtt / .json / .txt transcript")
    io.add_argument("--transcribe", action="store_true", help="Make a transcript with Whisper first")
    io.add_argument("--retranscribe", action="store_true",
                    help="With --transcribe, redo Whisper even if a .transcript.json exists")
    io.add_argument("--whisper-model", default="small", help="Whisper size for --transcribe (small / turbo / ...)")
    io.add_argument("--keep-audio", action="store_true",
                    help="Keep the extracted 16 kHz audio (.16k.flac) next to the video")
    io.add_argument("--out", type=Path,
                    help="Output folder (default: ./clips, or the folder of --clips-json)")
    io.add_argument("--clips-json", type=Path, help="Skip the AI: cut clips from an existing clips.json")

    ai = p.add_argument_group("clip finding")
    ai.add_argument("--model", default=os.environ.get("CLIPPER_MODEL", llm.DEFAULT_MODEL))
    ai.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium",
                    help="Claude reasoning effort (lower = cheaper, faster)")
    ai.add_argument("--triage-model", default=None,
                    help=f"Cheap first-pass model (e.g. {llm.SUGGESTED_TRIAGE_MODEL}); "
                         "--model then only rescores the best candidates")
    ai.add_argument("--batch", action="store_true",
                    help="Use the Message Batches API: 50%% cheaper, but can take minutes to hours")
    ai.add_argument("--batch-id", help="Resume polling a batch started earlier (same transcript and settings)")
    ai.add_argument("--refusal-model", default=llm.DEFAULT_REFUSAL_MODEL,
                    help="Model to re-run a section on if --model declines it ('none' to disable)")
    ai.add_argument("--no-server-fallback", action="store_true",
                    help="Don't send the API's server-side refusal fallback (fallbacks: \"default\")")
    ai.add_argument("--api-jobs", type=int, default=4, help="Transcript sections to send to Claude at once")
    ai.add_argument("--prompt-file", type=Path,
                    help="Custom system prompt (for other games). {min_len}/{max_len} get filled in")
    ai.add_argument("--context", default="", help="Notes for the AI, e.g. streamer name, what happened")
    ai.add_argument("--top", type=int, default=15, help="Max number of clips (0 = no limit)")
    ai.add_argument("--min-score", type=float, default=6, help="Drop clips scored below this")
    ai.add_argument("--min-len", type=float, default=12, help="Min clip length incl. padding (s)")
    ai.add_argument("--max-len", type=float, default=75, help="Max clip length incl. padding (s)")
    ai.add_argument("--window", type=float, default=900, help="Transcript chunk size (s)")
    ai.add_argument("--overlap", type=float, default=90, help="Chunk overlap (s)")

    fbk = p.add_argument_group("learning from posted clips (see feedback.py)")
    fbk.add_argument("--library", type=Path, default=fb.DEFAULT_LIBRARY,
                     help="Library of posted clips and their stats")
    fbk.add_argument("--examples", type=int, default=8,
                     help=f"Best-performing posted clips to show Claude as examples (0 = off; "
                          f"needs {fb.MIN_EXAMPLES}+ clips with stats)")
    fbk.add_argument("--ranker", type=Path, default=fb.DEFAULT_RANKER,
                     help="Trained re-ranker (used only if it beat Claude's score in testing)")
    fbk.add_argument("--no-ranker", action="store_true", help="Ignore the re-ranker")

    sig = p.add_argument_group("signals beyond speech")
    sig.add_argument("--chat", type=Path,
                     help="Twitch chat replay: TwitchDownloader .json, chat-downloader .json, "
                          "or .txt lines '[0:12:34] user: msg'")
    sig.add_argument("--chat-offset", type=float, default=0.0,
                     help="Seconds to add to chat times so they line up with the video")
    sig.add_argument("--no-audio-signals", action="store_true",
                     help="Don't scan the audio for loudness spikes")
    sig.add_argument("--audio-threshold", type=float, default=8.0,
                     help="dB above the surrounding 2 minutes that counts as a spike")
    sig.add_argument("--no-sound-tags", action="store_true",
                     help="Don't label sounds (laughter, gunfire, explosions, sirens...) with PANNs")

    lay = p.add_argument_group("facecam layout (per streamer)")
    lay.add_argument("--streamer", help="Use this streamer's facecam layout from the layouts file "
                                        "(implies a 9:16 split: facecam + gameplay)")
    lay.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS, help="Layouts file")
    lay.add_argument("--facecam", help="Put the streamer's cam above the gameplay. 'auto' finds it "
                                       "(one Claude call per streamer, then saved), or give the box "
                                       "x,y,w,h as fractions (0.72,0.03,0.26,0.30) or pixels")
    lay.add_argument("--no-facecam", action="store_true",
                     help="Don't add the streamer's cam, even if a layout is saved for them")
    lay.add_argument("--gameplay", help="Gameplay box x,y,w,h (default: whole frame, centre-cropped)")
    lay.add_argument("--cam-height", type=float, help="Share of the 9:16 frame for the facecam (default 0.35)")
    lay.add_argument("--cam-position", choices=["top", "bottom"], help="Facecam above or below gameplay")
    lay.add_argument("--save-layout", action="store_true",
                     help="Save --facecam/--gameplay/--cam-* under --streamer in the layouts file")
    lay.add_argument("--grab-frame", type=float, metavar="SECONDS",
                     help="Save a still at this time with a grid for reading off the facecam box, then exit")

    txt = p.add_argument_group("captions and hook")
    txt.add_argument("--caption-style", choices=["animated", "plain"], default="animated",
                     help="animated = the spoken word lights up (TikTok style)")
    txt.add_argument("--caption-font", default="Arial")
    txt.add_argument("--caption-color", default="FFFFFF", help="RRGGBB")
    txt.add_argument("--highlight-color", default="FFE500", help="RRGGBB for the current word")
    txt.add_argument("--caption-case", choices=["upper", "asis"], default="upper")
    txt.add_argument("--hook-overlay", action="store_true",
                     help="Show the AI's hook text at the top for the first seconds")
    txt.add_argument("--hook-seconds", type=float, default=3.0)

    cut = p.add_argument_group("cutting")
    cut.add_argument("--pad-before", type=float, default=1.5, help="Seconds of lead-in")
    cut.add_argument("--pad-after", type=float, default=1.5, help="Seconds of tail")
    cut.add_argument("--vertical", action="store_true", help="Crop clips to 9:16")
    cut.add_argument("--burn-captions", action="store_true", help="Burn captions into clips")
    cut.add_argument("--preview", action="store_true",
                     help="Render cheap 480p previews into OUT/previews instead of full clips")
    cut.add_argument("--reencode", action="store_true",
                     help="Re-encode even when stream copy would do (frame-accurate start)")
    cut.add_argument("--dry-run", action="store_true", help="Find clips but don't cut video")
    cut.add_argument("--only", help="Cut only these clip numbers from the list, e.g. 1,3,7 "
                                    "(files keep their numbers)")
    cut.add_argument("--jobs", type=int, default=4, help="Clips to cut at the same time")
    cut.add_argument("--encoder", choices=["auto", "h264_videotoolbox", "libx264"], default="auto",
                     help="auto uses the Mac's hardware encoder when available")

    keep = p.add_argument_group("clip segments (keep these instead of the whole VOD)")
    keep.add_argument("--save-segments", action="store_true",
                      help="Stream-copy the footage around each clip into OUT/segments/ and record it "
                           "in clips.json, so clips can be re-rendered after the VOD is deleted")
    keep.add_argument("--segment-pad", type=float, default=None,
                      help=f"Seconds of extra footage kept on each side of a clip (default "
                           f"{SEGMENT_PAD:g}; {SECTION_PAD:g} with --audio-first, which fetches "
                           f"more later if a trim needs it)")
    keep.add_argument("--from-segments", action="store_true",
                      help="With --clips-json, render from the saved segments even if the VOD exists "
                           "(automatic when it doesn't)")

    cloud = p.add_argument_group("cloud storage: S3 or Cloudflare R2 (settings in .env, see README)")
    cloud.add_argument("--upload", action="store_true",
                       help="After the run, upload the output folder to clips/STREAM/ and clips.json "
                            "+ transcript to recipes/STREAM/")
    cloud.add_argument("--upload-vod", action="store_true",
                       help="With --upload, also upload the source video to vods/ (deleted after 1 day)")
    cloud.add_argument("--stream", help="Name of this stream in the bucket (default: the video's file name)")
    cloud.add_argument("--links", action="store_true",
                       help=f"With --upload, print download links for the clips "
                            f"(valid {MAX_LINK_DAYS} days, the most S3/R2 allow)")
    cloud.add_argument("--fetch", metavar="STREAM",
                       help="Download clips/STREAM/ from the bucket into OUT/STREAM/ to re-render it, then exit")
    args = p.parse_args(argv)
    if args.live:
        clash = [flag for flag, on in (("--url", args.url), ("--video", args.video),
                                       ("--clips-json", args.clips_json), ("--transcript", args.transcript),
                                       ("--batch", args.batch), ("--audio-first", args.audio_first)) if on]
        if clash:
            p.error(f"--live can't be combined with {', '.join(clash)}.")
    if args.audio_first and not args.url:
        p.error("--audio-first needs --url (it downloads the video around each clip from there).")
    if args.window <= 0 or not 0 <= args.overlap < args.window:
        p.error(f"--overlap ({args.overlap:g}) must be at least 0 and smaller than "
                f"--window ({args.window:g}).")
    return args


def sound_tags(video: Path) -> list:
    """Laughter/gunfire/explosions... from the sound model; optional, never blocks a run."""
    cache = video.with_suffix(".sounds.json")
    try:
        from .sounds import sound_events, tag_audio
        import panns_inference  # noqa: F401  (checked here so the hint below is accurate)
    except ImportError:
        print("  (sound tags skipped: pip install panns-inference)")
        return []
    print("Labelling sounds (laughter, gunfire, explosions...)" +
          (" (cached)" if cache.exists() else "; about 1 min per hour of video") + "...")
    try:
        events = sound_events(tag_audio(video, cache))
    except Exception as e:  # a bonus signal; never block clipping on it
        print(f"  sound tagging failed ({e}); continuing without it")
        return []
    if events:
        top = Counter(e.detail for e in events).most_common(4)
        print("  heard " + ", ".join(f"{k} x{v}" for k, v in top))
    return events


def build_signals(args, segs) -> Signals | None:
    events = []
    if not args.no_audio_signals:
        cache = args.video.with_suffix(".loudness.json")
        print("Scanning audio for loudness spikes" + (" (cached)" if cache.exists() else "") + "...")
        try:
            events += audio_events(measure_loudness(args.video, cache), args.audio_threshold)
        except Exception as e:  # never block clipping on this
            print(f"  audio scan failed ({e}); continuing without it")
    if not args.no_sound_tags:
        events += sound_tags(args.video)
    if args.chat:
        events += chat_events(load_chat(args.chat), offset=args.chat_offset)
    if not events:
        return None
    n = Counter(e.kind for e in events)
    print(f"  {n['audio']} audio spike(s), {n['sound']} tagged sound(s), {n['chat']} chat burst(s)")
    return Signals(events, segs)


_BUILD = object()


def find_clips(args, segs, vlen, signals=_BUILD):
    """Clips for these transcript lines. `signals` defaults to scanning args.video and
    args.chat; live mode passes its own (or None)."""
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the Anthropic SDK:  pip install -r requirements.txt")
    # The SDK retries rate limits, 5xx and dropped connections itself.
    client = anthropic.Anthropic(max_retries=4)
    errors = (anthropic.APIError, json.JSONDecodeError, KeyError, TypeError)

    template = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else llm.SYSTEM_PROMPT
    refusal_model = None if (args.refusal_model or "").lower() in ("", "none") else args.refusal_model
    # Tell Claude the content limits clean_clips will enforce: the cut adds padding, so
    # a clip that uses the full --max-len would otherwise lose its ending.
    from .clips import content_limits
    lo, hi = content_limits(args.min_len, args.max_len, args.pad_before + args.pad_after)
    system = llm.render_prompt(template, lo, hi)
    examples = ""
    if args.examples > 0:
        examples = fb.examples_prompt(fb.load_library(args.library), args.examples,
                                      min(3, args.examples), args.streamer)
    if examples:
        system += "\n\n" + examples
        print(f"Using {examples.count('<example ')} posted clip(s) from {args.library} as examples.")
    cfg = llm.LLMConfig(model=args.model, effort=args.effort, system=system,
                        context=args.context, refusal_model=refusal_model,
                        server_fallback=not args.no_server_fallback,
                        cache_system=bool(examples))
    if signals is _BUILD:
        signals = build_signals(args, segs)
    if signals:
        cfg.notes = signals.notes

    ranker = None if args.no_ranker else fb.load_ranker(args.ranker)
    if ranker and not ranker.get("enabled"):
        print(f"Re-ranker in {args.ranker} didn't beat Claude's score in testing; not using it.")
        ranker = None

    first_model = args.triage_model or args.model
    chunks = chunk_segments(segs, args.window, args.overlap)
    how = "Batch API" if (args.batch or args.batch_id) else f"{args.api_jobs} at a time"
    print(f"Analysing {len(chunks)} section(s) with {first_model} (effort {args.effort}, {how})...")

    if args.batch or args.batch_id:
        def remember(batch_id):
            (args.out / "batch_id.txt").write_text(batch_id + "\n", encoding="utf-8")
        raw, failed = llm.find_clips_batch(client, cfg, first_model, chunks, errors,
                                           batch_id=args.batch_id, on_created=remember)
    else:
        raw, failed = llm.find_clips_parallel(client, cfg, first_model, chunks,
                                              args.api_jobs, errors)
    if failed == len(chunks):
        sys.exit("Every section failed, so no clips.json was written.")
    if failed:
        print(f"  Warning: {failed} section(s) failed; their clips are missing.")

    pad = args.pad_before + args.pad_after
    if args.triage_model:
        # Keep a generous shortlist (lower bar, twice the count), then let the main model judge.
        shortlist = clean_clips(raw, segs, args.min_len, args.max_len,
                                max(args.min_score - 2, 0), args.top * 2 if args.top else 0, vlen, pad)
        print(f"Rescoring {len(shortlist)} shortlisted clip(s) with {args.model}...")
        raw = llm.rescore(client, cfg, shortlist, segs, errors)

    # With a re-ranker, keep every good non-overlapping candidate and let it pick the top N.
    clips = clean_clips(raw, segs, args.min_len, args.max_len, args.min_score,
                        0 if ranker else args.top, vlen, pad)
    fingerprint = source_fingerprint(args.video)
    for c in clips:
        c.id = clip_id(args.video, c.start, fingerprint)
        c.text = fb.clip_text(c, segs)
        c.features = fb.clip_features(c, segs, signals, pad=pad)
    if ranker:
        print(f"Re-ranking {len(clips)} candidate(s) with {args.ranker} "
              f"(trained on {ranker['trained_on']} posted clips).")
        clips = fb.rank_clips(clips, ranker, args.top)
    return clips


def resolve_layout(args, probe: Path, quiet: bool = False) -> Layout | None:
    """Layout from --facecam (one-off or saved with --save-layout) or the streamer's saved one.
    `probe` is a video with the source's frame size (the VOD, or a segment cut from it).
    quiet: don't mention a missing saved layout (--facecam auto will look for the cam)."""
    if args.no_facecam:
        return None
    saved = load_layouts(args.layouts).get(args.streamer) if args.streamer else None
    if args.facecam or args.gameplay:
        frame = video_size(probe)
        base = saved
        if not args.facecam and not base:
            sys.exit("--gameplay needs a --facecam (or a saved layout for --streamer).")
        layout = Layout(
            facecam=parse_box(args.facecam, frame) if args.facecam else base.facecam,
            gameplay=parse_box(args.gameplay, frame) if args.gameplay
            else (base.gameplay if base else (0.0, 0.0, 1.0, 1.0)),
            cam_height=args.cam_height or (base.cam_height if base else 0.35),
            cam_position=args.cam_position or (base.cam_position if base else "top"))
    elif saved:
        layout = saved
        if args.cam_height:
            layout.cam_height = args.cam_height
        if args.cam_position:
            layout.cam_position = args.cam_position
        layout.__post_init__()
    elif args.streamer and not quiet:
        # The name still labels clips for the feedback library; just no split layout.
        print(f"Note: no facecam layout saved for '{args.streamer}' in {args.layouts}, so no "
              f"split screen. Set one up with --grab-frame, then --facecam ... --save-layout.")
        return None
    else:
        return None

    if args.save_layout:
        if not args.streamer:
            sys.exit("--save-layout needs --streamer NAME.")
        save_layout(args.layouts, args.streamer, layout)
        print(f"Saved layout for '{args.streamer}' in {args.layouts}.")
    return layout


def layout_from_manifest(saved: dict) -> Layout | None:
    """The layout recorded in clips.json by an earlier --facecam auto ({"facecam": null} =
    looked, found no cam)."""
    if not saved.get("facecam"):
        return None
    try:
        return Layout(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items()})
    except (TypeError, ValueError):
        return None


def detect_layout(args, segments: list[dict] | None, folder: Path, vlen: float | None,
                  recipe: Path) -> Layout | None:
    """--facecam auto: ask Claude where the streamer's cam is, from frames of the VOD (or of
    the clip segments), and remember the answer in clips.json and layouts.json."""
    from dataclasses import asdict
    from .facecam import find_facecam, frame_times
    if segments:
        if len(segments) == 1:
            seg = segments[0]
            sources = [(folder / seg["file"], t) for t in frame_times(seg["end"] - seg["start"])]
        else:
            sources = [(folder / s["file"], (s["end"] - s["start"]) / 2) for s in segments[:4]]
    else:
        sources = [(args.video, t) for t in frame_times(vlen)]
    print("Looking for the streamer's cam...")
    try:
        box, note = find_facecam(sources, args.model)
    except Exception as e:  # never block cutting on this
        print(f"  couldn't look for the cam ({type(e).__name__}: {e}); cutting without it")
        return None
    if box is None:
        print(f"  no streamer cam found ({note}); clips get a plain 9:16 crop")
        update_manifest(recipe, layout={"facecam": None})
        return None
    layout = Layout(facecam=box, cam_height=args.cam_height or 0.35,
                    cam_position=args.cam_position or "top")
    print(f"  found it at x={box[0]:.2f} y={box[1]:.2f} w={box[2]:.2f} h={box[3]:.2f} ({note})")
    update_manifest(recipe, layout={k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in asdict(layout).items()})
    if args.streamer and args.streamer not in load_layouts(args.layouts):
        save_layout(args.layouts, args.streamer, layout)
        print(f"  saved as {args.streamer}'s layout in {args.layouts}; edit it there if it's off")
    return layout


def fetch_url(args) -> None:
    """--url: download video, captions and chat, and fill in the matching options."""
    from .fetch import fetch_source, streamer_slug
    dest = args.source_dir or args.out / "source"
    print(f"Fetching {args.url} into {dest}/ (already-downloaded files are reused)...")
    got = fetch_source(args.url, dest, video=True, captions=not args.no_captions,
                       chat=not args.no_chat and not args.chat, max_height=args.max_height,
                       audio_only=args.audio_first)
    if args.audio_first:
        if not got.audio:
            sys.exit("The download finished but no audio file was found.")
        # The audio stands in for the video until clips are picked; then only the
        # footage around them is downloaded (see fetch_sections).
        args.video, args.source_url = got.audio, got.url
        print(f"  audio:    {got.audio.name}" + (f"  ({got.title})" if got.title else "")
              + "; the video around each clip comes later")
    else:
        if not got.video:
            sys.exit("The download finished but no video file was found.")
        args.video = got.video
        print(f"  video:    {got.video.name}" + (f"  ({got.title})" if got.title else ""))
    if not args.transcript and not args.transcribe and not args.clips_json:
        if got.captions:
            args.transcript = got.captions
            print(f"  captions: {got.captions.name} (the platform's own, so no Whisper needed)")
        else:
            args.transcribe = True
            print("  captions: none available; will transcribe with Whisper")
    if got.chat and not args.chat:
        args.chat = got.chat
        print(f"  chat:     {got.chat.name}")
    if not args.context and got.title:
        args.context = f"Stream title: {got.title}" + (f" (channel: {got.uploader})" if got.uploader else "")
    if not args.streamer and got.uploader:
        args.streamer = streamer_slug(got.uploader)
        print(f"  streamer: {args.streamer} (from the channel name; override with --streamer)")


def fetch_sections(args, url: str, spans: list[tuple[float, float]], segments: list[dict],
                   folder: Path, recipe: Path, vlen: float | None) -> list[dict]:
    """Audio first: download the video for clip spans no section covers yet (the first
    render, approved clips found earlier, or a trim past what was downloaded)."""
    todo = [(s, e) for s, e in spans
            if not any(g["start"] <= s + 1e-3 and e <= g["end"] + 1e-3 for g in segments)]
    if not todo:
        return segments
    windows = segment_windows(todo, args.segment_pad, vlen)
    minutes = sum(e - s for s, e in windows) / 60
    print(f"\nDownloading the video around {len(todo)} clip(s): {len(windows)} section(s), "
          f"{minutes:.1f} min of footage...")
    with STATS.stage("download"):
        new = download_sections(url, folder, windows, args.max_height)
    known = {g["file"] for g in segments}
    segments = segments + [g for g in new if g["file"] not in known]
    update_manifest(recipe, segments=segments)
    print(f"  got {len(new)} of {len(windows)} section(s) in {folder / 'segments'}")
    return segments


def report_run(recipe: Path, kind: str, n_clips: int) -> None:
    """Print time and cost, and keep a record of them in clips.json under "runs"."""
    summary = STATS.summary(n_clips)
    if summary:
        print("\n" + summary)
    try:
        data = json.loads(recipe.read_text(encoding="utf-8"))
        update_manifest(recipe, runs=(data.get("runs") or []) + [STATS.to_dict(kind, n_clips)])
    except (OSError, ValueError) as e:
        print(f"  (couldn't record run stats in {recipe}: {e})")
def fetch_run(args) -> None:
    """--fetch: download a stream's output folder from the bucket so it can be re-rendered."""
    name = stream_name(args.fetch)
    dest = args.out / name
    try:
        files = Store.from_env().fetch_run(name, dest, args.jobs)
    except StorageError as e:
        sys.exit(str(e))
    print(f"Downloaded {len(files)} file(s) to {dest}/. Re-render with:\n"
          f"  python clipper.py --clips-json {dest / 'clips.json'} --stream {name} ...")


def upload_run(args, folder: Path, mode: str, made: list[Path], segments: list[dict]) -> None:
    """--upload: push this run's files (and recipe, transcript, VOD) to the bucket.
    Only files this run made are sent, since the output folder may hold other streams'."""
    name = stream_name(args.stream or (args.video.stem if args.video else folder.resolve().name))
    recipe = folder / "clips.json"
    update_manifest(recipe, stream=name)  # so a fetched copy re-uploads under the same name
    # Paths in the bucket mirror the run folder: clips at the top, previews/ and segments/ below.
    files = {p.with_suffix(ext): ("previews/" if mode == "preview" else "") + p.with_suffix(ext).name
             for p in made for ext in (".mp4", ".srt", ".ass")}
    files.update({folder / s["file"]: s["file"] for s in segments})
    files[recipe] = "clips.json"
    print(f"\nUploading to {os.environ.get('CLIPPER_BUCKET', '?')} as {name}...")
    try:
        store = Store.from_env()
        keys = store.upload_run(name, files, recipe, args.transcript,
                                args.video if args.upload_vod else None, args.jobs)
    except StorageError as e:
        sys.exit(str(e))
    print(f"  uploaded {len(keys)} file(s) to clips/{name}/ and recipes/{name}/"
          + (", source to vods/" if args.upload_vod else ""))
    if args.links:
        sub = "previews/" if mode == "preview" else ""
        clip_keys = sorted(k for k in store.sizes(f"clips/{name}/{sub}")
                           if k.endswith(".mp4") and "/" not in k[len(f"clips/{name}/{sub}"):])
        print(f"Download links (valid {MAX_LINK_DAYS} days):")
        for k in clip_keys:
            print(f"  {k.rsplit('/', 1)[-1]}: {store.link(k)}")


def main(argv=None, quiet: bool = False) -> None:
    """quiet: don't list every clip or report cost (live mode renders a few clips at a time)."""
    load_dotenv()
    args = parse_args(argv)
    if args.out is None:
        args.out = args.clips_json.parent if args.clips_json else Path("clips")
    if args.fetch:
        return fetch_run(args)
    if args.live:
        from .live import run_live
        return run_live(args)

    manifest = {}
    if args.clips_json:
        manifest, clips = load_manifest(args.clips_json)
        # clips.json is the recipe: it remembers which video and transcript it came from.
        if not args.video and manifest.get("video"):
            args.video = Path(manifest["video"])
        if not args.stream and manifest.get("stream"):
            args.stream = manifest["stream"]
        if not args.transcript and manifest.get("transcript"):
            t = Path(manifest["transcript"])
            if not t.exists():  # e.g. a run fetched from storage: the transcript sits beside it
                t = args.clips_json.parent / t.name
            if t.exists():
                args.transcript = t

    args.source_url = None
    if args.url:
        with STATS.stage("download"):
            fetch_url(args)

    # Saved segments live next to clips.json; their paths in it are relative to it.
    folder = args.clips_json.parent if args.clips_json else args.out
    segments = manifest.get("segments") or []
    # Audio first: there's no VOD, only the source URL; video comes in sections per clip.
    # A clips.json from such a run records source_url and no video, so re-renders work alike.
    if not args.source_url and not manifest.get("video") and not args.video:
        args.source_url = manifest.get("source_url")
    sections = bool(args.source_url)
    if args.segment_pad is None:
        args.segment_pad = SECTION_PAD if sections else SEGMENT_PAD
    if sections and (args.grab_frame is not None or args.save_segments):
        sys.exit("--grab-frame and --save-segments need the whole video; run without --audio-first "
                 "(with --audio-first the footage around each clip is already kept in segments/).")
    if args.from_segments and not segments:
        sys.exit("--from-segments needs a --clips-json with saved segments (made with --save-segments).")
    use_segments = sections or (bool(segments) and
                                (args.from_segments or not (args.video and args.video.exists())))

    if sections:
        pass  # sections are downloaded just before cutting
    elif use_segments:
        missing = [s["file"] for s in segments if not (folder / s["file"]).exists()]
        if missing:
            sys.exit(f"Saved segment(s) missing from {folder}: {', '.join(missing)}")
        if args.grab_frame is not None or args.save_segments:
            sys.exit("--grab-frame and --save-segments need the source video.")
        why = ("." if args.from_segments or not args.video
               else f" (source video {args.video} not found).")
        print(f"Rendering from the {len(segments)} saved clip segment(s) in {folder / 'segments'}{why}")
    else:
        if not args.video:
            sys.exit("Give --video, --url, or a --clips-json that records one.")
        if not args.video.exists():
            sys.exit(f"Video not found: {args.video}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg isn't installed or isn't on your PATH.")
    if use_segments:
        probe = folder / segments[0]["file"] if segments else None
    else:
        probe = args.video

    if args.grab_frame is not None:
        out = args.out / f"frame_{args.grab_frame:.0f}s.png"
        args.out.mkdir(parents=True, exist_ok=True)
        grab_frame(args.video, args.grab_frame, out)
        size = video_size(args.video)
        print(f"Saved {out}" + (f" ({size[0]}x{size[1]})" if size else ""))
        print("White lines are every 0.1 of the frame, yellow every 0.05. Read the facecam's\n"
              "left edge, top edge, width and height, then save it, e.g.:\n"
              f"  python clipper.py --video {args.video} --streamer NAME "
              "--facecam 0.72,0.03,0.26,0.30 --save-layout")
        return

    if not args.streamer and manifest.get("streamer"):
        args.streamer = manifest["streamer"]
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    # Cam on/off chosen per clip in the web UI ("facecam" in clips.json) beats the run's
    # setting. If a clip being cut wants its cam, find the layout even if the run doesn't.
    clip_cam = {i: d["facecam"] for i, d in enumerate(manifest.get("clips") or [], 1)
                if isinstance(d, dict) and isinstance(d.get("facecam"), bool)}
    forced = any(v for i, v in clip_cam.items() if only is None or i in only)
    run_no_cam, want_auto = args.no_facecam, args.facecam == "auto"
    if forced:
        args.no_facecam = False
    # --facecam auto: a saved layout for the streamer wins, then one found on an earlier run
    # of this clips.json; otherwise the cam is looked for just before cutting.
    auto_cam = (want_auto or forced) and not args.no_facecam
    if args.facecam == "auto":
        args.facecam = None
    layout = resolve_layout(args, probe, quiet=auto_cam)
    args.no_facecam = run_no_cam
    # Whether clips without their own choice get the cam.
    default_cam = not run_no_cam and (want_auto or layout is not None)
    if layout is None and auto_cam and isinstance(manifest.get("layout"), dict):
        layout = layout_from_manifest(manifest["layout"])
        auto_cam = False  # already decided for this video (cam or no cam)
    elif layout is not None:
        auto_cam = False
    if layout or auto_cam:
        args.vertical = True
    if args.save_layout and not (args.transcript or args.transcribe or args.clips_json):
        return  # just saving a layout

    if args.transcribe:
        cached = transcript_path_for(args.video)
        if cached.exists() and not args.retranscribe:
            print(f"Reusing existing transcript {cached} (pass --retranscribe to redo it).")
            args.transcript = cached
        else:
            with STATS.stage("transcribe"):
                args.transcript = transcribe(args.video, args.whisper_model, args.keep_audio)
    if not args.transcript and not args.clips_json:
        sys.exit("Give --transcript, --transcribe or --clips-json.")

    args.out.mkdir(parents=True, exist_ok=True)
    vlen = video_duration(args.video) if args.video and (sections or not use_segments) else None
    segs = load_transcript(args.transcript) if args.transcript else []

    if not args.clips_json:
        if not segs:
            sys.exit(f"The transcript {args.transcript} has no lines, so there's nothing to find clips in.")
        print(f"Loaded {len(segs)} transcript lines "
              f"({segs[0].start/60:.0f}-{segs[-1].end/60:.0f} min).")
        with STATS.stage("find clips"):
            clips = find_clips(args, segs, vlen)
        media = ({"video": None, "audio": args.video.resolve(), "source_url": args.source_url}
                 if sections else {"video": args.video.resolve()})
        save_manifest(args.out / "clips.json", clips, **media,
                      transcript=args.transcript.resolve(), model=args.model,
                      triage_model=args.triage_model, effort=args.effort,
                      streamer=args.streamer)

    if not quiet:
        print(f"\n{len(clips)} clips selected:")
        for i, c in enumerate(clips, 1):
            print(f"  {i:2d}. [{c.score:.0f}/10] {fmt_srt_time(c.start)[:8]} "
                  f"({c.duration:.0f}s) {c.title}")

    if args.dry_run:
        if args.clips_json:  # nothing new was written; point at the file that exists
            print(f"\nDry run: nothing was cut or written. The clip list is {args.clips_json}; "
                  f"cut later with --clips-json {args.clips_json}")
        else:
            print(f"\nDry run: clip list saved to {args.out/'clips.json'}. "
                  f"Cut later with --clips-json {args.out/'clips.json'}")
        if args.save_segments:
            print("--save-segments was skipped because this is a dry run; "
                  "run again without --dry-run to save the segments.")
        report_run(args.clips_json or args.out / "clips.json", "find", len(clips))
        return

    if args.save_segments:
        recipe = args.clips_json or args.out / "clips.json"
        windows = segment_windows([(max(c.start - args.pad_before, 0.0), c.end + args.pad_after)
                                   for c in clips], args.segment_pad, vlen)
        print(f"\nSaving {len(windows)} clip segment(s) with {args.segment_pad:.0f}s either side "
              f"(stream copy)...")
        new = save_segments(args.video, folder, windows, args.jobs)
        for old in segments:  # drop files from an earlier run that the new list doesn't reuse
            if old["file"] not in {s["file"] for s in new}:
                (folder / old["file"]).unlink(missing_ok=True)
        update_manifest(recipe, segments=new)
        segments = new
        print(f"  saved to {folder / 'segments'}; {recipe} can now re-render without the VOD.")

    if sections:
        spans = [(max(c.start - args.pad_before, 0.0),
                  min(c.end + args.pad_after, vlen) if vlen else c.end + args.pad_after)
                 for i, c in enumerate(clips, 1) if only is None or i in only]
        segments = fetch_sections(args, args.source_url, spans, segments, folder,
                                  args.clips_json or args.out / "clips.json", vlen)
        if not segments:
            sys.exit("None of the clip sections could be downloaded, so there's nothing to cut.")
        probe = folder / segments[0]["file"]

    if auto_cam:
        layout = detect_layout(args, segments if use_segments else None, folder, vlen,
                               args.clips_json or args.out / "clips.json")

    burn = args.burn_captions and bool(segs)
    if args.burn_captions and not segs:
        print("No transcript, so no captions to burn in.")
    overlay = burn or args.hook_overlay
    mode = choose_mode(args.vertical, overlay, args.preview, args.reencode)
    if overlay:  # fail fast, before any work, if nothing can burn in text
        from .render import ffmpeg_for
        try:
            exe = ffmpeg_for(True)
        except RuntimeError as e:
            sys.exit(str(e))
        if exe != "ffmpeg":
            print("Your ffmpeg can't burn in text (no libass); using the bundled imageio-ffmpeg for that.")
    style = CaptionStyle(animated=args.caption_style == "animated", font=args.caption_font,
                         color=args.caption_color, highlight=args.highlight_color,
                         uppercase=args.caption_case == "upper")
    frame = VERTICAL_SIZE if args.vertical else (video_size(probe) or (1920, 1080))
    encoder = "libx264" if mode != "encode" else (
        pick_encoder() if args.encoder == "auto" else args.encoder)
    dest = args.out / "previews" if mode == "preview" else args.out
    dest.mkdir(parents=True, exist_ok=True)
    how = {"copy": "stream copy (no re-encode)", "preview": "480p previews",
           "encode": encoder}[mode]
    if layout and (default_cam or forced):
        how += f", {args.streamer or 'custom'} facecam layout"
    print(f"\nCutting clips with {how} ({args.jobs} at a time)...")

    jobs = []
    for i, c in enumerate(clips, 1):
        if only is not None and i not in only:
            continue
        start = max(c.start - args.pad_before, 0)
        end = c.end + args.pad_after
        if vlen:
            end = min(end, vlen)
        # Times stay in the source video's clock (captions use them); `offset` maps
        # them into a segment, which starts `offset` seconds into the source.
        src, offset = args.video, 0.0
        if use_segments:
            seg = segment_for(segments, start, end)
            if seg is None:
                print(f"  skipping clip {i} ({c.title}): no saved segment covers it")
                continue
            src, offset = folder / seg["file"], seg["start"]
            if start < seg["start"] or end > seg["end"]:
                print(f"  clip {i} reaches past its saved segment; trimmed to "
                      f"{fmt_srt_time(seg['start'])[:8]}-{fmt_srt_time(seg['end'])[:8]}")
                start, end = max(start, seg["start"]), min(end, seg["end"])
        clip_mode, clip_encoder = mode, encoder
        if mode == "copy":  # copy can only start on a keyframe
            key = offset + keyframe_before(src, start - offset)
            if end - key > args.max_len + 0.05:
                # Starting at that keyframe would break --max-len: re-encode this one
                # clip so it starts exactly where it should.
                clip_mode = "encode"
                clip_encoder = pick_encoder() if args.encoder == "auto" else args.encoder
            else:
                start = key
        base = dest / f"{i:02d}_{slugify(c.title)}"
        clip_layout = layout if clip_cam.get(i, default_cam) else None
        if segs:
            write_clip_srt(base.with_suffix(".srt"), segs, start, end)
        ass = None
        if overlay:
            ass = base.with_suffix(".ass")
            build_ass(ass, segs, start, end, frame, style, captions=burn,
                      caption_y=clip_layout.seam() if clip_layout else None,
                      hook=c.hook if args.hook_overlay else None,
                      hook_seconds=args.hook_seconds)
        jobs.append((src, base.with_suffix(".mp4"), start - offset, end - offset, clip_mode,
                     args.vertical, ass, clip_encoder, clip_layout))
    from .render import run_cut_jobs
    with STATS.stage("cut"):
        saved, failed = run_cut_jobs(jobs, args.jobs)

    recipe = args.clips_json or args.out / "clips.json"
    if failed:
        print(f"\nDone with errors: {len(saved)} of {len(jobs)} clips saved in {dest}/, "
              f"{len(failed)} failed (details in {recipe}):")
        for name, reason in sorted(failed):
            print(f"  - {name}: {reason}")
        print("Fix the cause above and re-run; the saved clips are fine.")
    else:
        print(f"\nDone. {len(jobs)} clips in {dest}/ (details in {recipe})")
    if not quiet:
        report_run(recipe, "render" if args.clips_json else "find+cut", len(saved))
    if mode == "preview" and not quiet:
        print(f"Review the previews in the web UI (python ui.py: Keep/Reject, then Render approved),\n"
              f"or delete the clips you don't want from clips.json and render\n"
              f"the keepers at full quality:  python clipper.py --clips-json {recipe}"
              f"{f' --streamer {args.streamer}' if args.streamer else ''}"
              f"{' --vertical' if args.vertical and not layout else ''}"
              f"{' --burn-captions' if args.burn_captions else ''}"
              f"{' --hook-overlay' if args.hook_overlay else ''}")
    if failed:
        if args.upload:
            print("Not uploaded: a clip failed, so nothing was sent to the bucket. "
                  "Fix it and re-run with --upload.")
        sys.exit(1)
    if args.upload:
        upload_run(args, folder, mode, [job[1] for job in jobs], segments)


if __name__ == "__main__":
    main()
