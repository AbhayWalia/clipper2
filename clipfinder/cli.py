"""Command-line entry point: transcript -> Claude -> clips.json -> ffmpeg."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import feedback as fb
from . import llm
from .captions import CaptionStyle, build_ass
from .clips import clean_clips, load_manifest, save_manifest, update_manifest
from .layout import DEFAULT_LAYOUTS, Layout, grab_frame, load_layouts, parse_box, save_layout
from .render import (SEGMENT_PAD, VERTICAL_SIZE, choose_mode, cut_clip, fmt_srt_time,
                     keyframe_before, pick_encoder, save_segments, segment_for, segment_windows,
                     slugify, video_duration, video_size, write_clip_srt)
from .signals import Signals, audio_events, chat_events, load_chat, measure_loudness
from .transcribe import transcribe, transcript_path_for
from .transcript import chunk_segments, load_transcript


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find and cut short-form clips from a gameplay stream.")
    io = p.add_argument_group("input / output")
    io.add_argument("--url", help="Twitch VOD / YouTube URL: downloads the video, its captions and chat")
    io.add_argument("--source-dir", type=Path, help="Where --url downloads go (default: OUT/source)")
    io.add_argument("--no-chat", action="store_true", help="With --url, don't download chat")
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
    io.add_argument("--out", type=Path, default=Path("clips"), help="Output folder")
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

    lay = p.add_argument_group("facecam layout (per streamer)")
    lay.add_argument("--streamer", help="Use this streamer's facecam layout from the layouts file "
                                        "(implies a 9:16 split: facecam + gameplay)")
    lay.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS, help="Layouts file")
    lay.add_argument("--facecam", help="Facecam box x,y,w,h as fractions (0.72,0.03,0.26,0.30) "
                                       "or pixels; overrides the saved one")
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
    cut.add_argument("--jobs", type=int, default=4, help="Clips to cut at the same time")
    cut.add_argument("--encoder", choices=["auto", "h264_videotoolbox", "libx264"], default="auto",
                     help="auto uses the Mac's hardware encoder when available")

    keep = p.add_argument_group("clip segments (keep these instead of the whole VOD)")
    keep.add_argument("--save-segments", action="store_true",
                      help="Stream-copy the footage around each clip into OUT/segments/ and record it "
                           "in clips.json, so clips can be re-rendered after the VOD is deleted")
    keep.add_argument("--segment-pad", type=float, default=SEGMENT_PAD,
                      help="Seconds of extra footage kept on each side of a clip")
    keep.add_argument("--from-segments", action="store_true",
                      help="With --clips-json, render from the saved segments even if the VOD exists "
                           "(automatic when it doesn't)")
    return p.parse_args(argv)


def build_signals(args, segs) -> Signals | None:
    events = []
    if not args.no_audio_signals:
        cache = args.video.with_suffix(".loudness.json")
        print("Scanning audio for loudness spikes" + (" (cached)" if cache.exists() else "") + "...")
        try:
            events += audio_events(measure_loudness(args.video, cache), args.audio_threshold)
        except Exception as e:  # never block clipping on this
            print(f"  audio scan failed ({e}); continuing without it")
    if args.chat:
        events += chat_events(load_chat(args.chat), offset=args.chat_offset)
    if not events:
        return None
    n_audio = sum(e.kind == "audio" for e in events)
    print(f"  {n_audio} audio spike(s), {len(events) - n_audio} chat burst(s)")
    return Signals(events, segs)


def find_clips(args, segs, vlen):
    try:
        import anthropic
    except ImportError:
        sys.exit("Install the Anthropic SDK:  pip install -r requirements.txt")
    # The SDK retries rate limits, 5xx and dropped connections itself.
    client = anthropic.Anthropic(max_retries=4)
    errors = (anthropic.APIError, json.JSONDecodeError, KeyError, TypeError)

    template = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else llm.SYSTEM_PROMPT
    refusal_model = None if (args.refusal_model or "").lower() in ("", "none") else args.refusal_model
    system = llm.render_prompt(template, args.min_len, args.max_len)
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
    for c in clips:
        c.id = f"{args.video.stem}@{c.start:.1f}"
        c.text = fb.clip_text(c, segs)
        c.features = fb.clip_features(c, segs, signals)
    if ranker:
        print(f"Re-ranking {len(clips)} candidate(s) with {args.ranker} "
              f"(trained on {ranker['trained_on']} posted clips).")
        clips = fb.rank_clips(clips, ranker, args.top)
    return clips


def resolve_layout(args, probe: Path) -> Layout | None:
    """Layout from --facecam (one-off or saved with --save-layout) or the streamer's saved one.
    `probe` is a video with the source's frame size (the VOD, or a segment cut from it)."""
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
    elif args.streamer:
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


def fetch_url(args) -> None:
    """--url: download video, captions and chat, and fill in the matching options."""
    from .fetch import fetch_source, streamer_slug
    dest = args.source_dir or args.out / "source"
    print(f"Fetching {args.url} into {dest}/ (already-downloaded files are reused)...")
    got = fetch_source(args.url, dest, video=True, captions=not args.no_captions,
                       chat=not args.no_chat and not args.chat, max_height=args.max_height)
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


def main(argv=None) -> None:
    args = parse_args(argv)

    manifest = {}
    if args.clips_json:
        manifest, clips = load_manifest(args.clips_json)
        # clips.json is the recipe: it remembers which video and transcript it came from.
        if not args.video and manifest.get("video"):
            args.video = Path(manifest["video"])
        if not args.transcript and manifest.get("transcript") and Path(manifest["transcript"]).exists():
            args.transcript = Path(manifest["transcript"])

    if args.url:
        fetch_url(args)

    # Saved segments live next to clips.json; their paths in it are relative to it.
    folder = args.clips_json.parent if args.clips_json else args.out
    segments = manifest.get("segments") or []
    if args.from_segments and not segments:
        sys.exit("--from-segments needs a --clips-json with saved segments (made with --save-segments).")
    use_segments = bool(segments) and (args.from_segments or not (args.video and args.video.exists()))

    if use_segments:
        missing = [s["file"] for s in segments if not (folder / s["file"]).exists()]
        if missing:
            sys.exit(f"Saved segment(s) missing from {folder}: {', '.join(missing)}")
        if args.grab_frame is not None or args.save_segments:
            sys.exit("--grab-frame and --save-segments need the source video.")
        print(f"Rendering from the {len(segments)} saved clip segment(s) in {folder / 'segments'}"
              + ("." if args.from_segments else f" (source video {args.video} not found)."))
    else:
        if not args.video:
            sys.exit("Give --video, --url, or a --clips-json that records one.")
        if not args.video.exists():
            sys.exit(f"Video not found: {args.video}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg isn't installed or isn't on your PATH.")
    probe = folder / segments[0]["file"] if use_segments else args.video

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
    layout = resolve_layout(args, probe)
    if layout:
        args.vertical = True
    if args.save_layout and not (args.transcript or args.transcribe or args.clips_json):
        return  # just saving a layout

    if args.transcribe:
        cached = transcript_path_for(args.video)
        if cached.exists() and not args.retranscribe:
            print(f"Reusing existing transcript {cached} (pass --retranscribe to redo it).")
            args.transcript = cached
        else:
            args.transcript = transcribe(args.video, args.whisper_model, args.keep_audio)
    if not args.transcript and not args.clips_json:
        sys.exit("Give --transcript, --transcribe or --clips-json.")

    args.out.mkdir(parents=True, exist_ok=True)
    vlen = None if use_segments else video_duration(args.video)
    segs = load_transcript(args.transcript) if args.transcript else []

    if not args.clips_json:
        print(f"Loaded {len(segs)} transcript lines "
              f"({segs[0].start/60:.0f}-{segs[-1].end/60:.0f} min).")
        clips = find_clips(args, segs, vlen)
        save_manifest(args.out / "clips.json", clips, video=args.video.resolve(),
                      transcript=args.transcript.resolve(), model=args.model,
                      triage_model=args.triage_model, effort=args.effort,
                      streamer=args.streamer)

    print(f"\n{len(clips)} clips selected:")
    for i, c in enumerate(clips, 1):
        print(f"  {i:2d}. [{c.score:.0f}/10] {fmt_srt_time(c.start)[:8]} "
              f"({c.duration:.0f}s) {c.title}")

    if args.dry_run:
        print(f"\nDry run: clip list saved to {args.out/'clips.json'}. "
              f"Cut later with --clips-json {args.out/'clips.json'}")
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
           "encode": encoder}[mode] + (f", {args.streamer or 'custom'} facecam layout" if layout else "")
    print(f"\nCutting clips with {how} ({args.jobs} at a time)...")

    jobs = []
    for i, c in enumerate(clips, 1):
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
        if mode == "copy":  # copy can only start on a keyframe
            start = offset + keyframe_before(src, start - offset)
        base = dest / f"{i:02d}_{slugify(c.title)}"
        if segs:
            write_clip_srt(base.with_suffix(".srt"), segs, start, end)
        ass = None
        if overlay:
            ass = base.with_suffix(".ass")
            build_ass(ass, segs, start, end, frame, style, captions=burn,
                      caption_y=layout.seam() if layout else None,
                      hook=c.hook if args.hook_overlay else None,
                      hook_seconds=args.hook_seconds)
        jobs.append((src, base.with_suffix(".mp4"), start - offset, end - offset, mode,
                     args.vertical, ass, encoder, layout))
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        futures = {pool.submit(cut_clip, *job): job[1].name for job in jobs}
        for done in as_completed(futures):
            done.result()
            print(f"  saved {futures[done]}")

    recipe = args.clips_json or args.out / "clips.json"
    print(f"\nDone. {len(clips)} clips in {dest}/ (details in {recipe})")
    if mode == "preview":
        print(f"Review the previews, delete the clips you don't want from clips.json, then render\n"
              f"the keepers at full quality:  python clipper.py --clips-json {recipe}"
              f"{f' --streamer {args.streamer}' if args.streamer else ''}"
              f"{' --vertical' if args.vertical and not layout else ''}"
              f"{' --burn-captions' if args.burn_captions else ''}"
              f"{' --hook-overlay' if args.hook_overlay else ''}")


if __name__ == "__main__":
    main()
