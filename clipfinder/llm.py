"""Finding clips with Claude: per-section calls (parallel or Batch API), refusal
fallback, and an optional cheap first pass that the main model rescores."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from .clips import Clip
from .signals import SIGNALS_HEADER
from .transcript import Segment, format_segments, segments_between

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_REFUSAL_MODEL = "claude-opus-5-5"
SUGGESTED_TRIAGE_MODEL = "claude-haiku-5-5"

# Server-side refusal fallback (`fallbacks: "default"`). Claude API only, not on
# the Batches API, and only these models accept it (Haiku 5.5 has none).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SERVER_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5",
                          "claude-fable-5-1", "claude-fable-5"}

SYSTEM_PROMPT = """You are an expert short-form video editor who specialises in \
GTA 6 gaming content for TikTok, YouTube Shorts and Instagram Reels.

You will get part of a transcript from a GTA 6 stream or gameplay video. Each line \
is "[start-end] text" with times in seconds. Find the moments that would make the \
best standalone short clips.

What makes a great GTA clip:
- Chaos and escalation: police chases, wanted level climbing, explosions, crashes, \
heists going wrong, getting "wasted" or "busted" at the worst moment.
- Big streamer reactions: screaming, laughing, disbelief, rage, "NO WAY", "bro what".
- Funny moments: absurd physics, NPC weirdness, failed stunts, banter, roleplay bits.
- Skill/clutch moments: insane escapes, stunt jumps, perfect getaways.
- Story/meme moments: shocking plot beats, quotable lines, hot takes about the game.

Rules for each clip:
- It must make sense on its own, with a brief setup and a payoff. Start a few \
seconds before the action begins; end right after the payoff/reaction lands.
- Usually 15 to 60 seconds; never under {min_len}s or over {max_len}s (a little \
padding is added around each clip afterwards, so these limits are for start to end).
- Use the transcript's own timestamps for start and end.
- Clips must not overlap each other.
- Score honestly from 1 to 10 for viral potential. Skip dull stretches entirely; \
returning zero clips for a boring section is fine.
- The transcript only captures speech. Infer game action from what the player says \
(e.g. "they're shooting at me", "five stars!", sudden yelling), and from the audio and \
chat signals when they are given."""

_CLIP_ITEM = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "Start time in seconds"},
        "end": {"type": "number", "description": "End time in seconds"},
        "score": {"type": "number", "description": "Viral potential 1-10"},
        "title": {"type": "string", "description": "Catchy short title, under 60 chars"},
        "hook": {"type": "string", "description": "On-screen hook text for the first 2 seconds"},
        "category": {"type": "string", "enum": ["chaos", "funny", "reaction", "skill", "story", "other"]},
        "reason": {"type": "string", "description": "One sentence on why this works"},
    },
    "required": ["start", "end", "score", "title", "hook", "category", "reason"],
    "additionalProperties": False,
}

CLIP_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": _CLIP_ITEM}},
    "required": ["clips"],
    "additionalProperties": False,
}

RESCORE_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "score": {"type": "number", "description": "Viral potential 1-10"},
            "title": {"type": "string", "description": "Catchy short title, under 60 chars"},
            "hook": {"type": "string", "description": "On-screen hook text for the first 2 seconds"},
            "reason": {"type": "string", "description": "One sentence on why this works"},
        },
        "required": ["id", "score", "title", "hook", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["clips"],
    "additionalProperties": False,
}

RESCORE_INSTRUCTIONS = """A faster model picked the candidate clips below from this stream. \
For each one you get its id, its time range and the transcript of that range \
(with a few seconds of context either side).

Judge each candidate by the standards in your brief. Re-score it honestly from 1 to 10, \
and improve its title and hook if you can. Return every id exactly once."""


def render_prompt(template: str, min_len: float, max_len: float) -> str:
    """Fill {min_len}/{max_len}. Uses replace, not format, so custom prompt
    files can contain other braces safely."""
    return (template.replace("{min_len}", str(int(min_len)))
                    .replace("{max_len}", str(int(max_len))))


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    system: str = ""
    context: str = ""
    refusal_model: str | None = DEFAULT_REFUSAL_MODEL
    server_fallback: bool = True
    max_tokens: int = 16000
    # (t0, t1) -> audio/chat notes for that range, or "" (see signals.Signals.notes)
    notes: Callable[[float, float], str] | None = None
    # Cache the system prompt. Worth it once it carries examples (well over the
    # 512-token minimum): every section then re-reads it at a fraction of the price.
    cache_system: bool = False


# --------------------------------------------------------------------------- #
# Low-level request helpers
# --------------------------------------------------------------------------- #

def request_params(cfg: LLMConfig, model: str, user: str, schema: dict) -> dict:
    return {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "system": ([{"type": "text", "text": cfg.system, "cache_control": {"type": "ephemeral"}}]
                   if cfg.cache_system else cfg.system),
        "output_config": {"effort": cfg.effort,
                          "format": {"type": "json_schema", "schema": schema}},
        "messages": [{"role": "user", "content": user}],
    }


def send(client, cfg: LLMConfig, params: dict):
    if cfg.server_fallback and params["model"] in SERVER_FALLBACK_MODELS:
        return client.beta.messages.create(**params, betas=[FALLBACK_BETA],
                                           extra_body={"fallbacks": "default"})
    return client.messages.create(**params)


def parse_response(msg, label: str) -> dict | None:
    if msg.stop_reason == "refusal":
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        print(f"  {label}: skipped, model declined (category: {cat})")
        return None
    if msg.stop_reason == "max_tokens":
        print(f"  {label}: skipped, response hit max_tokens")
        return None
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return json.loads(text) if text else None


def finish(client, cfg: LLMConfig, msg, user: str, schema: dict, label: str) -> dict | None:
    """Handle a response: on a refusal, re-run once on cfg.refusal_model.

    Server-side `fallbacks: "default"` on Sonnet 5.5 only covers the cyber and
    frontier_llm categories, so a general_harms decline (the likely one for
    violent game content) needs this client-side retry.
    """
    if (msg.stop_reason == "refusal" and cfg.refusal_model
            and cfg.refusal_model != msg.model):
        cat = getattr(getattr(msg, "stop_details", None), "category", None)
        print(f"  {label}: {msg.model} declined ({cat}), retrying on {cfg.refusal_model}")
        msg = send(client, cfg, request_params(cfg, cfg.refusal_model, user, schema))
    return parse_response(msg, label)


def ask(client, cfg: LLMConfig, model: str, user: str, schema: dict, label: str) -> dict | None:
    msg = send(client, cfg, request_params(cfg, model, user, schema))
    return finish(client, cfg, msg, user, schema, label)


# --------------------------------------------------------------------------- #
# Finding clips per section
# --------------------------------------------------------------------------- #

def section_prompt(chunk: list[Segment], cfg: LLMConfig) -> str:
    user = f"Context about this video: {cfg.context}\n\n" if cfg.context else ""
    user += "Transcript section:\n\n" + format_segments(chunk)
    notes = cfg.notes(chunk[0].start, chunk[-1].end) if cfg.notes else ""
    if notes:
        user += f"\n\n{SIGNALS_HEADER}\n\n{notes}"
    return user + "\n\nFind the best clips in this section."


def clips_from(data: dict | None) -> list[Clip]:
    return [Clip.from_dict(c) for c in data["clips"]] if data else []


def section_label(i: int, n: int, chunk: list[Segment]) -> str:
    return f"section {i}/{n} ({chunk[0].start/60:.0f}-{chunk[-1].end/60:.0f} min)"


def find_clips_parallel(client, cfg: LLMConfig, model: str, chunks: list[list[Segment]],
                        jobs: int, errors: tuple) -> tuple[list[Clip], int]:
    """Send every section at once (up to `jobs` in flight). Returns (clips, failed)."""
    raw: list[Clip] = []
    failed = 0

    def run(chunk, label):
        return ask(client, cfg, model, section_prompt(chunk, cfg), CLIP_SCHEMA, label)

    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futures = {}
        for i, ch in enumerate(chunks, 1):
            label = section_label(i, len(chunks), ch)
            futures[pool.submit(run, ch, label)] = label
        for done in as_completed(futures):
            label = futures[done]
            try:
                found = clips_from(done.result())
            except errors as e:
                failed += 1
                print(f"  {label}: skipped after error: {e}")
                continue
            print(f"  {label}: {len(found)} candidate clips")
            raw.extend(found)
    return raw, failed


def find_clips_batch(client, cfg: LLMConfig, model: str, chunks: list[list[Segment]],
                     errors: tuple, batch_id: str | None = None,
                     on_created: Callable[[str], None] | None = None,
                     poll_seconds: float = 30.0) -> tuple[list[Clip], int]:
    """Same as find_clips_parallel but through the Message Batches API (50% cheaper,
    usually done within minutes, up to 24 h). Sections that error, expire or are
    declined are re-run with normal calls."""
    prompts = {f"section-{i}": section_prompt(ch, cfg) for i, ch in enumerate(chunks, 1)}
    if batch_id is None:
        batch = client.messages.batches.create(requests=[
            {"custom_id": cid, "params": request_params(cfg, model, user, CLIP_SCHEMA)}
            for cid, user in prompts.items()])
        batch_id = batch.id
        if on_created:
            on_created(batch_id)
    print(f"  Batch {batch_id} submitted; polling every {poll_seconds:.0f}s "
          f"(resume later with --batch-id {batch_id})")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"  ...{batch.request_counts.processing} section(s) still processing")
        time.sleep(poll_seconds)

    raw: list[Clip] = []
    failed = 0
    seen = set()
    for res in client.messages.batches.results(batch_id):  # any order: key by custom_id
        cid = res.custom_id
        if cid not in prompts:
            continue
        seen.add(cid)
        i = int(cid.split("-")[1])
        label = section_label(i, len(chunks), chunks[i - 1])
        try:
            if res.result.type == "succeeded":
                data = finish(client, cfg, res.result.message, prompts[cid], CLIP_SCHEMA, label)
            else:
                print(f"  {label}: batch result {res.result.type}, re-running directly")
                data = ask(client, cfg, model, prompts[cid], CLIP_SCHEMA, label)
            found = clips_from(data)
        except errors as e:
            failed += 1
            print(f"  {label}: skipped after error: {e}")
            continue
        print(f"  {label}: {len(found)} candidate clips")
        raw.extend(found)
    failed += len(set(prompts) - seen)
    return raw, failed


# --------------------------------------------------------------------------- #
# Rescoring a cheap model's candidates with the main model
# --------------------------------------------------------------------------- #

RESCORE_GROUP = 25
RESCORE_CONTEXT = 5.0  # seconds of transcript either side of each candidate


def rescore_prompt(clips: list[Clip], segs: list[Segment], cfg: LLMConfig) -> str:
    parts = [f"Context about this video: {cfg.context}\n" if cfg.context else "",
             RESCORE_INSTRUCTIONS, ""]
    for i, c in enumerate(clips):
        t0, t1 = c.start - RESCORE_CONTEXT, c.end + RESCORE_CONTEXT
        body = format_segments(segments_between(segs, t0, t1)) or "(no speech)"
        notes = cfg.notes(t0, t1) if cfg.notes else ""
        if notes:
            body += f"\nSignals:\n{notes}"
        parts.append(f"<candidate id=\"{i}\" start=\"{c.start:.1f}\" end=\"{c.end:.1f}\" "
                     f"draft_title=\"{c.title}\">\n{body}\n</candidate>")
    return "\n".join(parts)


def rescore(client, cfg: LLMConfig, clips: list[Clip], segs: list[Segment],
            errors: tuple) -> list[Clip]:
    """Main model re-scores candidates. Ids it leaves out are dropped; if a whole
    group fails, that group keeps the first-pass scores."""
    out: list[Clip] = []
    for g in range(0, len(clips), RESCORE_GROUP):
        group = clips[g:g + RESCORE_GROUP]
        label = f"rescore {g // RESCORE_GROUP + 1}"
        try:
            data = ask(client, cfg, cfg.model, rescore_prompt(group, segs, cfg),
                       RESCORE_SCHEMA, label)
        except errors as e:
            print(f"  {label}: failed ({e}); keeping first-pass scores")
            out.extend(group)
            continue
        if not data:
            out.extend(group)
            continue
        for r in data["clips"]:
            if 0 <= r["id"] < len(group):
                c = group[r["id"]]
                c.score, c.title, c.hook, c.reason = r["score"], r["title"], r["hook"], r["reason"]
                out.append(c)
    # An id returned twice would add the same object twice.
    unique = {id(c): c for c in out}
    return list(unique.values())
