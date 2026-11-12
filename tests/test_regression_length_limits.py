"""Regression tests: the prompt states the same clip-length limits clean_clips enforces."""

from clipfinder.clips import Clip, clean_clips
from clipfinder.llm import render_prompt
from clipfinder.transcript import Segment

SEGS = [Segment(i * 5.0, i * 5.0 + 5.0, f"line {i}") for i in range(10)]


def test_prompt_limits_match_what_clean_clips_enforces():
    # The prompt said "never over 75s" while clean_clips trimmed content to 72s,
    # cutting the ending (usually the payoff) off clips that used the full length.
    from clipfinder.clips import content_limits
    from clipfinder.llm import SYSTEM_PROMPT
    lo, hi = content_limits(12, 75, 3.0)
    assert (lo, hi) == (9.0, 72.0)
    assert "never under 9s or over 72s" in render_prompt(SYSTEM_PROMPT, lo, hi)
    longest = Clip(0.0, hi, 9, "t", "h", "funny", "r")
    out = clean_clips([longest], [Segment(0, 100, "x")], 12, 75, 0, 0, None, pad=3.0)
    assert out[0].end == hi            # a clip at the stated limit is not trimmed


def test_cli_prompt_uses_content_limits(tmp_path, monkeypatch):
    # The system prompt sent to Claude carries the padded-aware limits.
    import clipfinder.cli as cli
    seen = {}

    def fake_parallel(client, cfg, model, chunks, jobs, errors):
        seen["system"] = cfg.system
        return [], 0
    monkeypatch.setattr(cli.llm, "find_clips_parallel", fake_parallel)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    args = cli.parse_args(["--video", str(tmp_path / "v.mp4"), "--no-audio-signals",
                           "--examples", "0", "--max-len", "60", "--pad-before", "2",
                           "--pad-after", "3"])
    cli.find_clips(args, SEGS, None)
    assert "never under 7s or over 55s" in seen["system"]
