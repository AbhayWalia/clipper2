"""Regression tests: .srt timestamps must carry millisecond rounding into seconds."""

from clipfinder.render import fmt_srt_time, write_clip_srt
from clipfinder.transcript import Segment


def test_fmt_srt_time_carries_rounding_into_seconds():
    # Used to wrap 1000 ms to 000 without carrying: 59.9996 -> 00:00:59,000.
    assert fmt_srt_time(59.9996) == "00:01:00,000"
    assert fmt_srt_time(3599.9999) == "01:00:00,000"
    assert fmt_srt_time(1.0004) == "00:00:01,000"
    assert fmt_srt_time(0.9994) == "00:00:00,999"
    assert fmt_srt_time(12.3456) == "00:00:12,346"


def test_clip_srt_cues_never_end_before_they_start(tmp_path):
    segs = [Segment(10.0, 10.9996, "a"), Segment(10.9996, 12.0, "b")]
    p = tmp_path / "c.srt"
    write_clip_srt(p, segs, t0=0.0, t1=20.0)
    for line in p.read_text().splitlines():
        if "-->" in line:
            a, b = line.split(" --> ")
            assert a <= b, line
