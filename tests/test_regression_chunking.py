"""Regression tests: --overlap >= --window used to loop forever."""

import pytest

from clipfinder.transcript import Segment, chunk_segments

SEGS = [Segment(i * 5.0, i * 5.0 + 5.0, f"line {i}") for i in range(10)]


def test_chunk_segments_rejects_overlap_not_smaller_than_window():
    # Used to loop forever: t += window - overlap never moved forward.
    for window, overlap in [(60, 90), (60, 60), (0, 0), (60, -1)]:
        with pytest.raises(ValueError):
            chunk_segments(SEGS, window=window, overlap=overlap)


def test_cli_rejects_overlap_not_smaller_than_window(capsys):
    from clipfinder.cli import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--video", "v.mp4", "--window", "60", "--overlap", "90"])
    assert "--overlap (90) must be at least 0 and smaller than --window (60)" in capsys.readouterr().err
    assert parse_args(["--video", "v.mp4", "--window", "60", "--overlap", "10"]).overlap == 10
