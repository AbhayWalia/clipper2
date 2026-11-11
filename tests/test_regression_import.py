"""Regression tests: one bad CSV row must not abort `feedback.py import`."""

import csv
import json

import pytest

from clipfinder.clips import save_manifest
from clipfinder.feedback_cli import main as fb_main
from clipfinder.clips import Clip


def clip(i=0, start=10.0, end=40.0, score=8.0, cat="chaos", **kw):
    return Clip(start, end, score, f"title {i}", f"hook {i}", cat, "r",
                id=kw.get("id", f"vod@{start:.1f}"), text=kw.get("text", f"words {i}"),
                features=kw.get("features", {}))


def test_import_skips_bad_rows_and_saves_the_rest(tmp_path, capsys):
    # A bad clip number used to sys.exit mid-import, losing every row before it.
    mf, lib = tmp_path / "clips.json", tmp_path / "library.json"
    save_manifest(mf, [clip(i, start=float(i * 60)) for i in range(3)], video="v.mp4")
    rows = tmp_path / "stats.csv"
    with open(rows, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "views"])
        w.writerow(["1", "100"])
        w.writerow(["16", "5"])        # no such clip
        w.writerow(["2", "lots"])      # not a number
        w.writerow(["3", "300"])
    fb_main(["--library", str(lib), "import", str(rows), "--clips-json", str(mf)])
    out = capsys.readouterr().out
    assert "line 3: Clip 16 doesn't exist" in out and "line 4: a number couldn't be read" in out
    assert "Imported 2 row(s), skipped 2" in out
    saved = json.loads(lib.read_text())
    assert set(saved) == {"vod@0.0", "vod@120.0"}


def test_record_still_stops_on_unknown_clip(tmp_path):
    mf = tmp_path / "clips.json"
    save_manifest(mf, [clip(0)], video="v.mp4")
    with pytest.raises(SystemExit, match="Clip 5 doesn't exist"):
        fb_main(["--library", str(tmp_path / "l.json"), "record", str(mf), "5", "--views", "1"])
