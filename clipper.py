#!/usr/bin/env python3
"""
AI clip finder for gaming streams. Thin entry point; the code lives in clipfinder/.

Usage:
    python clipper.py --video stream.mp4 --transcript stream.srt
    python clipper.py --video stream.mp4 --transcribe --preview      # cheap 480p previews first
    python clipper.py --clips-json clips/clips.json --vertical --burn-captions
    python clipper.py --video stream.mp4 --transcript stream.srt --batch --triage-model claude-haiku-5-5

Run with --help for every option.
"""

from clipfinder.cli import main

if __name__ == "__main__":
    main()
