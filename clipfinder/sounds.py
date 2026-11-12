"""What the loud moments *are*: laughter, screaming, gunfire, explosions, sirens, crashes.

The loudness scan says "something loud happened at 1:02:13"; this says "an explosion and
screaming". It runs PANNs (Cnn14, trained on Google's AudioSet: 527 everyday sounds) on the
stream's audio, locally and for free. Optional: `pip install panns-inference`. The model
(~330 MB) downloads to ~/panns_data the first time.

Each second gets a score per sound group; seconds where a group is both likely and well
above its level in the surrounding two minutes become events, so a stream with sirens the
whole time doesn't flag every second.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

from .signals import Event, merge_runs, rolling_median

SAMPLE_RATE = 32000
WINDOW = 2.0  # seconds of audio per prediction; one prediction per second (50% overlap)
BATCH = 32

DATA_DIR = Path.home() / "panns_data"  # where panns-inference expects its files
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
MODEL_FILE = "Cnn14_mAP=0.431.pth"
MODEL_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

# Sound group -> AudioSet labels. A group's score is its most likely label's score.
GROUPS: dict[str, list[str]] = {
    "laughter": ["Laughter", "Giggle", "Belly laugh", "Chuckle, chortle"],
    "screaming": ["Screaming", "Shout", "Yell", "Bellow", "Whoop"],
    "gunfire": ["Gunshot, gunfire", "Machine gun", "Artillery fire"],
    "explosion": ["Explosion", "Boom", "Fireworks"],
    "siren": ["Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"],
    "crash": ["Smash, crash", "Shatter"],
    "skidding": ["Skidding", "Tire squeal"],
    "cheering": ["Cheering", "Applause"],
}


def _download(url: str, dest: Path, min_bytes: int) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} (one time)...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download of {dest.name} looks incomplete")
    tmp.replace(dest)


def load_model():
    """PANNs Cnn14 on the Apple GPU when there is one (panns-inference only knows CUDA/CPU)."""
    # panns-inference fetches these with wget at import time, which macOS doesn't have.
    _download(LABELS_URL, DATA_DIR / "class_labels_indices.csv", 10_000)
    _download(MODEL_URL, DATA_DIR / MODEL_FILE, 300_000_000)
    import torch
    from panns_inference import AudioTagging
    from panns_inference.config import labels

    tagger = AudioTagging(checkpoint_path=str(DATA_DIR / MODEL_FILE), device="cpu")
    if torch.backends.mps.is_available():
        tagger.model.to("mps")
        tagger.device = "mps"
    index = {lb: i for i, lb in enumerate(labels)}
    groups = {g: [index[lb] for lb in lbs if lb in index] for g, lbs in GROUPS.items()}
    return tagger, groups


def decode_cmd(video: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vn",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]


def split_windows(samples, final: bool, sr: int = SAMPLE_RATE, window: float = WINDOW):
    """One window per second of audio, starting on each whole second.

    Returns (windows, used): with final=False only windows that fit entirely are made and
    `used` is how many samples can be dropped (the rest is kept for the next block); with
    final=True every remaining second gets a window, zero-padded at the end."""
    import numpy as np
    win = int(sr * window)
    if final:
        starts = range(0, len(samples), sr)
    else:
        starts = range(0, len(samples) - win + 1, sr) if len(samples) >= win else range(0)
    out = []
    for st in starts:
        chunk = samples[st:st + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        out.append(chunk)
    return out, len(starts) * sr


def tag_audio(video: Path, cache: Path | None = None, model=None) -> dict[str, list[float]]:
    """Score every second of the video for each sound group (0-1). Cached as JSON."""
    if cache and cache.exists():
        return json.loads(cache.read_text())
    import numpy as np
    import torch

    tagger, groups = model or load_model()
    scores: dict[str, list[float]] = {g: [] for g in groups}

    def run(batch):
        for i in range(0, len(batch), BATCH):
            with torch.no_grad():
                probs, _ = tagger.inference(np.stack(batch[i:i + BATCH]))
            for g, idx in groups.items():
                scores[g].extend(round(float(v), 3) for v in probs[:, idx].max(axis=1))

    # Stream the audio from ffmpeg in pieces: 10 h at 32 kHz would be ~4.6 GB in memory.
    proc = subprocess.Popen(decode_cmd(video), stdout=subprocess.PIPE)
    block = SAMPLE_RATE * 60 * 4  # 60 s of float32 per read
    carry = np.zeros(0, dtype=np.float32)
    try:
        while raw := proc.stdout.read(block):
            carry = np.concatenate([carry, np.frombuffer(raw, dtype=np.float32)])
            batch, used = split_windows(carry, final=False)
            run(batch)
            carry = carry[used:]
        batch, _ = split_windows(carry, final=True)
        if batch:
            run(batch)
    finally:
        proc.stdout.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError("ffmpeg couldn't decode the audio")
    if cache:
        cache.write_text(json.dumps(scores))
    return scores


# Game sounds under the streamer's voice score lower than clean recordings, so the bar is
# modest; the margin over the local median is what keeps constant background out.
def sound_events(scores: dict[str, list[float]], threshold: float = 0.25, margin: float = 0.1,
                 radius: int = 60) -> list[Event]:
    """Seconds where a sound group is likely and clearly above its usual level nearby."""
    events = []
    for group, vals in scores.items():
        if not vals:
            continue
        base = rolling_median(vals, radius)
        hits = [(float(t), v) for t, (v, b) in enumerate(zip(vals, base))
                if v >= threshold and v - b >= margin]
        events += [Event(a, b, "sound", round(peak, 2), group)
                   for a, b, peak in merge_runs(hits, 1.0, 2.0)]
    return sorted(events, key=lambda e: e.start)
