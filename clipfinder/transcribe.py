"""Whisper transcription. Extracts a small 16 kHz mono audio track first."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


def transcript_path_for(video: Path) -> Path:
    return video.with_suffix(".transcript.json")


def extract_audio_cmd(video: Path, out: Path) -> list[str]:
    # 16 kHz mono is what Whisper uses internally. FLAC keeps it lossless at ~1 MB/min,
    # versus 50-100+ MB/min for 1080p video, so uploads and decoding are far cheaper.
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(out)]


def extract_audio(video: Path, out: Path) -> Path:
    subprocess.run(extract_audio_cmd(video, out), check=True)
    return out


def looks_hallucinated(no_speech_prob: float | None, avg_logprob: float | None) -> bool:
    """Whisper invents filler ("you", "Thank you.") over silence. Those segments
    come with a high no-speech probability *and* low confidence; real speech,
    even a shouted "BOOM!", doesn't."""
    if no_speech_prob is None or avg_logprob is None:
        return False
    return no_speech_prob > 0.6 and avg_logprob < -0.5


def _transcribe_mlx(audio: Path, model_size: str, progress: bool = True) -> list[dict]:
    """Whisper on the Apple Silicon GPU."""
    import mlx_whisper
    repo = MLX_MODELS.get(model_size, model_size)
    # verbose=False shows a progress bar; None is silent (no bar, no language line).
    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=repo,
                                    verbose=False if progress else None,
                                    condition_on_previous_text=False, word_timestamps=True)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
             "words": [{"start": w["start"], "end": w["end"], "word": w["word"].strip()}
                       for w in s.get("words", [])]}
            for s in result["segments"] if s["text"].strip()
            and not looks_hallucinated(s.get("no_speech_prob"), s.get("avg_logprob"))]


_CPU_MODELS: dict = {}  # loaded faster-whisper models, so live mode doesn't reload per piece


def _transcribe_cpu(audio: Path, model_size: str, progress: bool = True) -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("--transcribe needs mlx-whisper (Apple Silicon) or faster-whisper:\n"
                 "  pip install mlx-whisper   or   pip install faster-whisper")
    model = _CPU_MODELS.get(model_size)
    if model is None:
        model = _CPU_MODELS[model_size] = WhisperModel(model_size, compute_type="auto")
    # VAD skips silence; word timestamps drive the animated captions.
    segments, _ = model.transcribe(str(audio), vad_filter=True, word_timestamps=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "words": [{"start": w.start, "end": w.end, "word": w.word.strip()}
                       for w in (s.words or [])]}
            for s in segments
            if not looks_hallucinated(getattr(s, "no_speech_prob", None),
                                      getattr(s, "avg_logprob", None))]


def transcribe_audio(audio: Path, model_size: str = "small", progress: bool = True) -> list[dict]:
    """Whisper segments ({start, end, text, words}) for an audio file, GPU if available."""
    use_mlx = importlib.util.find_spec("mlx_whisper") is not None
    return (_transcribe_mlx if use_mlx else _transcribe_cpu)(audio, model_size, progress)


def transcribe(video: Path, model_size: str = "small", keep_audio: bool = False) -> Path:
    """Make a Whisper JSON transcript next to the video (GPU if available)."""
    use_mlx = importlib.util.find_spec("mlx_whisper") is not None
    run = _transcribe_mlx if use_mlx else _transcribe_cpu
    print(f"Transcribing {video.name} with Whisper '{model_size}' "
          f"on the {'GPU' if use_mlx else 'CPU'}...")

    if video.suffix.lower() in AUDIO_EXTS:
        out = run(video, model_size)
    elif keep_audio:
        audio = extract_audio(video, video.with_suffix(".16k.flac"))
        print(f"Saved audio track -> {audio}")
        out = run(audio, model_size)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            audio = extract_audio(video, Path(tmp) / "audio.flac")
            out = run(audio, model_size)

    path = transcript_path_for(video)
    path.write_text(json.dumps({"segments": out}, indent=1), encoding="utf-8")
    print(f"Saved transcript -> {path}")
    return path
