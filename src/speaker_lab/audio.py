from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize_audio(source: str | Path, destination: Path, max_seconds: float) -> np.ndarray:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Audio file does not exist: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found. On Ubuntu: sudo apt install ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-t", str(max_seconds + 1), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(destination)],
        capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not decode audio: {result.stderr[-2000:]}")
    audio, sample_rate = sf.read(destination, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1 or not len(audio):
        raise ValueError("Decoded audio must be nonempty 16 kHz mono.")
    if len(audio) > round(max_seconds * sample_rate):
        raise ValueError(f"Audio exceeds the configured limit of {max_seconds:g} seconds.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinity.")
    if np.max(np.abs(audio)) < 1e-7:
        raise ValueError("Audio is silent.")
    return audio


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Floating PCM preserves upstream gains without clipping or renormalizing each track.
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate, subtype="FLOAT")
    return str(path.resolve())


def read_source(path: str | Path, expected_samples: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000 or audio.ndim != 1 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Backend must return finite mono 16 kHz audio: {path}")
    # Neural encoders sometimes pad their last frame, but large timing differences are errors.
    if abs(len(audio) - expected_samples) > 1600:
        raise ValueError(f"Backend changed audio duration: {path}")
    return np.pad(audio[:expected_samples], (0, max(0, expected_samples - len(audio))))
