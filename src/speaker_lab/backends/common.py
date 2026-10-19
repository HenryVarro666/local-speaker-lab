"""Small helpers shared by adapters; all ML imports stay inside worker calls."""

import importlib
from pathlib import Path
from typing import Any


def dependency(name: str, environment: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import {name!r} in this worker. Install/check the {environment} "
            "environment with scripts/install_ubuntu.sh, and check its python path in config. "
            f"Original error: {exc}"
        ) from exc


def absolute_path(value: str, *, exists: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Worker paths must be absolute: {value!r}")
    if exists and not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_audio(value: str):
    sf = dependency("soundfile", "core")
    np = dependency("numpy", "core")
    path = absolute_path(value)
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != 16000 or audio.ndim != 1 or not len(audio):
        raise ValueError(f"Expected nonempty 16kHz mono WAV: {path} (rate={rate})")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    return audio


def audio_dict(value: str, torch: Any) -> dict[str, Any]:
    # In-memory input also avoids native TorchCodec decoder/version problems.
    return {"waveform": torch.from_numpy(read_audio(value)).unsqueeze(0), "sample_rate": 16000}


def torch_device(torch: Any, value: str):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this worker. Check the NVIDIA "
            "driver and Blackwell-compatible PyTorch wheel using scripts/doctor.py."
        )
    return device
