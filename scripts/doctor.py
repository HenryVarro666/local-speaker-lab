#!/usr/bin/env python3
"""Check installed runtimes, actual CUDA kernels, and local Ollama; no downloads."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def probe_numpy_bridge(torch) -> None:
    """Exercise the ABI bridge used by all workers, not just package imports."""
    import numpy as np

    expected = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
    actual = torch.from_numpy(expected).clone().numpy()
    np.testing.assert_array_equal(actual, expected)
    print("  NumPy <-> torch float32 round trip: PASS")


def probe_audio_decode(*, torchcodec: bool = False) -> None:
    """Decode our actual float-WAV interchange format through native libraries."""
    import numpy as np
    import soundfile as sf

    expected = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, expected, 16000, format="WAV", subtype="FLOAT")
    encoded = buffer.getvalue()
    actual, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32")
    if sample_rate != 16000:
        raise RuntimeError("libsndfile changed the WAV sample rate")
    np.testing.assert_array_equal(actual, expected)
    print("  libsndfile float-WAV encode/decode: PASS")
    if torchcodec:
        # A successful import can still leave FFmpeg/libtorch loading or decoder
        # failures undiscovered until a real file is opened.
        from torchcodec.decoders import AudioDecoder

        decoded = AudioDecoder(encoded, sample_rate=16000, num_channels=1).get_all_samples()
        actual = decoded.data.detach().cpu().numpy()
        if decoded.sample_rate != 16000 or actual.shape != (1, len(expected)):
            raise RuntimeError("TorchCodec changed float-WAV shape or sample rate")
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=1e-6)
        print("  TorchCodec native FFmpeg float-WAV decode: PASS")


def probe(stage: str) -> int:
    packages = {
        "app": [("numpy", "numpy"), ("scipy", "scipy"), ("soundfile", "soundfile"), ("gradio", "gradio")],
        "diarize": [("pyannote.audio", "pyannote.audio"), ("torchcodec", "torchcodec")],
        "match": [("pyannote.audio", "pyannote.audio")],
        "separate": [("clearvoice", "clearvoice")],
        "separate-sb": [("speechbrain", "speechbrain")],
        "transcribe": [("qwen_asr", "qwen-asr"), ("transformers", "transformers")],
    }
    try:
        for module, distribution in packages[stage]:
            importlib.import_module(module)
            print(f"  {distribution}: {importlib.metadata.version(distribution)}")
        probe_audio_decode(torchcodec=stage in {"diarize", "match"})
        if stage != "app":
            import torch
            import torchaudio

            print(f"  torch={torch.__version__}, torchaudio={torchaudio.__version__}, CUDA wheel={torch.version.cuda}")
            probe_numpy_bridge(torch)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; check NVIDIA driver and the cu128 wheel")
            device = torch.device("cuda:0")
            props = torch.cuda.get_device_properties(device)
            print(f"  GPU={props.name}, capability={props.major}.{props.minor}, VRAM={props.total_memory / 2**30:.1f} GiB")
            # Import success alone does not catch missing Blackwell CUDA kernels.
            with torch.inference_mode():
                x = torch.randn(1, 4, 256, device=device)
                y = torch.nn.functional.conv1d(x, torch.randn(8, 4, 3, device=device))
                a = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
                z = a @ a.T
                query = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16, device=device)
                attention = torch.nn.functional.scaled_dot_product_attention(query, query, query)
                mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_fft=256, hop_length=64, n_mels=32
                ).to(device)(torch.randn(2048, device=device))
                torch.cuda.synchronize()
                if not all(torch.isfinite(output).all().item() for output in (y, z, attention, mel)):
                    raise RuntimeError("CUDA computation produced non-finite values")
            print("  CUDA float32 convolution + bfloat16 matrix multiplication/SDPA + STFT/mel: PASS")
        return 0
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.toml"))
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--with-speechbrain", action="store_true")
    parser.add_argument("--probe", choices=["app", "diarize", "match", "separate", "separate-sb", "transcribe"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        return probe(args.probe)
    config_path = Path(args.config).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project_root = (config_path.parent / config.get("project", {}).get("root", "..")).resolve()
    env = os.environ.copy()
    env.update(HF_HUB_DISABLE_TELEMETRY="1", PYANNOTE_METRICS_ENABLED="0", GRADIO_ANALYTICS_ENABLED="False")
    failures = 0
    for command in ["ffmpeg", "nvidia-smi"]:
        available = shutil.which(command)
        print(f"{command}: {available or 'MISSING'}")
        failures += int(not available)
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], check=False)
        failures += int(result.returncode != 0)
    print(f"HF_TOKEN: {'set (value hidden)' if os.environ.get('HF_TOKEN') else 'not set (gated downloads require access or cached files)'}")
    interpreters = {"app": str(project_root / ".venvs/app/bin/python")}
    for stage in ["diarize", "separate", "match", "transcribe"]:
        value = config.get("workers", {}).get(f"{stage}_python")
        if value:
            path = Path(value)
            interpreters[stage] = str(path if path.is_absolute() else project_root / path)
        else:
            print(f"{stage}: missing [workers].{stage}_python", file=sys.stderr)
            failures += 1
    if args.with_speechbrain:
        interpreters["separate-sb"] = str(project_root / ".venvs/separate-sb/bin/python")
    for stage, executable in interpreters.items():
        print(f"\n[{stage}] {executable}", flush=True)
        try:
            probe_stage = "separate-sb" if stage == "separate" and config.get("separate", {}).get("backend") == "sepformer" else stage
            result = subprocess.run([executable, str(Path(__file__).resolve()), "--probe", probe_stage], cwd=project_root, env=env, timeout=180, check=False)
            failures += int(result.returncode != 0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            failures += 1
    if not args.skip_ollama:
        llm = config.get("summary", {})
        base_url = str(llm.get("url", "http://127.0.0.1:11434")).rstrip("/")
        model = llm.get("model", "qwen3:8b")
        print(f"\n[ollama] {base_url}; model={model}")
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as response:
                models = json.load(response).get("models", [])
            names = {item.get("name") for item in models} | {item.get("model") for item in models}
            if model not in names:
                raise RuntimeError(f"Model absent. Run: ollama pull {model}")
            print("  Server reachable; model is installed (generation/GPU placement not tested)")
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            failures += 1
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failed checks. No model weights were downloaded.")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
