from __future__ import annotations

import math
import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    root = (path.parent / config.get("project", {}).get("root", "..")).resolve()
    config["_root"] = str(root)
    config["_path"] = str(path)
    pipeline = config.setdefault("pipeline", {})
    defaults = {
        "output_dir": "runs", "sample_rate": 16000, "max_duration_seconds": 1800,
        "separation_chunk_seconds": 12.0, "context_seconds": 0.5,
        "asr_chunk_seconds": 20.0, "min_reference_seconds": 1.0, "reference_seconds": 12.0,
    }
    for key, value in defaults.items():
        pipeline.setdefault(key, value)
    if pipeline["sample_rate"] != 16000:
        raise ValueError("This MVP requires sample_rate=16000; model adapters resample internally if needed.")
    for key in defaults.keys() - {"output_dir", "sample_rate"}:
        if isinstance(pipeline[key], bool) or not isinstance(pipeline[key], (int, float)):
            raise ValueError(f"pipeline.{key} must be a number.")
        value = float(pipeline[key])
        if not math.isfinite(value) or value < 0 or (key != "context_seconds" and value == 0):
            raise ValueError(f"Invalid pipeline.{key}: {value}")
        if key != "context_seconds" and round(value * pipeline["sample_rate"]) < 1:
            raise ValueError(f"pipeline.{key} must cover at least one audio sample.")
        pipeline[key] = value
    if pipeline["separation_chunk_seconds"] + 2 * pipeline["context_seconds"] > 20:
        raise ValueError("Separation chunk + 2*context must be <=20 seconds for the default adapter.")
    if pipeline["reference_seconds"] < pipeline["min_reference_seconds"]:
        raise ValueError("reference_seconds must be >= min_reference_seconds.")
    for stage in ("diarize", "separate", "match", "transcribe"):
        if not config.get(stage, {}).get("backend"):
            raise ValueError(f"Missing [{stage}].backend")
        if not config.get("workers", {}).get(f"{stage}_python"):
            raise ValueError(f"Missing [workers].{stage}_python")
    timeout = config["workers"].get("timeout_seconds", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("workers.timeout_seconds must be a positive finite number.")
    return config


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_root"]) / path).resolve()
