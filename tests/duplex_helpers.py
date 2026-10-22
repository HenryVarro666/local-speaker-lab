import hashlib
import json

import numpy as np
import soundfile as sf


def make_clip(root, clip_id="clip1", *, split="dev", origin="real", seconds=1.2):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{clip_id}.wav"
    samples = 0.2 * np.sin(2 * np.pi * 230 * np.arange(round(16000 * seconds)) / 16000)
    sf.write(path, samples, 16000, subtype="PCM_16")
    return {
        "schema_version": 1, "clip_id": clip_id, "audio_relpath": path.name,
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_id": "test-source", "source_group_id": clip_id, "split": split,
        "language_group": "mixed", "reference": "明天 at four thirty",
        "origin": origin, "license_id": "CC0-1.0", "label_status": "source_reference",
        "sample_rate": 16000, "channels": 1, "frames": len(samples),
        "duration_seconds": len(samples) / 16000,
    }


def write_manifest(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path
