"""Audio auditing without changing original recordings or normalizing gains."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from .protocol import audio_path, content_hash, file_hash


def audit_clip(record: dict, data_root: Path, derived_root: Path, rules: dict) -> dict:
    result = {"schema_version": 1, "clip_id": record["clip_id"], "split": record["split"],
              "audio_sha256": record["audio_sha256"], "rule_hash": content_hash(rules),
              "decision": "reject", "reasons": []}
    path = audio_path(record, data_root)
    if file_hash(path) != record["audio_sha256"]:
        raise ValueError("Audio hash changed before audit")
    try:
        info = sf.info(path)
    except (RuntimeError, OSError, ValueError):
        result["reasons"] = ["decode_error"]
        return result
    if info.frames <= 0 or info.samplerate <= 0:
        result["reasons"] = ["empty"]
        return result
    duration = info.frames / info.samplerate
    result.update(sample_rate=info.samplerate, channels=info.channels, frames=info.frames, duration_seconds=duration)
    if not rules.get("min_seconds", 1.0) <= duration <= rules.get("max_seconds", 12.0):
        result["reasons"] = ["duration"]
        return result
    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        result["reasons"] = ["decode_error"]
        return result
    if not np.isfinite(samples).all():
        result["reasons"] = ["nonfinite"]
        return result
    if not np.any(samples):
        result["reasons"] = ["silent"]
        return result
    if any(record[key] != result[key] for key in ("sample_rate", "channels", "frames")):
        result["reasons"] = ["metadata_mismatch"]
        return result
    mono = samples.mean(axis=1)
    if not np.any(mono):
        result["reasons"] = ["channel_cancellation"]
        return result
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    rms_dbfs = 20 * math.log10(rms)
    clipped = float(np.mean(np.abs(samples) >= .999))
    window = max(1, round(rate * .02))
    silence_threshold = rules.get("quiet_dbfs", -50.0)
    windows = [mono[pos:pos + window].astype(np.float64) for pos in range(0, len(mono), window)]
    silence_fraction = sum(float(np.mean(block ** 2)) < 10 ** (silence_threshold / 10) for block in windows) / len(windows)
    reasons = []
    if clipped > rules.get("clipping_fraction", .01):
        reasons.append("clipping")
    if rms_dbfs < silence_threshold:
        reasons.append("quiet")
    if silence_fraction >= rules.get("silence_fraction", .8):
        reasons.append("mostly_silence")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    mono = np.ascontiguousarray(mono, dtype="<f4")
    derived = Path(derived_root)
    derived.mkdir(parents=True, exist_ok=True)
    destination = derived / f"{record['clip_id']}.wav"
    if destination.is_symlink() or destination.resolve() == path:
        raise ValueError("Derived audio must not replace its source or follow a symlink")
    sf.write(destination, mono, 16000, subtype="FLOAT")
    result.update(decision="needs_review" if reasons else "pass", reasons=reasons,
                  rms_dbfs=rms_dbfs, clipped_fraction=clipped, silence_fraction=silence_fraction,
                  normalized_path=str(destination.resolve()), normalized_sha256=file_hash(destination),
                  pcm_sha256=hashlib.sha256(mono.tobytes()).hexdigest(),
                  decoder={"soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__})
    return result


def find_exact_duplicates(records: list[dict]) -> list[dict]:
    indexes = {"audio_sha256": {}, "pcm_sha256": {}}
    duplicates = []
    for record in sorted(records, key=lambda r: r["clip_id"]):
        matches = []
        for field, index in indexes.items():
            value = record.get(field)
            if not value:
                continue
            if value in index:
                other = index[value]
                if other["split"] != record["split"]:
                    raise ValueError("Exact duplicate audio crosses split boundaries")
                matches.append(other["clip_id"])
            else:
                index[value] = record
        if matches:
            duplicates.append({"clip_id": record["clip_id"], "duplicate_of": min(matches)})
    return duplicates
