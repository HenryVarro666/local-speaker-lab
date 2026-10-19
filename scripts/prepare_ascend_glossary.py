#!/usr/bin/env python3
"""Export one official ASCEND evaluation split for the glossary experiment.

Only the requested split Parquet is downloaded. Optional dependencies are imported
after argument parsing, so --help works without installing the data tooling.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import itertools
import json
from math import gcd
from pathlib import Path
import re
import tempfile
from typing import Iterable


REPO_ID = "CAiRE/ASCEND"
LICENSE = "CC-BY-SA-4.0"
SPLIT_FILES = {
    split: f"main/{split}-00000-of-00001.parquet"
    for split in ("validation", "test")
}
REQUIRED_FIELDS = {"id", "audio", "transcription", "language"}
GROUPS = {"zh": "mandarin", "en": "english", "mixed": "mixed"}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def resolve_split(split: str, revision: str, *, api=None, downloader=None):
    """Return (local Parquet path, resolved commit, source filename)."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unsupported split: {split}")
    if api is None or downloader is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub and pyarrow to import ASCEND") from exc
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    info = api.dataset_info(repo_id=REPO_ID, revision=revision)
    commit = info.sha
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Hugging Face did not return a valid immutable dataset commit")
    filename = SPLIT_FILES[split]
    if filename not in {item.rfilename for item in info.siblings}:
        raise ValueError(f"Official split file missing at {commit}: {filename}")
    path = downloader(repo_id=REPO_ID, repo_type="dataset", revision=commit, filename=filename)
    return Path(path), commit, filename


def read_parquet(path: Path):
    """Validate the physical Parquet schema and return (row iterator, row count)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow and huggingface_hub to import ASCEND") from exc
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if not REQUIRED_FIELDS.issubset(schema.names):
        raise ValueError(f"Invalid ASCEND schema: requires {sorted(REQUIRED_FIELDS)}")
    for name in ("id", "transcription", "language"):
        kind = schema.field(name).type
        if not (pa.types.is_string(kind) or pa.types.is_large_string(kind)):
            raise ValueError(f"Invalid ASCEND schema: {name} must be a string")
    audio_type = schema.field("audio").type
    if not pa.types.is_struct(audio_type) or "bytes" not in [field.name for field in audio_type]:
        raise ValueError("Invalid ASCEND schema: audio must contain embedded bytes")
    byte_type = audio_type.field("bytes").type
    if not (pa.types.is_binary(byte_type) or pa.types.is_large_binary(byte_type)):
        raise ValueError("Invalid ASCEND schema: audio.bytes must be binary")

    def rows():
        for batch in parquet.iter_batches(batch_size=32, columns=sorted(REQUIRED_FIELDS)):
            yield from batch.to_pylist()

    return rows(), parquet.metadata.num_rows


def write_audio(data: bytes, path: Path) -> None:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid audio for {path.name}: empty, nonfinite, or invalid sample rate")
    mono = samples.mean(axis=1)
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, sample_rate // divisor)
    sf.write(path, mono, 16000, subtype="PCM_16")


def export_rows(
    rows: Iterable[dict], output_dir: Path, *, split: str, commit: str,
    requested_revision: str, source_filename: str, official_count: int,
    limit: int | None = None,
) -> dict:
    """Export atomically into a new directory; references never become prompts."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_dir}")
    if split not in SPLIT_FILES or official_count < 1 or (limit is not None and limit < 1):
        raise ValueError("Invalid split, row count, or limit")
    expected_count = min(official_count, limit) if limit is not None else official_count
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    seen = set()
    # TemporaryDirectory owns only this newly created sibling directory. Failed
    # conversions leave neither a completed manifest nor partial user output.
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "audio").mkdir()
        with (staging / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest:
            selected = itertools.islice(rows, expected_count)
            for row in selected:
                if not isinstance(row, dict) or not REQUIRED_FIELDS.issubset(row):
                    raise ValueError("Invalid ASCEND row schema")
                clip_id, reference, language = row["id"], row["transcription"], row["language"]
                if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", clip_id):
                    raise ValueError(f"Invalid or unsafe clip id: {clip_id!r}")
                if clip_id in seen:
                    raise ValueError(f"Duplicate clip id: {clip_id}")
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"Empty or invalid reference for {clip_id}")
                if not isinstance(language, str) or not language.strip():
                    raise ValueError(f"Empty or invalid language metadata for {clip_id}")
                audio = row["audio"]
                if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes) or not audio["bytes"]:
                    raise ValueError(f"Missing embedded audio bytes for {clip_id}")
                relative_audio = f"audio/{clip_id}.wav"
                write_audio(audio["bytes"], staging / relative_audio)
                group = GROUPS.get(language, language)
                manifest.write(json.dumps({
                    "id": clip_id, "audio": relative_audio, "reference": reference,
                    "split": split, "group": group,
                }, ensure_ascii=False) + "\n")
                seen.add(clip_id)
                counts[group] += 1
        if len(seen) != expected_count:
            raise ValueError(f"Expected {expected_count} rows but received {len(seen)}")
        metadata = {
            "schema_version": 1,
            "source": {
                "repo_id": REPO_ID, "revision": commit,
                "requested_revision": requested_revision, "filename": source_filename,
                "url": f"https://huggingface.co/datasets/{REPO_ID}", "license": LICENSE,
            },
            "split": split, "count": len(seen), "official_split_count": official_count,
            "limit": limit, "subset": len(seen) < official_count,
            "selection": "first_n_in_official_order" if len(seen) < official_count else "official_split",
            "groups": dict(counts), "manifest": "manifest.jsonl",
            "sample_rate": 16000, "channels": 1, "audio_subtype": "PCM_16",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "dataset.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during conversion: {output_dir}")
        staging.rename(output_dir)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLIT_FILES), required=True)
    parser.add_argument("--revision", default="main", help="HF revision; resolved to an immutable commit")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--limit", type=positive_int, help="Export first N rows; dataset.json labels a subset")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.expanduser().resolve().exists():
            raise FileExistsError(f"Output already exists: {args.output_dir}")
        # Check the optional reader before downloading a roughly 106 MB file.
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install pyarrow and huggingface_hub to import ASCEND") from exc
        parquet, commit, filename = resolve_split(args.split, args.revision)
        rows, count = read_parquet(parquet)
        metadata = export_rows(
            rows, args.output_dir, split=args.split, commit=commit,
            requested_revision=args.revision, source_filename=filename,
            official_count=count, limit=args.limit,
        )
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    label = "SUBSET" if metadata["subset"] else "official split"
    print(f"Exported {metadata['count']} {args.split} rows ({label}) to {args.output_dir.resolve()}")
    print(f"Source: {REPO_ID}@{commit}; license: {LICENSE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
