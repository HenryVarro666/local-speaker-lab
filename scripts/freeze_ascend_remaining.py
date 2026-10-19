"""Freeze every duration-eligible validation clip outside a prior cohort.

Selection reads only clip IDs, language groups and WAV headers. Reference text
is copied into the local runtime manifest without influencing eligibility or
ordering. No vocabulary, model predictions, or official test data are opened.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import stat
import wave

DEFAULT_SEED = 20261018
GROUP_ORDER = ("mixed", "mandarin", "english")
MIN_SECONDS = 1
MAX_SECONDS = 12


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    """Use explicit UTF-8/LF bytes, including on Windows."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def safe_audio_path(source_dir, value):
    """Require a regular descendant file without symlinks or Windows junctions."""
    if not isinstance(value, str) or not value:
        raise ValueError("Every audio path must be a nonempty relative path")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if (windows.drive or windows.root or posix.is_absolute()
            or ".." in posix.parts or ":" in value):
        raise ValueError(f"Unsafe audio path: {value!r}")
    candidate = source_dir
    for part in posix.parts:
        candidate = candidate / part
        info = candidate.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"Audio path contains a symlink or reparse point: {value!r}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(source_dir) or not resolved.is_file():
        raise ValueError(f"Audio path is not a regular source file: {value!r}")
    return resolved


def load_candidates(source, data):
    """Validate metadata and WAV headers without inspecting reference values."""
    rows, seen = [], set()
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Manifest line {line_number} must be an object")
        clip_id = item.get("id")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"Manifest line {line_number} has an invalid ID")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip ID: {clip_id}")
        seen.add(clip_id)
        if item.get("split") != "validation":
            raise ValueError(f"Only validation rows are allowed: {clip_id}")
        if item.get("group") not in GROUP_ORDER:
            raise ValueError(f"Unknown language group: {clip_id}")
        audio = safe_audio_path(source.parent, item.get("audio"))
        with wave.open(str(audio), "rb") as stream:
            if (stream.getframerate() != 16000 or stream.getnchannels() != 1
                    or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE"):
                raise ValueError(f"Audio must be 16kHz mono PCM_16 WAV: {clip_id}")
            frames = stream.getnframes()
        if frames <= 0:
            raise ValueError(f"Audio must contain frames: {clip_id}")
        rows.append({"id": clip_id, "group": item["group"],
                     "duration_seconds": frames / 16000,
                     "audio_path": audio, "source_row": item})
    if not rows:
        raise ValueError("Source validation manifest is empty")
    return rows


def freeze(source_manifest, prior_protocol, output_dir, seed=DEFAULT_SEED):
    """Validate all inputs before creating a new, non-overwritable output."""
    source = Path(source_manifest).resolve(strict=True)
    prior = Path(prior_protocol).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    prior_ids_path = prior / "selected_ids.json" if prior.is_dir() else prior
    source_data = source.read_bytes()
    prior_data = prior_ids_path.read_bytes()
    prior_ids = json.loads(prior_data.decode("utf-8-sig"))
    if (not isinstance(prior_ids, list)
            or any(not isinstance(value, str) or not value for value in prior_ids)):
        raise ValueError("Prior selected IDs must be a list of nonempty strings")
    if len(set(prior_ids)) != len(prior_ids):
        raise ValueError("Duplicate IDs in prior protocol")
    rows = load_candidates(source, source_data)
    eligible = [row for row in rows if MIN_SECONDS <= row["duration_seconds"] <= MAX_SECONDS]
    eligible_by_id = {row["id"]: row for row in eligible}
    missing = sorted(set(prior_ids) - set(eligible_by_id))
    if missing:
        raise ValueError(f"Prior IDs are missing or duration-ineligible: {', '.join(missing)}")

    hashes, hash_owner = {}, {}
    for row in eligible:
        digest = file_hash(row["audio_path"])
        if digest in hash_owner:
            previous = hash_owner[digest]
            raise ValueError(f"Duplicate eligible audio: {previous['id']} ({previous['group']}) "
                             f"and {row['id']} ({row['group']})")
        hashes[row["id"]] = digest
        hash_owner[digest] = row

    old_ids = set(prior_ids)
    chosen = []
    for group in GROUP_ORDER:
        pool = [row for row in eligible if row["group"] == group and row["id"] not in old_ids]
        pool.sort(key=lambda row: (sha256_bytes(f"{seed}:{row['id']}".encode()), row["id"]))
        chosen.extend(pool)
    if not chosen:
        raise ValueError("No remaining eligible validation clips")
    selected_ids = [row["id"] for row in chosen]
    runtime_rows = []
    for row in chosen:
        item = dict(row["source_row"])
        item["audio"] = row["audio_path"].as_posix()
        runtime_rows.append(item)
    manifest_data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in runtime_rows)
                     + "\n").encode("utf-8")
    selected_data = json_bytes(selected_ids)
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source), "source_manifest_sha256": sha256_bytes(source_data),
        "prior_selected_ids": str(prior_ids_path),
        "prior_selected_ids_sha256": sha256_bytes(prior_data),
        "sampling_source_code_sha256": file_hash(Path(__file__).resolve()),
        "seed": seed, "split": "validation",
        "selection": "all remaining 1-12s clips; group then sha256(seed:id), id",
        "group_order": list(GROUP_ORDER), "reference_not_used_for_selection": True,
        "vocabulary_or_predictions_used_for_selection": False,
        "official_test_opened": False,
        "min_duration_seconds": MIN_SECONDS, "max_duration_seconds": MAX_SECONDS,
        "source_count": len(rows), "source_groups": dict(Counter(row["group"] for row in rows)),
        "eligible_count": len(eligible),
        "eligible_groups": dict(Counter(row["group"] for row in eligible)),
        "excluded_duration_count": len(rows) - len(eligible),
        "excluded_duration_groups": dict(Counter(row["group"] for row in rows
                                                if row["id"] not in eligible_by_id)),
        "prior_count": len(prior_ids),
        "prior_groups": dict(Counter(eligible_by_id[clip_id]["group"] for clip_id in prior_ids)),
        "count": len(chosen), "groups": dict(Counter(row["group"] for row in chosen)),
        "total_audio_seconds": sum(row["duration_seconds"] for row in chosen),
        "audio_hash_audit": {"scope": "all duration-eligible clips, including prior cohort",
                             "checked": len(hashes), "unique": len(hash_owner),
                             "duplicates": 0, "cross_group_duplicates": 0},
        "eligible_audio_sha256": dict(sorted(hashes.items())),
        "selected_ids_sha256": sha256_bytes(selected_data),
        "manifest_sha256": sha256_bytes(manifest_data),
        "runtime_manifest_platform": platform.system(),
        "runtime_manifest_has_absolute_audio_paths": True,
        "runtime_manifest_portable": False,
        "portability_note": "Regenerate the local manifest on another machine; selected IDs are portable.",
    }
    if prior.is_dir() and (prior / "protocol.json").is_file():
        audit["prior_protocol_sha256"] = file_hash(prior / "protocol.json")
    payloads = {"manifest.jsonl": manifest_data, "selected_ids.json": selected_data,
                "selection_audit.json": json_bytes(audit)}
    # No filesystem mutation occurs until validation, hashing and serialization
    # have all succeeded. mkdir(exist_ok=False) also rejects concurrent writers.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prior-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        result = freeze(args.source_manifest, args.prior_protocol, args.output_dir, seed=args.seed)
    except (ValueError, OSError, wave.Error) as exc:
        parser.exit(1, f"Freeze failed: {exc}\n")
    print(json.dumps({key: result[key] for key in
                      ("source_count", "eligible_count", "prior_count", "count", "groups",
                       "excluded_duration_count", "total_audio_seconds", "manifest_sha256")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
