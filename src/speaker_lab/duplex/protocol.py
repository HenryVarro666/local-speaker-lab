"""Portable manifests and create-only, content-verified experiment protocols."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import tomllib


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: Path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audio_path(record: dict, data_root: Path) -> Path:
    relative = Path(record["audio_relpath"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Audio path must remain inside data_root")
    root = Path(data_root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Audio resolves outside data_root")
    if not target.is_file():
        raise ValueError(f"Missing audio: {relative}")
    return target


def load_manifest(path: Path, data_root: Path) -> list[dict]:
    records, ids, groups, hashes = [], set(), {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(row, dict) or type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("ClipRecord schema_version must be 1")
        for key in ("clip_id", "audio_relpath", "audio_sha256", "source_id", "source_group_id",
                    "reference", "license_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing nonempty {key}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["clip_id"]):
            raise ValueError("Unsafe clip_id")
        if row["clip_id"] in ids:
            raise ValueError(f"Duplicate clip_id: {row['clip_id']}")
        for key, choices in {
            "split": {"train", "dev", "test"}, "origin": {"real", "synthetic", "fixture"},
            "language_group": {"zh", "en", "mixed"},
            "label_status": {"source_reference", "needs_review", "human_approved"},
        }.items():
            if not isinstance(row.get(key), str) or row[key] not in choices:
                raise ValueError(f"Invalid {key}")
        for key in ("sample_rate", "channels", "frames", "duration_seconds"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        path_audio = audio_path(row, data_root)
        if not re.fullmatch(r"[0-9a-f]{64}", row["audio_sha256"]) or file_hash(path_audio) != row["audio_sha256"]:
            raise ValueError(f"Audio hash mismatch: {row['clip_id']}")
        for key, index in ((row["source_group_id"], groups), (row["audio_sha256"], hashes)):
            if key in index and index[key] != row["split"]:
                raise ValueError("Source family or exact audio crosses split boundaries")
            index[key] = row["split"]
        if row["origin"] == "synthetic":
            for key in ("script_id", "voice_id", "style_requested", "generator_model", "generator_revision"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing synthetic {key}")
            row.setdefault("style_verified", False)
            if not isinstance(row["style_verified"], bool):
                raise ValueError("style_verified must be boolean")
            script_key = ("script", row["script_id"])
            if script_key in groups and groups[script_key] != row["split"]:
                raise ValueError("Synthetic script crosses split boundaries")
            groups[script_key] = row["split"]
        records.append(row)
        ids.add(row["clip_id"])
    if not records:
        raise ValueError("Manifest is empty")
    return records


def freeze_protocol(manifest: Path, data_root: Path, config: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records = load_manifest(manifest, data_root)
    config_text = Path(config).read_text(encoding="utf-8")
    tomllib.loads(config_text)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "protocol"
        stage.mkdir()
        (stage / "manifest.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in records), encoding="utf-8")
        (stage / "config.toml").write_text(config_text, encoding="utf-8")
        write_json(stage / "analysis_plan.json", {
            "schema_version": 1, "count": len(records), "missing_predictions": "reject_comparison",
            "real_and_synthetic": "report_separately", "normalization": "glossary_metrics_v1",
        })
        files = {name: file_hash(stage / name) for name in ("manifest.jsonl", "config.toml", "analysis_plan.json")}
        frozen = {"schema_version": 1, "files": files, "protocol_hash": content_hash(files)}
        write_json(stage / "FREEZE.json", frozen)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return frozen


def verify_protocol(protocol_dir: Path, data_root: Path) -> list[dict]:
    root = Path(protocol_dir)
    frozen = read_json(root / "FREEZE.json")
    files = frozen.get("files", {})
    if set(files) != {"manifest.jsonl", "config.toml", "analysis_plan.json"}:
        raise ValueError("Invalid frozen files")
    if frozen.get("protocol_hash") != content_hash(files):
        raise ValueError("Protocol hash mismatch")
    for name, expected in files.items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Frozen file hash mismatch: {name}")
    return load_manifest(root / "manifest.jsonl", data_root)
