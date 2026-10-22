"""Speaker-disjoint selection and verified, create-only ASR training exports."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile

from speaker_lab.glossary_metrics import score_pair
from .protocol import audio_path, content_hash, file_hash, read_json, write_json


def split_by_speaker(records: list[dict], policy: dict) -> dict[str, list[dict]]:
    roles = {name: set(values) for name, values in policy.items()}
    seen = set()
    for values in roles.values():
        if seen & values:
            raise ValueError("A speaker occurs in multiple split roles")
        seen |= values
    output = {name: [] for name in [*roles, "train"]}
    for row in records:
        speaker = row.get("original_speaker_id")
        if speaker is None:
            raise ValueError("Missing official speaker metadata")
        role = next((k for k, values in roles.items() if speaker in values), "train")
        output[role].append(dict(row))
    return output


def sample_source_records(records: list[dict], quotas: dict, seed: int,
                          per_speaker_limit: int | None = None) -> list[dict]:
    chosen = []
    for language, count in sorted(quotas.items()):
        ordered = sorted((r for r in records if r["language_group"] == language),
                         key=lambda r: hashlib.sha256(f"{seed}:{r['clip_id']}".encode()).hexdigest())
        speakers = Counter()
        selected = []
        for row in ordered:
            speaker = row["original_speaker_id"]
            if per_speaker_limit is not None and speakers[speaker] >= per_speaker_limit:
                continue
            selected.append(row)
            speakers[speaker] += 1
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f"Insufficient source quota for {language}: {len(selected)}/{count}")
        chosen.extend(selected)
    return sorted(chosen, key=lambda r: r["clip_id"])


def stratum(row: dict) -> str:
    seconds = row["duration_seconds"]
    if not 1 <= seconds <= 12:
        raise ValueError("Training duration is outside frozen strata")
    bucket = "1-3" if seconds < 3 else "3-6" if seconds < 6 else "6-12"
    return f"{row['language_group']}|{bucket}"


def stratum_quotas(pool: list[dict], languages: dict) -> dict[str, int]:
    available = Counter(stratum(r) for r in pool if r.get("audit_decision") == "pass" and r["origin"] == "real")
    result = {}
    for language, count in sorted(languages.items()):
        bins = sorted(k for k in available if k.startswith(language + "|"))
        total = sum(available[k] for k in bins)
        if total < count:
            raise ValueError(f"Insufficient training quota: {language} {total}/{count}")
        exact = {k: count * available[k] / total for k in bins}
        allocation = {k: int(v) for k, v in exact.items()}
        for k in sorted(bins, key=lambda k: (-(exact[k] - allocation[k]), k))[:count - sum(allocation.values())]:
            allocation[k] += 1
        result.update(allocation)
    return result


def validate_partitions(parts: dict[str, list[dict]]) -> None:
    indexes = {key: {} for key in ["clip_id", "source_group_id", "original_speaker_id", "audio_sha256", "pcm_sha256"]}
    for role, rows in parts.items():
        for row in rows:
            for key, index in indexes.items():
                value = row.get(key)
                if value is None:
                    continue
                if value in index and index[value] != role:
                    raise ValueError(f"Cross-split partition leakage: {key}")
                index[value] = role


def select_training_arms(pool: list[dict], predictions: list[dict], reviews: list[dict], policy: dict,
                         audit: dict | None = None) -> dict[str, list[dict]]:
    predictions_by_id = {r["clip_id"]: r for r in predictions}
    if len(predictions_by_id) != len(predictions):
        raise ValueError("Duplicate prediction ID")
    approved = {r["clip_id"]: r for r in reviews if r.get("decision") == "approve" and r.get("listened") is True}
    buckets = defaultdict(list)
    duplicate_hashes = set()
    for original in sorted(pool, key=lambda r: r["clip_id"]):
        row = dict(original)
        if row["split"] != "train" or row["origin"] != "real" or row.get("audit_decision") == "reject":
            continue
        review = approved.get(row["clip_id"])
        if review and review.get("audio_sha256") == row["audio_sha256"]:
            row["source_reference"] = original["reference"]
            row["reference"] = review["corrected_text"]
            row["label_status"] = "human_approved"
            row["audit_resolution"] = "human_approved"
        elif row.get("audit_decision") != "pass":
            continue
        fingerprint = row.get("pcm_sha256", row["audio_sha256"])
        if fingerprint in duplicate_hashes:
            continue
        duplicate_hashes.add(fingerprint)
        buckets[stratum(row)].append(row)
    quotas = policy["quotas"]
    if any(len(buckets[k]) < n for k, n in quotas.items()):
        raise ValueError("Insufficient eligible quota for equal-budget arms")
    random_arm, filtered = [], []
    ranks = {}
    rng = random.Random(policy["seed"])
    for key, count in sorted(quotas.items()):
        rows = buckets[key]
        for row in rows:
            if row["clip_id"] not in predictions_by_id:
                raise ValueError("Missing training-pool prediction")
            scores = score_pair(row["reference"], predictions_by_id[row["clip_id"]]["hypothesis"], terms=[])
            row["selection_mer"] = (scores["S"] + scores["D"] + scores["I"]) / scores["N"] if scores["N"] else float("inf")
        ordered = sorted(rows, key=lambda r: (r["selection_mer"], r.get("silence_fraction", 1), r["clip_id"]))
        ranks.update({r["clip_id"]: i for i, r in enumerate(ordered)})
        random_arm.extend(rng.sample(sorted(rows, key=lambda r: r["clip_id"]), count))
        filtered.extend(ordered[:count])
    target = sum(r["duration_seconds"] for r in random_arm)
    limit = policy.get("max_duration_difference", .01) * target
    if audit is not None:
        audit.update(initial_filtered_ids=[r["clip_id"] for r in filtered], duration_swaps=[])
    # Deterministic, within-stratum budget correction, not a global optimizer.
    for _ in range(len(pool)):
        total = sum(r["duration_seconds"] for r in filtered)
        if abs(total - target) <= limit:
            break
        ids = {r["clip_id"] for r in filtered}
        best = None
        for index, remove in enumerate(filtered):
            for add in buckets[stratum(remove)]:
                if add["clip_id"] in ids:
                    continue
                gap = abs(total - remove["duration_seconds"] + add["duration_seconds"] - target)
                if gap >= abs(total - target) - 1e-9:
                    continue
                key = (gap, ranks[add["clip_id"]] - ranks[remove["clip_id"]], remove["clip_id"], add["clip_id"])
                if best is None or key < best[0]:
                    best = (key, index, add)
        if best is None:
            raise ValueError("Cannot match duration budget within frozen strata")
        if audit is not None:
            audit["duration_swaps"].append({"removed": filtered[best[1]]["clip_id"], "added": best[2]["clip_id"],
                                             "gap_after_seconds": best[0][0]})
        filtered[best[1]] = best[2]
    if not target or abs(sum(r["duration_seconds"] for r in filtered) - target) > limit:
        raise ValueError("Duration budget mismatch")
    if audit is not None:
        audit["duration_difference_fraction"] = abs(sum(r["duration_seconds"] for r in filtered) - target) / target
    return {"random": sorted(random_arm, key=lambda r: r["clip_id"]),
            "filtered": sorted(filtered, key=lambda r: r["clip_id"])}


def export_training(arms: dict[str, list[dict]], data_root: Path, output_dir: Path,
                    dev_records: list[dict] | None = None) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if not dev_records or any(r["split"] != "dev" for r in dev_records):
        raise ValueError("A separate dev partition is required")
    for rows in arms.values():
        if not rows or any(r["split"] != "train" or r["origin"] != "real" or
                           (r.get("audit_decision") != "pass" and not
                            (r.get("audit_decision") == "needs_review" and r.get("audit_resolution") == "human_approved")) for r in rows):
            raise ValueError("Only audited real train records may be exported")
    validate_partitions({"train": [r for rows in arms.values() for r in rows], "dev": dev_records})
    roles = {**{name: "train" for name in arms}, "dev": "dev"}
    source = {**arms, "dev": dev_records}
    for rows in source.values():
        for row in rows:
            if file_hash(audio_path(row, data_root)) != row["audio_sha256"]:
                raise ValueError("Export audio hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".export-") as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        for name, rows in source.items():
            if not name.replace("_", "").isalnum():
                raise ValueError("Unsafe arm name")
            payload = []
            for row in rows:
                language = {"en": "English", "zh": "Chinese", "mixed": "None"}[row["language_group"]]
                payload.append({"audio": str(audio_path(row, data_root)),
                                "text": f"language {language}<asr_text>{row['reference']}"})
            (stage / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload), encoding="utf-8")
        metadata = {"schema_version": 1, "roles": roles, "records": source,
                    "data_root": str(Path(data_root).resolve()),
                    "files": {f"{name}.jsonl": file_hash(stage / f"{name}.jsonl") for name in source}}
        metadata["export_hash"] = content_hash(metadata)
        write_json(stage / "EXPORT.json", metadata)
        stage.rename(output)
    return metadata


def verify_export(export_dir: Path) -> dict:
    root = Path(export_dir)
    metadata = read_json(root / "EXPORT.json")
    if metadata.get("export_hash") != content_hash({k: v for k, v in metadata.items() if k != "export_hash"}):
        raise ValueError("Export metadata hash mismatch")
    for name, sha in metadata["files"].items():
        if Path(name).name != name or file_hash(root / name) != sha:
            raise ValueError("Export file hash mismatch")
    for rows in metadata["records"].values():
        for row in rows:
            if file_hash(audio_path(row, Path(metadata["data_root"]))) != row["audio_sha256"]:
                raise ValueError("Export audio hash mismatch")
    return metadata
