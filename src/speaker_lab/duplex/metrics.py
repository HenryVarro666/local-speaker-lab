"""Complete-case, origin-separated ASR scores using the established tokenizer."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from speaker_lab.glossary_metrics import aggregate_scores, score_pair

from .protocol import read_json, verify_protocol
from .store import RunStore


def score_run(run_dir: Path, protocol_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    state = read_json(run_dir / "run.json")
    if state.get("status") != "complete":
        raise ValueError("Only a complete run can produce a quality comparison")
    records = verify_protocol(protocol_dir, data_root)
    frozen = read_json(Path(protocol_dir) / "FREEZE.json")
    if state.get("protocol_hash") != frozen["protocol_hash"]:
        raise ValueError("Run belongs to another protocol")
    results = RunStore(run_dir / "state.sqlite").iter_results()
    by_id = {row["clip_id"]: row for row in results}
    if len(results) != len(by_id) or set(by_id) != {row["clip_id"] for row in records}:
        raise ValueError("Run is not complete: missing or duplicate final predictions")
    buckets = defaultdict(lambda: defaultdict(list))
    per_clip, excluded = [], 0
    for row in records:
        prediction = by_id[row["clip_id"]]
        if prediction.get("audio_sha256") != row["audio_sha256"]:
            raise ValueError("Prediction audio hash mismatch")
        if row["origin"] == "fixture":
            excluded += 1
            continue
        counts = score_pair(row["reference"], prediction["hypothesis"], terms=[])
        buckets[row["origin"]][row["language_group"]].append(counts)
        per_clip.append({"clip_id": row["clip_id"], "origin": row["origin"],
                         "language_group": row["language_group"], "reference": row["reference"],
                         "hypothesis": prediction["hypothesis"], **counts})
    origins = {}
    for origin, languages in buckets.items():
        origins[origin] = {
            "overall": aggregate_scores([r for rows in languages.values() for r in rows]),
            "languages": {language: aggregate_scores(rows) for language, rows in languages.items()},
        }
    return {"schema_version": 1, "status": "complete", "mode": state.get("mode", "offline"),
            "normalization": "glossary_metrics_v1", "protocol_hash": frozen["protocol_hash"],
            "excluded_fixture": excluded, "origins": origins, "per_clip": per_clip,
            "interpretation": "development data; no training effect or population-level claim"}
