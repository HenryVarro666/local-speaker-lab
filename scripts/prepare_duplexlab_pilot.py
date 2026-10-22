"""Prepare the preregistered, speaker-disjoint ASCEND continuation (no ASR)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from speaker_lab.duplex.audio import audit_clip, find_exact_duplicates
from speaker_lab.duplex.datasets import sample_source_records, split_by_speaker, stratum_quotas, validate_partitions
from speaker_lab.duplex.protocol import file_hash, freeze_protocol, write_json

DATASET_REVISION = "737e9800ae31be9932ba8464c80366559bd28424"
SPEAKERS = {"excluded": [1, 2], "language_confirmation": [13, 14], "dev": [5, 6], "test": [18, 24]}


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ["parquet", "output", "old-manifest", "asr-config"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    import pyarrow.parquet as pq
    import soundfile as sf
    import numpy as np

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "original").mkdir()
    rows = pq.read_table(args.parquet).to_pylist()
    metadata, waves = [], {}
    original_speakers = {"ascend-train-" + r["id"]: r["original_speaker_id"] for r in rows}
    old_rows = [json.loads(line) for line in args.old_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {original_speakers[r["clip_id"]] for r in old_rows} == {1, 2}
    old_audio = set(r["audio_sha256"] for r in old_rows)
    old_pcm = set()
    for row in old_rows:
        samples, _ = sf.read(args.old_manifest.parent / row["audio_relpath"], dtype="float32", always_2d=True)
        old_pcm.add(hashlib.sha256(np.ascontiguousarray(samples.mean(axis=1), dtype="<f4").tobytes()).hexdigest())
    for source in rows:
        if source["original_speaker_id"] in SPEAKERS["excluded"]:
            continue
        raw = source["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))
        if not 1 <= info.duration <= 12 or not source["transcription"].strip():
            continue
        cid = "ascend-train-" + source["id"]
        row = {"schema_version": 1, "clip_id": cid, "audio_relpath": f"original/{cid}.wav",
               "audio_sha256": hashlib.sha256(raw).hexdigest(),
               "source_id": f"CAiRE/ASCEND@{DATASET_REVISION}",
               "source_group_id": f"ascend-speaker-{source['original_speaker_id']}",
               "original_speaker_id": source["original_speaker_id"], "session_id": source["session_id"],
               "topic": source["topic"], "original_filename": Path(source["path"]).name,
               "split": "train", "original_split": "train", "language_group": source["language"],
               "reference": source["transcription"], "origin": "real", "license_id": "CC-BY-SA-4.0",
               "label_status": "source_reference", "sample_rate": info.samplerate, "channels": info.channels,
               "frames": info.frames, "duration_seconds": info.duration, "split_unit": "original_speaker_id"}
        metadata.append(row)
        waves[cid] = raw
    parts = split_by_speaker(metadata, SPEAKERS)
    candidates = {"language_confirmation": sample_source_records(parts["language_confirmation"], {"en": 40}, 20261020, 20),
                  "dev": sample_source_records(parts["dev"], {"mixed": 20, "zh": 10, "en": 10}, 20261020),
                  "test": sample_source_records(parts["test"], {"mixed": 50, "zh": 25, "en": 25}, 20261020),
                  "train": sorted(parts["train"], key=lambda r: hashlib.sha256(f"20261020:{r['clip_id']}".encode()).hexdigest())[:2000]}
    audits = []
    for role, records in candidates.items():
        for row in records:
            row["split"] = "dev" if role == "dev" else "test" if role in {"test", "language_confirmation"} else "train"
            (out / row["audio_relpath"]).write_bytes(waves[row["clip_id"]])
            audit = audit_clip(row, out, out / "derived", {})
            row.update(audit_decision=audit["decision"], silence_fraction=audit.get("silence_fraction"), pcm_sha256=audit.get("pcm_sha256"))
            if row["audio_sha256"] in old_audio or row["pcm_sha256"] in old_pcm:
                raise ValueError("Old development audio overlaps new samples")
            if role != "train" and audit["decision"] == "reject":
                raise ValueError(f"Frozen evaluation source fails audit: {row['clip_id']}")
            audits.append(audit)
    validate_partitions(candidates)
    duplicates = find_exact_duplicates(audits)
    repeated = {r["clip_id"] for r in duplicates}
    candidates["train"] = [r for r in candidates["train"] if r["audit_decision"] == "pass" and r["clip_id"] not in repeated]
    quotas = stratum_quotas(candidates["train"], {"mixed": 250, "zh": 125, "en": 125})
    for role, records in candidates.items():
        (out / f"{role}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(records, key=lambda r: r["clip_id"])), encoding="utf-8")
    config = args.asr_config.read_text(encoding="utf-8")
    for suffix, language in [("auto", ""), ("english", "English")]:
        text = config.replace('condition = "qwen06_cpu_auto"', f'condition = "confirm40_cpu_{suffix}"').replace('language = ""', f'language = "{language}"')
        config_path = out / f"confirm40-{suffix}.toml"
        config_path.write_text(text, encoding="utf-8")
        freeze_protocol(out / "language_confirmation.jsonl", out, config_path, out / "protocols" / f"confirm40-{suffix}")
    policy = {"schema_version": 1, "seed": 20261020, "quotas": quotas, "max_duration_difference": .01,
              "speakers": SPEAKERS, "dataset_revision": DATASET_REVISION, "parquet_sha256": file_hash(args.parquet),
              "counts": {k: len(v) for k, v in candidates.items()},
              "manifest_sha256": {k: file_hash(out / f"{k}.jsonl") for k in candidates},
              "old_metadata_correction": {r["clip_id"]: {"original_speaker_id": original_speakers[r["clip_id"]]} for r in old_rows},
              "training_reference_status": "official source_reference; no project human approval"}
    write_json(out / "POLICY.json", policy)
    write_json(out / "AUDIT.json", audits)
    write_json(out / "DUPLICATES.json", duplicates)
    print(json.dumps({"counts": policy["counts"], "quotas": quotas, "duplicates": len(duplicates)}))


if __name__ == "__main__":
    main()
