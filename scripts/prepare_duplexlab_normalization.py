"""Freeze the new split and one coverage-constrained selector before new ASR."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile

from speaker_lab.duplex.audio import audit_clip, find_exact_duplicates
from speaker_lab.duplex.datasets import (
    sample_source_records,
    validate_partitions,
)
from speaker_lab.duplex.protocol import file_hash, read_json, write_json


def load(path):
    return [
        json.loads(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / "data/duplexlab/normalization-v1"
    out.mkdir(parents=True, exist_ok=False)
    (out / "original").mkdir()
    runs = root / "runs/duplexlab/normalization-v1"
    runs.mkdir(parents=True, exist_ok=True)
    from speaker_lab.duplex.normalization import uniform_sidecar

    old = root / "data/duplexlab/deletion-v1"
    old_policy = read_json(old / "POLICY.json")
    for role, digest in old_policy["manifest_sha256"].items():
        assert file_hash(old / f"{role}.jsonl") == digest
    train, dev = load(old / "train.jsonl"), load(old / "dev.jsonl")
    for row in train + dev:
        assert file_hash(old / row["audio_relpath"]) == row["audio_sha256"]
        (out / row["audio_relpath"]).write_bytes(
            (old / row["audio_relpath"]).read_bytes()
        )
    selection = read_json(root / "runs/duplexlab/deletion-v1/SELECTION.json")
    assert (
        file_hash(root / "runs/duplexlab/deletion-v1/SELECTION.json")
        == old_policy["selection_sha256"]
    )
    arms = selection["arms"]
    arms["uniform"] = arms["selected"]
    write_json(runs / "SELECTION.json", selection)
    audit = selection["audit"]
    weights = read_json(old / "LOSS_WEIGHTS.json")
    assert file_hash(old / "LOSS_WEIGHTS.json") == old_policy["loss_weights_sha256"]
    write_json(out / "LOSS_WEIGHTS.json", weights)
    write_json(out / "LOSS_UNIFORM.json", uniform_sidecar(weights))
    prior = []
    historical = {}
    for path in sorted((root / "data/duplexlab").rglob("*.jsonl")):
        if out in path.parents:
            continue
        rows = load(path)
        rows = [r for r in rows if "clip_id" in r and "audio_sha256" in r]
        if rows:
            prior += rows
            historical[str(path.relative_to(root))] = file_hash(path)
    old20 = load(root / "data/duplexlab/dev20-ascend-train-v1/manifest.jsonl")
    known_ids = {r["clip_id"] for r in prior + old20}
    known_audio = {r["audio_sha256"] for r in prior + old20}
    known_pcm = {r["pcm_sha256"] for r in prior if r.get("pcm_sha256")}
    for row in old20:
        a, sr = sf.read(
            root / "data/duplexlab/dev20-ascend-train-v1" / row["audio_relpath"],
            dtype="float32",
            always_2d=True,
        )
        assert sr == 16000
        known_pcm.add(
            hashlib.sha256(
                np.ascontiguousarray(a.mean(axis=1), dtype="<f4").tobytes()
            ).hexdigest()
        )
    protected = []
    for source in pq.read_table(
        args.shards / "validation-00000-of-00001.parquet"
    ).to_pylist():
        raw = source["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        assert sr == 16000
        pcm = hashlib.sha256(
            np.ascontiguousarray(audio.mean(axis=1), dtype="<f4").tobytes()
        ).hexdigest()
        digest = hashlib.sha256(raw).hexdigest()
        known_audio.add(digest)
        known_pcm.add(pcm)
        protected.append(dict(audio_sha256=digest, pcm_sha256=pcm))
    assert len(protected) == 1130
    write_json(
        runs / "EXCLUSION_BINDING.json",
        dict(
            historical=historical,
            protected_validation=protected,
            prior_ids=sorted(known_ids),
            prior_files=sorted(known_audio),
            prior_pcm=sorted(known_pcm),
        ),
    )
    eligible, waves, audits, shards = [], {}, [], {}
    with tempfile.TemporaryDirectory(dir=out, prefix="audit-candidates-") as temporary:
        temp = Path(temporary)
        for name in ["train-00001-of-00003.parquet", "train-00002-of-00003.parquet"]:
            path = args.shards / name
            shards[name] = file_hash(path)
            for source in pq.read_table(path).to_pylist():
                if (
                    source["original_speaker_id"]
                    not in [1, 2, 4, 10, 13, 14, 16, 18, 24, 26]
                    or not source["transcription"].strip()
                ):
                    continue
                cid = "ascend-train-" + source["id"]
                raw = source["audio"]["bytes"]
                info = sf.info(io.BytesIO(raw))
                if cid in known_ids or not 1 <= info.duration <= 12:
                    continue
                row = {
                    "schema_version": 1,
                    "clip_id": cid,
                    "audio_relpath": f"original/{cid}.wav",
                    "audio_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_id": "CAiRE/ASCEND@737e9800ae31be9932ba8464c80366559bd28424",
                    "source_group_id": f"ascend-speaker-{source['original_speaker_id']}",
                    "original_speaker_id": source["original_speaker_id"],
                    "session_id": source["session_id"],
                    "topic": source["topic"],
                    "original_filename": Path(source["path"]).name,
                    "source_shard": name,
                    "split": "test",
                    "original_split": "train",
                    "language_group": source["language"],
                    "reference": source["transcription"],
                    "origin": "real",
                    "license_id": "CC-BY-SA-4.0",
                    "label_status": "source_reference",
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "duration_seconds": info.duration,
                    "split_unit": "original_speaker_id",
                }
                f = temp / row["audio_relpath"]
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(raw)
                a = audit_clip(row, temp, out / "candidate-derived", {})
                audits.append(a)
                if (
                    a["decision"] == "pass"
                    and row["audio_sha256"] not in known_audio
                    and a["pcm_sha256"] not in known_pcm
                ):
                    row.update(
                        audit_decision="pass",
                        pcm_sha256=a["pcm_sha256"],
                        silence_fraction=a["silence_fraction"],
                    )
                    eligible.append(row)
                    waves[cid] = raw
                f.unlink()
    counts = Counter((r["original_speaker_id"], r["language_group"]) for r in eligible)
    quotas_test = {
        s: {
            "mixed": 3 if s == 24 else 10,
            "zh": 10 if s == 13 else 20,
            "en": 3 if s in [4, 10, 26] else 10,
        }
        for s in [1, 2, 4, 10, 13, 14, 16, 18, 24, 26]
    }
    speakers = [1, 2, 4, 10, 13, 14, 16, 18, 24, 26]
    if any(
        counts[(s, lang)] < n for s in speakers for lang, n in quotas_test[s].items()
    ):
        raise ValueError(f"Frozen new-test quotas infeasible: {counts}")
    test = [
        r
        for s in speakers
        for r in sample_source_records(
            [r for r in eligible if r["original_speaker_id"] == s],
            quotas_test[s],
            20261024,
        )
    ]
    validate_partitions({"train": train, "dev": dev, "test": test})
    selected_ids = {r["clip_id"] for r in test}
    if find_exact_duplicates([a for a in audits if a["clip_id"] in selected_ids]):
        raise ValueError("Exact duplicates within new test")
    for row in test:
        (out / row["audio_relpath"]).write_bytes(waves[row["clip_id"]])
    for role, rows in [("train", train), ("dev", dev), ("test", test)]:
        (out / f"{role}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    write_json(out / "AUDIT-test-candidates.json", audits)
    write_json(
        out / "POLICY.json",
        {
            "schema_version": 1,
            "selection_seed": 20261021,
            "test_selection_seed": 20261024,
            "train_speakers": [8, 9, 11, 15, 19, 21],
            "dev_speakers": [5, 6],
            "test_speakers": [1, 2, 4, 10, 13, 14, 16, 18, 24, 26],
            "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
            "test_quotas_per_speaker": quotas_test,
            "test_audit_decision": "pass only",
            "eligible_test_counts": {
                f"{s}|{lang}": n for (s, lang), n in sorted(counts.items())
            },
            "shard_sha256": shards,
            "manifest_sha256": {
                r: file_hash(out / f"{r}.jsonl") for r in ["train", "dev", "test"]
            },
            "selection_sha256": file_hash(runs / "SELECTION.json"),
            "loss_weights_sha256": file_hash(out / "LOSS_WEIGHTS.json"),
            "protocol_sha256": file_hash(
                root / "experiments/duplexlab/NORMALIZATION_PROTOCOL.md"
            ),
            "uniform_weights_sha256": file_hash(out / "LOSS_UNIFORM.json"),
            "historical_manifest_sha256": historical,
            "previous_policy_sha256": file_hash(old / "POLICY.json"),
            "test_independence": "New clips, current-experiment speaker-disjoint; speaker IDs have prior confirmation/test exposure; all twelve SFTs restart from base.",
            "legacy_text_exposure": "Historical glossary extraction read all official train texts; no legacy glossary or trained checkpoint used here.",
            "prior_clip_id_file_pcm_overlap": 0,
        },
    )
    print(
        json.dumps(
            {"policy": read_json(out / "POLICY.json"), "selection": audit},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
