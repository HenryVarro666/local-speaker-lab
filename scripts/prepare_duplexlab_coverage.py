"""Freeze the new split and one coverage-constrained selector before new ASR."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile

from speaker_lab.duplex.audio import audit_clip, find_exact_duplicates
from speaker_lab.duplex.coverage import coverage_select, feature_marginals
from speaker_lab.duplex.datasets import (
    sample_source_records,
    stratum,
    stratum_quotas,
    validate_partitions,
)
from speaker_lab.duplex.protocol import file_hash, read_json, write_json
from speaker_lab.glossary_metrics import score_pair


def load(path):
    return [
        json.loads(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import scipy
    import soundfile as sf

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / "data/duplexlab/coverage-v1"
    out.mkdir(parents=True, exist_ok=False)
    (out / "original").mkdir()
    runs = root / "runs/duplexlab/coverage-v1"
    runs.mkdir(parents=True, exist_ok=True)
    previous = root / "data/duplexlab/continuation-v1"
    train = [
        r
        for r in load(previous / "train.jsonl")
        if r["original_speaker_id"] not in [10, 16, 26]
    ]
    dev = load(previous / "dev.jsonl")
    assert len(train) == 1015 and len(dev) == 40
    for row in train + dev:
        (out / row["audio_relpath"]).write_bytes(
            (previous / row["audio_relpath"]).read_bytes()
        )
    preds_path = (
        root / "runs/duplexlab/h100-continuation-v1/pool-base/predictions.jsonl"
    )
    predictions = {
        r["clip_id"]: r
        for r in load(preds_path)
        if r["clip_id"] in {v["clip_id"] for v in train}
    }
    for r in train:
        pred = predictions[r["clip_id"]]
        if pred["audio_sha256"] != r["audio_sha256"]:
            raise ValueError("Reused train prediction has wrong audio")
        c = score_pair(r["reference"], pred["hypothesis"], terms=[])
        if c["N"] <= 0:
            raise ValueError("No training reference tokens")
        r["selection_mer"] = (c["S"] + c["D"] + c["I"]) / c["N"]
    quotas = stratum_quotas(train, {"mixed": 250, "zh": 125, "en": 125})
    rng = random.Random(20261021)
    random_rows = []
    for key, n in sorted(quotas.items()):
        random_rows += rng.sample(
            sorted([r for r in train if stratum(r) == key], key=lambda r: r["clip_id"]),
            n,
        )
    filtered, audit = coverage_select(train, random_rows)
    arms = {
        "random": sorted(random_rows, key=lambda r: r["clip_id"]),
        "filtered": filtered,
    }
    write_json(
        runs / "SELECTION.json",
        {
            "arms": arms,
            "audit": audit,
            "quotas": quotas,
            "scipy": scipy.__version__,
            "pool_marginals": feature_marginals(train),
            "source_predictions_sha256": file_hash(preds_path),
            "duration_seconds": {
                a: sum(r["duration_seconds"] for r in rows) for a, rows in arms.items()
            },
            "strategy": "matched-marginal coverage-constrained reference agreement; no test feedback",
        },
    )

    prior = [
        r
        for role in ["train", "dev", "test", "language_confirmation"]
        for r in load(previous / f"{role}.jsonl")
    ]
    prior += load(root / "data/duplexlab/multiseed-v1/primary.jsonl")
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
    eligible, waves, audits, shards = [], {}, [], {}
    with tempfile.TemporaryDirectory(dir=out, prefix="audit-candidates-") as temporary:
        temp = Path(temporary)
        for name in ["train-00001-of-00003.parquet", "train-00002-of-00003.parquet"]:
            path = args.shards / name
            shards[name] = file_hash(path)
            for source in pq.read_table(path).to_pylist():
                if (
                    source["original_speaker_id"] not in [10, 16, 26]
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
    en = 15 if min(counts[(s, "en")] for s in [10, 16, 26]) >= 15 else 10
    quotas_test = {"mixed": 50, "zh": 50 - en, "en": en}
    test = [
        r
        for s in [10, 16, 26]
        for r in sample_source_records(
            [r for r in eligible if r["original_speaker_id"] == s],
            quotas_test,
            20261021,
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
            "train_speakers": [8, 9, 11, 15, 19, 21],
            "dev_speakers": [5, 6],
            "test_speakers": [10, 16, 26],
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
            "protocol_sha256": file_hash(
                root / "experiments/duplexlab/COVERAGE_PROTOCOL.md"
            ),
            "test_independence": "New clips, current-experiment speaker-disjoint; speakers occurred in previous train pool; all six SFTs restart from base.",
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
