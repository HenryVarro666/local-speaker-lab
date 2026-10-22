"""Independently audit frozen input provenance without loading any ASR model."""

from collections import Counter
import argparse
import hashlib
import io
import json
from pathlib import Path

from speaker_lab.duplex.protocol import file_hash, read_json, write_json
from speaker_lab.duplex.datasets import validate_partitions


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--shards", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data/duplexlab/normalization-v1"
    runs = root / "runs/duplexlab/normalization-v1"
    policy = read_json(data / "POLICY.json")
    parts = {
        r: [json.loads(s) for s in (data / f"{r}.jsonl").read_text().splitlines()]
        for r in ["train", "dev", "test"]
    }
    for role, rows in parts.items():
        assert len(rows) == policy["counts"][role]
        assert file_hash(data / f"{role}.jsonl") == policy["manifest_sha256"][role]
    validate_partitions(parts)
    required = {r["clip_id"]: r for rows in parts.values() for r in rows}
    assert len(required) == 1429
    seen = {}
    hashes = {}
    for name in [f"train-{i:05d}-of-00003.parquet" for i in range(3)]:
        path = args.shards / name
        hashes[name] = file_hash(path)
        for source in pq.read_table(path).to_pylist():
            cid = "ascend-train-" + source["id"]
            if cid not in required:
                continue
            assert cid not in seen
            row = required[cid]
            raw = source["audio"]["bytes"]
            assert row["reference"] == source["transcription"]
            assert row["language_group"] == source["language"]
            assert row["original_speaker_id"] == source["original_speaker_id"]
            assert (
                row["session_id"] == source["session_id"]
                and row["topic"] == source["topic"]
            )
            assert row["original_filename"] == Path(source["path"]).name
            assert (
                file_hash(data / row["audio_relpath"])
                == row["audio_sha256"]
                == hashlib.sha256(raw).hexdigest()
            )
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            assert sr == row["sample_rate"] == 16000 and len(audio) == row["frames"]
            assert (
                hashlib.sha256(
                    np.ascontiguousarray(audio.mean(axis=1), dtype="<f4").tobytes()
                ).hexdigest()
                == row["pcm_sha256"]
            )
            if row["split"] == "test":
                assert row["source_shard"] == name
            seen[cid] = name
    assert set(seen) == set(required)
    exclusion = read_json(runs / "EXCLUSION_BINDING.json")
    for path, digest in exclusion["historical"].items():
        assert file_hash(root / path) == digest
    for key, field in [
        ("clip_id", "prior_ids"),
        ("audio_sha256", "prior_files"),
        ("pcm_sha256", "prior_pcm"),
    ]:
        assert not {r[key] for r in parts["test"]} & set(exclusion[field])
        assert len({r[key] for r in parts["test"]}) == 362
    protected = []
    val_ids = set()
    path = args.shards / "validation-00000-of-00001.parquet"
    hashes[path.name] = file_hash(path)
    for source in pq.read_table(path).to_pylist():
        val_ids.add(source["id"])
        raw = source["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        assert sr == 16000
        protected.append(
            dict(
                audio_sha256=hashlib.sha256(raw).hexdigest(),
                pcm_sha256=hashlib.sha256(
                    np.ascontiguousarray(audio.mean(axis=1), dtype="<f4").tobytes()
                ).hexdigest(),
            )
        )
    assert protected == exclusion["protected_validation"] and len(protected) == 1130
    selected_sets = [
        set(
            read_json(
                root / "experiments/glossary/protocols" / name / "selected_ids.json"
            )
        )
        for name in [
            "ascend-val200-train-vocab30-v2",
            "ascend-val-remaining725-train-vocab30-v1",
        ]
    ]
    assert len(selected_sets[0]) == 200 and len(selected_sets[1]) == 725
    assert not selected_sets[0] & selected_sets[1]
    assert selected_sets[0] | selected_sets[1] <= val_ids

    for key in ["audio_sha256", "pcm_sha256"]:
        assert not {r[key] for r in protected} & {r[key] for r in parts["test"]}
    quotas = {int(k): v for k, v in policy["test_quotas_per_speaker"].items()}
    counts = Counter(
        (r["original_speaker_id"], r["language_group"]) for r in parts["test"]
    )
    assert counts == Counter(
        {(s, language): n for s, q in quotas.items() for language, n in q.items()}
    )
    # Independent check of primary source labels, hashes and frozen quotas. Selection
    # hash ordering is recorded in preparation; no outcome-dependent change is made.
    for name, digest in policy["shard_sha256"].items():
        assert hashes[name] == digest
    result = dict(
        status="verified",
        all_source_labels_metadata_file_pcm_verified=1429,
        new_test=362,
        test_speakers=len(quotas),
        test_languages=dict(Counter(r["language_group"] for r in parts["test"])),
        validation_excluded=1130,
        protected200_and725_verified_subset=True,
        prior_id_file_pcm_overlap=0,
        all_test_ids_file_pcm_unique=True,
        manifest_sha256=policy["manifest_sha256"],
        source_parquet_sha256=hashes,
        protected_inference_started=False,
        model_loaded=False,
        preparer_sha256=file_hash(root / "scripts/prepare_duplexlab_normalization.py"),
        auditor_sha256=file_hash(Path(__file__)),
    )
    write_json(runs / "INPUT_VERIFICATION.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
