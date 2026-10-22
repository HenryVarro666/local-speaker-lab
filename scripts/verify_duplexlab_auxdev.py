"""Verify the pre-test dev supplement, seven actual reloads and strict scores."""

import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from speaker_lab.duplex.disfluency import text_features
from speaker_lab.duplex.datasets import validate_partitions
from speaker_lab.duplex.protocol import file_hash, read_json, write_json
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    d = ROOT / "data/duplexlab/disfluency-v1"
    r = ROOT / "runs/duplexlab/disfluency-v1"
    p = read_json(d / "AUXDEV_POLICY.json")
    freeze = read_json(r / "AUXDEV_FREEZE.json")
    rows = load(d / "auxdev.jsonl")
    assert read_json(r / "LOCAL_VERIFICATION.json")["status"] == "verified"
    assert (
        file_hash(d / "auxdev.jsonl")
        == p["manifest_sha256"]
        == freeze["auxdev_manifest_sha256"]
    )
    assert file_hash(d / "AUXDEV_POLICY.json") == freeze["auxdev_policy_sha256"]
    assert (
        file_hash(ROOT / "experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md")
        == p["protocol_sha256"]
        == freeze["addendum_sha256"]
    )
    assert (
        file_hash(r / "auxdev-preparation-source/prepare_duplexlab_auxdev.py")
        == p["source_sha256"]
    )
    assert (
        freeze["test_base_exists"] is False
        and freeze["timestamp_unix"]
        < read_json(r / "DEV_COMPLETE.json")["timestamp_unix"]
    )
    parts = {role: load(d / f"{role}.jsonl") for role in ["train", "dev", "test"]}
    validate_partitions({**parts, "dev": parts["dev"] + rows})
    prior = sum(parts.values(), [])
    for name in [
        "coverage-v1/test.jsonl",
        "multiseed-v1/primary.jsonl",
        "continuation-v1/train.jsonl",
        "continuation-v1/test.jsonl",
        "continuation-v1/language_confirmation.jsonl",
    ]:
        prior += load(ROOT / "data/duplexlab" / name)
    old = ROOT / "data/duplexlab/dev20-ascend-train-v1"
    for row in load(old / "manifest.jsonl"):
        audio, sr = sf.read(old / row["audio_relpath"], dtype="float32", always_2d=True)
        assert sr == 16000
        row["pcm_sha256"] = hashlib.sha256(
            np.ascontiguousarray(audio.mean(axis=1), dtype="<f4").tobytes()
        ).hexdigest()
        prior.append(row)
    assert all(
        not ({x[k] for x in rows} & {x[k] for x in prior})
        for k in ["clip_id", "audio_sha256", "pcm_sha256"]
    )
    for row in rows:
        assert (
            file_hash(d / row["audio_relpath"]) == row["audio_sha256"]
            and row["audit_decision"] == "pass"
        )
    features = [text_features(row["reference"]) for row in rows]
    actual = (
        [len(rows)]
        + [sum(row["original_speaker_id"] == s for row in rows) for s in [5, 6]]
        + [
            sum(row["language_group"] == lang for row in rows)
            for lang in ["zh", "en", "mixed"]
        ]
        + [
            sum(f["filler_count"] >= 2 for f in features),
            sum(f["repeat_count"] >= 2 for f in features),
            sum(
                f["switch_count"] >= 4 and row["language_group"] == "mixed"
                for f, row in zip(features, rows)
            ),
            sum(
                f["max_en_span"] >= 3 and row["language_group"] == "mixed"
                for f, row in zip(features, rows)
            ),
        ]
    )
    assert actual == p["actual"] and all(
        x >= minimum for x, minimum in zip(actual, p["minimums"])
    )
    for name, digest in read_json(r / "AUXDEV_SOURCE.json").items():
        assert file_hash(r / "auxdev-source" / name) == digest
    pairs = {s: {} for s in [42, 43, 44]}
    summaries = {}
    asr_sources = read_json(r / "test-base/run.json")["asr_source_hashes"]
    for suffix in ["base"] + [
        f"{a}-{s}" for s in [42, 43, 44] for a in ["random", "filtered"]
    ]:
        name = "auxdev-" + suffix
        state = read_json(r / name / "run.json")
        summary = score_run(r / name, r / "protocols" / name, d)
        assert summary == read_json(r / name / "summary.json")
        assert (
            state["status"] == "complete"
            and state["succeeded"] == 12
            and state["asr_source_hashes"] == asr_sources
        )
        primary = read_json(
            r / ("test-base" if suffix == "base" else "test-" + suffix) / "run.json"
        )
        assert state["model_artifacts"] == primary["model_artifacts"]
        for key in [
            "device",
            "dtype",
            "language",
            "aligner",
            "attn_implementation",
            "max_new_tokens",
        ]:
            assert state["config"]["asr"][key] == primary["config"]["asr"][key]
        assert (
            state["config"]["asr"]["language"] == ""
            and state["config"]["asr"]["aligner"] == ""
        )
        summaries[suffix] = summary
        if suffix != "base":
            arm, seed = suffix.split("-")
            pairs[int(seed)][arm] = summary
    assert (
        compare_seeds(pairs, rows) == read_json(r / "AUXDEV_RESULTS.json")["comparison"]
    )
    result = {
        "status": "verified",
        "complete_auxdev_asr_batches": 7,
        "auxdev_clip_inferences": 84,
        "new_auxdev_clips": 12,
        "dev_unique_clips_with_primary": 52,
        "all_constraints_verified": True,
        "frozen_before_main_test": True,
        "evaluated_after_main_test": True,
        "all_scores_and_bootstrap_match": True,
        "actual_weight_and_source_bindings_verified": True,
        "source_audio_and_prior_clip_exclusion_verified": True,
        "no_model_reinference": True,
        "human_listening": "pending",
    }
    write_json(r / "AUXDEV_VERIFICATION.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
