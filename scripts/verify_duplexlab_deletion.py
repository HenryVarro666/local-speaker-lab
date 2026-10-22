"""Recompute coverage-study results from frozen audio, SQLite and checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

from speaker_lab.duplex.coverage import has_filler
from speaker_lab.duplex.deletion import feature_marginals, retention_spans
from speaker_lab.duplex.disfluency import alignment_edits
from speaker_lab.duplex.datasets import validate_partitions
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds
from speaker_lab.duplex.protocol import (
    content_hash,
    file_hash,
    load_manifest,
    read_json,
    write_json,
)
from speaker_lab.glossary_metrics import score_pair

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/duplexlab/deletion-v1"
DATA = ROOT / "data/duplexlab/deletion-v1"


def load(path):
    return [
        json.loads(s)
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def main():
    import hashlib
    import numpy as np
    import soundfile as sf

    policy = read_json(DATA / "POLICY.json")
    parts = {}
    for role, digest in policy["manifest_sha256"].items():
        assert file_hash(DATA / f"{role}.jsonl") == digest
        parts[role] = load_manifest(DATA / f"{role}.jsonl", DATA)
        for row in parts[role]:
            assert file_hash(DATA / row["audio_relpath"]) == row["audio_sha256"]
    validate_partitions(parts)
    assert (
        file_hash(ROOT / "experiments/duplexlab/DELETION_PROTOCOL.md")
        == policy["protocol_sha256"]
    )
    assert file_hash(RUNS / "SELECTION.json") == policy["selection_sha256"]
    selection = read_json(RUNS / "SELECTION.json")
    for name, digest in read_json(RUNS / "PREPARATION_SOURCE.json").items():
        assert file_hash(RUNS / "preparation-source" / Path(name).name) == digest
    for name, digest in read_json(RUNS / "SOURCE.json").items():
        assert file_hash(RUNS / "source" / Path(name).name) == digest
    arms = selection["arms"]
    previous = read_json(ROOT / "runs/duplexlab/coverage-v1/SELECTION.json")
    assert [r["clip_id"] for r in arms["random"]] == [
        r["clip_id"] for r in previous["arms"]["random"]
    ]
    assert len(parts["test"]) == 280
    assert len(parts["dev"]) == 52
    assert {r["original_speaker_id"] for r in parts["test"]} == {
        10,
        13,
        14,
        16,
        18,
        24,
        26,
    }
    assert all(len(r) == 500 for r in arms.values())
    assert (
        feature_marginals(arms["random"])
        == feature_marginals(arms["filtered"])
        == selection["audit"]["marginals"]
    )
    assert (
        abs(
            sum(r["duration_seconds"] for r in arms["random"])
            - sum(r["duration_seconds"] for r in arms["filtered"])
        )
        / sum(r["duration_seconds"] for r in arms["random"])
        <= 0.01
    )
    original_predictions = (
        ROOT / "runs/duplexlab/h100-continuation-v1/pool-base/predictions.jsonl"
    )
    assert file_hash(original_predictions) == selection["source_predictions_sha256"]
    preds = {
        r["clip_id"]: r
        for r in load(original_predictions)
        if r["clip_id"] in {v["clip_id"] for v in parts["train"]}
    }
    for r in parts["train"]:
        p = preds[r["clip_id"]]
        assert p["audio_sha256"] == r["audio_sha256"]
        c = score_pair(r["reference"], p["hypothesis"], terms=[])
        edits = alignment_edits(r["reference"], p["hypothesis"])
        assert r["base_deletion_count"] == c["D"]
        assert r["base_tail_deletion_count"] == sum(
            x["index"] >= 0.75 * c["N"] for x in edits["deleted"]
        )
        assert r["selection_cost"] == (c["S"] + c["I"] + 0.5 * c["D"]) / c["N"]
        assert r["selection_mer"] == (c["S"] + c["D"] + c["I"]) / c["N"]
    assert arms["selected"] == arms["filtered"]
    sidecar = read_json(DATA / "LOSS_WEIGHTS.json")
    assert file_hash(DATA / "LOSS_WEIGHTS.json") == policy["loss_weights_sha256"]
    entries = []
    for r in arms["filtered"]:
        lang = {"en": "English", "zh": "Chinese", "mixed": "None"}[r["language_group"]]
        prefix = f"language {lang}<asr_text>"
        spans = retention_spans(r["reference"], preds[r["clip_id"]]["hypothesis"])
        for span in spans:
            span["start"] += len(prefix)
            span["end"] += len(prefix)
        entries.append(
            dict(
                audio=Path(r["audio_relpath"]).name,
                target=prefix + r["reference"],
                spans=spans,
            )
        )
    assert entries == sidecar["entries"]
    prior = [
        r
        for role in ["train", "dev", "test", "language_confirmation"]
        for r in load(ROOT / f"data/duplexlab/continuation-v1/{role}.jsonl")
    ]
    prior += load(ROOT / "data/duplexlab/multiseed-v1/primary.jsonl")
    prior += load(ROOT / "data/duplexlab/coverage-v1/test.jsonl")
    prior += load(ROOT / "data/duplexlab/disfluency-v1/test.jsonl")
    prior += load(ROOT / "data/duplexlab/disfluency-v1/auxdev.jsonl")
    old20 = load(ROOT / "data/duplexlab/dev20-ascend-train-v1/manifest.jsonl")
    for row in old20:
        audio, sr = sf.read(
            ROOT / "data/duplexlab/dev20-ascend-train-v1" / row["audio_relpath"],
            dtype="float32",
            always_2d=True,
        )
        assert sr == 16000
        row["pcm_sha256"] = hashlib.sha256(
            np.ascontiguousarray(audio.mean(axis=1), dtype="<f4").tobytes()
        ).hexdigest()
    prior += old20
    for key in ["clip_id", "audio_sha256", "pcm_sha256"]:
        assert not {r.get(key) for r in prior if r.get(key)} & {
            r[key] for r in parts["test"]
        }
    assert all(r["audit_decision"] == "pass" for r in parts["test"])
    export = read_json(RUNS / "export/EXPORT.json")
    assert (
        content_hash({k: v for k, v in export.items() if k != "export_hash"})
        == export["export_hash"]
    )
    assert export["roles"] == {
        "random": "train",
        "selected": "train",
        "filtered": "train",
        "dev": "dev",
    }
    for name, digest in export["files"].items():
        assert file_hash(RUNS / "export" / name) == digest
    for arm in ["random", "selected", "filtered"]:
        assert export["records"][arm] == arms[arm]
        lines = load(RUNS / f"export/{arm}.jsonl")
        for row, line in zip(arms[arm], lines):
            language = {"en": "English", "zh": "Chinese", "mixed": "None"}[
                row["language_group"]
            ]
            assert line["text"] == f"language {language}<asr_text>{row['reference']}"
            assert line["audio"] == export["data_root"] + "/" + row["audio_relpath"]
    computed = {}
    asr_sources = []
    for role in ["dev", "test"]:
        for suffix in ["base"] + [
            f"{a}-{s}" for s in [42, 43, 44] for a in ["random", "selected", "filtered"]
        ]:
            name = f"{role}-{suffix}"
            summary = score_run(RUNS / name, RUNS / "protocols" / name, DATA)
            assert summary == read_json(RUNS / name / "summary.json"), name
            computed[name] = summary
            asr_sources.append(read_json(RUNS / name / "run.json")["asr_source_hashes"])
        for contrast, (left, right) in {
            "joint": ("random", "filtered"),
            "selection": ("random", "selected"),
            "objective": ("selected", "filtered"),
        }.items():
            paired = {
                s: {
                    "random": computed[f"{role}-{left}-{s}"],
                    "filtered": computed[f"{role}-{right}-{s}"],
                }
                for s in [42, 43, 44]
            }
            assert compare_seeds(paired, parts[role]) == read_json(
                RUNS / f"COMPARISON-{role}-{contrast}.json"
            )
    assert all(value == asr_sources[0] for value in asr_sources)
    marker = read_json(RUNS / "DEV_COMPLETE.json")
    assert marker["complete_before_test"] and not marker["strategy_changed"]
    for name, digest in marker["dev_summary_sha256"].items():
        assert file_hash(RUNS / f"dev-{name}/summary.json") == digest
    actual = read_json(RUNS / "ACTUAL_ARGUMENTS.json")
    template = {k: v for k, v in actual["random-42"].items() if k != "seed"}
    assert all(
        {k: v for k, v in r.items() if k != "seed"} == template for r in actual.values()
    )
    assert template["per_device_eval_batch_size"] == 1
    weights = []
    for seed in [42, 43, 44]:
        for arm in ["random", "selected", "filtered"]:
            record = read_json(RUNS / f"train-{arm}-{seed}-record/training.json")
            assert (
                record["status"] == "trained_and_reloaded"
                and record["test_complete"]
                and record["checkpoint_reloaded"]
            )
            assert (
                record["optimizer_steps"] == 63
                and actual[f"{arm}-{seed}"]["seed"] == seed
            )
            assert record["configuration"]["seed"] == seed
            assert record["export_hash"] == export["export_hash"]
            base_state = read_json(RUNS / "test-base/run.json")
            assert record["configuration"]["model_path"] == base_state["model"]
            remote_run = Path(record["configuration"]["record_dir"]).parent
            command = record["command"]
            assert command[command.index("--train_file") + 1] == str(
                remote_run / f"export/{arm}.jsonl"
            )
            assert command[command.index("--eval_file") + 1] == str(
                remote_run / "export/dev.jsonl"
            )
            path = RUNS / f"training-state/{arm}-{seed}.json"
            assert file_hash(path) == record["checkpoint_hashes"]["trainer_state.json"]
            state = read_json(path)
            assert state["global_step"] == 63 and state["epoch"] == 1
            for key in ["loss", "grad_norm", "eval_loss"]:
                values = [r[key] for r in state["log_history"] if key in r]
                assert values and all(math.isfinite(v) for v in values)
            weight = record["checkpoint_hashes"]["model.safetensors"]
            for role in ["dev", "test"]:
                state = read_json(RUNS / f"{role}-{arm}-{seed}/run.json")
                assert state["model_artifacts"]["model.safetensors"] == weight
                assert (
                    state["config"]["asr"]["language"] == ""
                    and state["config"]["asr"]["aligner"] == ""
                )
            if arm == "filtered":
                runtime = read_json(RUNS / f"LOSS_RUNTIME-{seed}.json")
                assert (
                    runtime["train_calls"] == 500 and runtime["boosted_subwords"] == 369
                )
                assert (
                    runtime["dev_calls"] == 52 and runtime["exact_suffix_checks"] == 552
                )
                assert runtime["sidecar_sha256"] == policy["loss_weights_sha256"]
                assert record["loss_runtime"] == runtime
                assert "speaker_lab.duplex.retention_sft" in command
                assert record["configuration"]["objective"] == "retention-v1"
                assert (
                    runtime["adapter_sha256"]
                    == read_json(RUNS / "SOURCE.json")[
                        "src/speaker_lab/duplex/retention_sft.py"
                    ]
                )
            else:
                assert "objective" not in record["configuration"]
                assert "speaker_lab.duplex.study_sft" in command
            weights.append(weight)
    assert len(set(weights)) == 9
    assert (
        read_json(RUNS / "test-base/run.json")["model_artifacts"]["model.safetensors"]
        not in weights
    )
    filler = read_json(RUNS / "FILLER_GROUPS.json")
    metadata = {r["clip_id"]: r for r in parts["test"]}
    for seed in [42, 43, 44]:
        for arm in ["random", "selected", "filtered"]:
            for flag in [False, True]:
                rows = [
                    r
                    for r in computed[f"test-{arm}-{seed}"]["per_clip"]
                    if has_filler(metadata[r["clip_id"]]["reference"]) == flag
                ]
                assert len(rows) == filler[str(seed)][arm][str(flag)]["clips"]
                assert all(
                    sum(r[k] for r in rows) == filler[str(seed)][arm][str(flag)][k]
                    for k in ["S", "D", "I", "N"]
                )
    result = {
        "status": "verified",
        "complete_asr_runs": 20,
        "model_inferences": 3320,
        "new_test_clips": 280,
        "dev_clips_reused": 52,
        "current_split_speaker_disjoint": True,
        "new_test_prior_clip_overlap": 0,
        "coverage_marginals_exact": True,
        "nine_fresh_checkpoints_verified": True,
        "server_scores_and_bootstrap_match": True,
        "all_train_dev_test_audio_hashes_verified": True,
        "export_verified": True,
        "no_model_reinference": True,
    }
    write_json(RUNS / "LOCAL_VERIFICATION.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
