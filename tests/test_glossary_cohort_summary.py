"""Synthetic-only cohort publication checks; no model or real ASCEND data."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_glossary_cohorts.py"
SPEC = importlib.util.spec_from_file_location("glossary_cohort_summary_tests", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seal_protocol(directory, spec):
    portable = {name: report.analysis.file_hash(directory / name)
                for name in ("correct.json", "unrelated.json", "selected_ids.json", "analysis_plan.json")}
    spec["portable_files_sha256"] = portable
    write_json(directory / "protocol.json", spec)
    write_json(directory / "FREEZE.json", {
        "sha256": {"protocol.json": report.analysis.file_hash(directory / "protocol.json"), **portable}})


@pytest.fixture
def cohorts(tmp_path):
    terms, controls = ["cache", "API"], ["cairn", "ash"]
    settings = {"model": "synthetic-model", "revision": "synthetic-revision", "device": "cpu",
                "dtype": "float32", "language": "Chinese", "max_new_tokens": 256, "torch_threads": 8}
    model = {"repo_id": "synthetic/model", **{k: v for k, v in settings.items() if k != "model"},
             "seed": 7, "do_sample": False, "num_beams": 1}
    samples = [
        [("old-positive", "mixed", "检查 cache API", "检查 cash API", "检查 cache app", "检查 cash API"),
         ("old-negative", "mandarin", "今天继续", "今天继续", "今天 cache 继续", "今天继续")],
        [("new-long", "english", "one two three four five six seven eight nine cache",
          "one two three four five six seven eight nine cash",
          "one two three four five six seven eight nine cache",
          "one two three four five six seven eight nine cash"),
         ("new-negative", "mandarin", "[UNK] 明天继续", "[UNK] 明天继续", "[UNK] 明天继续", "[UNK] 明天继续"),
         ("new-positive", "english", "API", "API", "API", "ash")],
    ]
    arguments = []
    for name, clips in zip(("old", "new"), samples):
        run = tmp_path / (name + "-run")
        protocol = tmp_path / (name + "-protocol")
        run.mkdir()
        protocol.mkdir()
        rows, predictions, audio = [], [], {}
        for identifier, group, reference, *hypotheses in clips:
            wav = run / (identifier + ".wav")
            wav.write_bytes(("Frozen synthetic audio: " + identifier).encode())
            audio[identifier] = report.analysis.file_hash(wav)
            rows.append({"id": identifier, "group": group, "split": "validation",
                         "audio": wav.name, "reference": reference})
            predictions.extend({"id": identifier, "condition": arm, "text": text,
                                "seconds": float(index), "duration_seconds": 2.0}
                               for index, (arm, text) in enumerate(zip(report.analysis.CONDITIONS, hypotheses), 1))
        manifest = run / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        (run / "predictions.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions), encoding="utf-8")
        write_json(protocol / "correct.json", terms)
        write_json(protocol / "unrelated.json", controls)
        write_json(protocol / "selected_ids.json", [row["id"] for row in rows])
        write_json(protocol / "analysis_plan.json", {
            "uncertainty": {"replicates": 7, "seed": 17, "confidence": .95}})
        metadata = {"status": "complete", "config": settings, "config_sha256": "0" * 64,
                    "manifest_sha256": report.analysis.file_hash(manifest),
                    "personal_sha256": report.analysis.file_hash(protocol / "correct.json"),
                    "unrelated_sha256": report.analysis.file_hash(protocol / "unrelated.json"),
                    "audio_sha256": audio, "contexts": report.analysis.make_contexts(terms, controls),
                    "seed": 7, "offline": True, "normalization": report.NORMALIZATION,
                    "versions": {"qwen-asr": "synthetic-only"},
                    "generation_config": {"do_sample": False, "num_beams": 1},
                    "model_load_seconds": 1.5, "warmup_seconds": .5}
        write_json(run / "run.json", metadata)
        # The saved score artifact is generated with the actual unchanged scorer.
        from speaker_lab.glossary_experiment import summarize

        write_json(run / "scores.json", summarize(rows, predictions, terms, controls))
        comparison = report.analysis.analyze_run(run, manifest, protocol / "correct.json",
                                                  protocol / "unrelated.json", bootstrap_replicates=7,
                                                  bootstrap_seed=17)
        comparison["execution"]["timing_caveat"] = "Synthetic note; not a measured performance claim"
        write_json(run / "comparison.json", comparison)
        spec = {"normalization": report.NORMALIZATION,
                "dataset": {"repo_id": "synthetic/dataset", "revision": "fixed",
                            "evaluation_split": "validation", "glossary_split": "train"},
                "model": model, "selection": {"count": len(rows),
                    "groups": {group: sum(row["group"] == group for row in rows) for group in sorted({r["group"] for r in rows})},
                    "total_audio_seconds": 2 * len(rows), "min_duration_seconds": 1, "max_duration_seconds": 12},
                "input_sha256": {"runtime_manifest": report.analysis.file_hash(manifest)},
                "source_sha256": {key: report.analysis.file_hash(report.REPO / key) for key in report.CORE_SOURCES}}
        if name == "new":
            spec["selection"].update(eligible_count=5, prior_excluded_count=2)
        seal_protocol(protocol, spec)
        arguments.extend((run, manifest, protocol))
    return tuple(arguments)


def test_publish_recomputes_micro_counts_transitions_slices_timing_and_keeps_ci(cohorts, tmp_path):
    before = {path: report.analysis.file_hash(path) for path in tmp_path.rglob("*") if path.is_file()}
    result = report.summarize_cohorts(*cohorts)
    assert {path: report.analysis.file_hash(path) for path in before} == before
    old, new, pooled = (result["cohorts"][key] for key in ("development", "remaining", "pooled"))
    assert [item["reference_statistics"]["clips"] for item in (old, new, pooled)] == [2, 3, 5]
    old_counts, new_counts, total = [item["summary"]["all"]["none"] for item in (old, new, pooled)]
    for field in report.analysis.COUNT_FIELDS:
        assert total[field] == old_counts[field] + new_counts[field]
    assert total["mer"] == (total["S"] + total["D"] + total["I"]) / total["N"]
    assert total["mer"] != (old_counts["mer"] + new_counts["mer"]) / 2
    paired = pooled["paired"]["all"]["personal_vs_none"]
    assert paired["reference_term_transitions"] == {
        "miss_to_hit": 2, "hit_to_miss": 1, "stable_hit": 1, "stable_miss": 0}
    assert paired["false_terms_gained"] == 1
    assert paired["bootstrap_ci"] is None
    assert old["paired"]["all"]["personal_vs_none"]["bootstrap_ci"] == read_json(cohorts[0] / "comparison.json")["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]
    assert pooled["summary"]["coverage:uncovered"]["personal"]["term_false_mentions"] == 1
    assert pooled["summary"]["group:english/coverage:covered"]["none"]["clips"] == 2
    assert pooled["execution"]["timed_inferences"] == 15
    assert pooled["execution"]["unique_audio_seconds"] == 10
    assert pooled["execution"]["total_timed_inference_seconds"] == 30
    assert pooled["reference_statistics"]["reference_unknown_markers"] == 1
    assert pooled["per_term"]["cache"]["conditions"]["personal"]["correct_mentions"] == 2
    public = json.dumps(result, ensure_ascii=False, allow_nan=False)
    assert str(tmp_path) not in public
    assert "检查 cache API" not in public and "one two three four" not in public
    assert result["verification"]["saved_scores_comparisons_and_ci_independently_recomputed"] is True
    assert len(result["provenance"]["cohorts"]["remaining"]["artifact_sha256"]["comparison.json"]) == 64


@pytest.mark.parametrize("position", [0, 3])
def test_incomplete_gate_precedes_manifest_or_prediction_read(cohorts, position):
    metadata = read_json(cohorts[position] / "run.json")
    metadata["status"] = "started"
    write_json(cohorts[position] / "run.json", metadata)
    cohorts[position + 1].write_text("malformed manifest must not be read", encoding="utf-8")
    (cohorts[position] / "predictions.jsonl").write_text("malformed", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        report.summarize_cohorts(*cohorts)


@pytest.mark.parametrize("change,match", [
    ("missing_prediction", "Incomplete paired"), ("duplicate_prediction", "Duplicate"),
    ("audio", "Audio hash mismatch"), ("cached_count", "Saved comparison mismatch: summary"),
    ("cached_ci", "Saved comparison mismatch: paired"), ("cached_score", "Saved score mismatch"),
    ("id_overlap", "ID intersection"), ("audio_overlap", "audio-hash intersection"),
    ("runtime", "runtime provenance differs"), ("ids", "Hash mismatch"),
    ("source", "Hash mismatch"), ("unfrozen", "FREEZE"),
    ("bootstrap_plan", "frozen bootstrap plan"), ("manifest", "manifest hash mismatch"),
    ("id_order", "frozen ID order"), ("eligible_count", "frozen eligible population"),
])
def test_rejects_incomplete_tampered_or_overlapping_evidence(cohorts, change, match):
    old, _, _, new, _, protocol = cohorts
    if change in ("missing_prediction", "duplicate_prediction"):
        path = new / "predictions.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1] if change == "missing_prediction" else lines + lines[:1]), encoding="utf-8")
    elif change == "audio":
        next(new.glob("*.wav")).write_bytes(b"changed")
    elif change.startswith("cached_"):
        path = new / ("scores.json" if change == "cached_score" else "comparison.json")
        value = read_json(path)
        if change == "cached_ci":
            value["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]["metrics"]["mer_delta_percentage_points"]["upper"] = 999
        else:
            value["slices" if change == "cached_score" else "summary"]["all"]["none"]["S"] += 1
        write_json(path, value)
    elif change in ("id_overlap", "audio_overlap", "runtime"):
        path = new / "run.json"
        value, previous = read_json(path), read_json(old / "run.json")
        if change == "id_overlap":
            value["audio_sha256"][next(iter(previous["audio_sha256"]))] = "f" * 64
        elif change == "audio_overlap":
            value["audio_sha256"][next(iter(value["audio_sha256"]))] = next(iter(previous["audio_sha256"].values()))
        else:
            value["seed"] += 1
        write_json(path, value)
    elif change == "ids":
        write_json(protocol / "selected_ids.json", ["wrong-id"])
    elif change == "source":
        spec = read_json(protocol / "protocol.json")
        spec["source_sha256"]["src/speaker_lab/glossary_metrics.py"] = "f" * 64
        seal_protocol(protocol, spec)
    elif change == "unfrozen":
        (protocol / "FREEZE.json").unlink()
    elif change == "bootstrap_plan":
        path = new / "comparison.json"
        value = read_json(path)
        value["analysis_parameters"]["bootstrap_replicates"] += 1
        write_json(path, value)
    elif change == "manifest":
        cohorts[4].write_text(cohorts[4].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif change == "id_order":
        path = protocol / "selected_ids.json"
        write_json(path, list(reversed(read_json(path))))
        seal_protocol(protocol, read_json(protocol / "protocol.json"))
    elif change == "eligible_count":
        spec = read_json(protocol / "protocol.json")
        spec["selection"]["eligible_count"] += 1
        seal_protocol(protocol, spec)
    with pytest.raises(ValueError, match=match):
        report.summarize_cohorts(*cohorts)


def cli_arguments(cohorts, output):
    result = []
    for name, values in zip(("old", "new"), (cohorts[:3], cohorts[3:])):
        for flag, path in zip(("run-dir", "manifest", "protocol-dir"), values):
            result.extend((f"--{name}-{flag}", str(path)))
    return [*result, "--output", str(output)]


def test_cli_writes_new_utf8_json_and_never_overwrites(cohorts, tmp_path):
    output = tmp_path / "published" / "cohorts.json"
    args = cli_arguments(cohorts, output)
    assert report.main(args) == 0
    first = output.read_bytes()
    assert first.endswith(b"\n") and b"\r\n" not in first
    with pytest.raises(SystemExit):
        report.main(args)
    assert output.read_bytes() == first


def test_failure_creates_no_partial_publication(cohorts, tmp_path):
    (cohorts[3] / "predictions.jsonl").write_text("invalid", encoding="utf-8")
    output = tmp_path / "failed-publication.json"
    with pytest.raises(SystemExit):
        report.main(cli_arguments(cohorts, output))
    assert not output.exists()
