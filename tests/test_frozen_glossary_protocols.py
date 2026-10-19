"""Audit portable preregistration records without loading data or model packages.

These checks bind historical records to one another, not to the current source
tree: later legitimate model/scorer development must remain possible.
"""

import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "experiments" / "glossary"
OLD = GLOSSARY / "protocols" / "ascend-val200-train-vocab30-v2"
NEW = GLOSSARY / "protocols" / "ascend-val-remaining725-train-vocab30-v1"
AUDIT = GLOSSARY / "audits" / "ascend-remaining-v1"
FROZEN_FILES = {
    "analysis_plan.json", "config.toml", "correct.json", "environment.json",
    "protocol.json", "README.md", "selected_ids.json", "selection_audit.json",
    "unrelated.json",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_hashes():
    return read_json(AUDIT / "preservation_before.json")["sha256"]


def test_freeze_inventory_has_no_missing_or_unrecorded_files():
    frozen = read_json(NEW / "FREEZE.json")
    assert set(frozen["sha256"]) == FROZEN_FILES
    assert {path.name for path in NEW.iterdir()} == FROZEN_FILES | {"FREEZE.json"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", value)
               for value in frozen["sha256"].values())


@pytest.mark.parametrize("filename", sorted(FROZEN_FILES))
def test_frozen_file_matches_its_recorded_bytes(filename):
    assert sha256(NEW / filename) == read_json(NEW / "FREEZE.json")["sha256"][filename]


def test_protocol_portable_hashes_match_freeze_and_actual_files():
    protocol = read_json(NEW / "protocol.json")
    frozen = read_json(NEW / "FREEZE.json")["sha256"]
    expected = FROZEN_FILES - {"protocol.json", "README.md"}
    assert set(protocol["portable_files_sha256"]) == expected
    for filename, digest in protocol["portable_files_sha256"].items():
        assert digest == frozen[filename] == sha256(NEW / filename)


def test_original_protocol_inputs_and_vocabulary_are_preserved():
    guard = recorded_hashes()
    old_prefix = OLD.relative_to(ROOT).as_posix() + "/"
    old_records = {name: digest for name, digest in guard.items()
                   if name.startswith(old_prefix)}
    assert set(Path(name).name for name in old_records) == {
        "analysis_plan.json", "correct.json", "protocol.json", "README.md",
        "selected_ids.json", "unrelated.json",
    }
    for name, digest in old_records.items():
        assert sha256(ROOT / name) == digest
    for filename in ("correct.json", "unrelated.json"):
        assert (NEW / filename).read_bytes() == (OLD / filename).read_bytes()
    # Compare to the recorded original configuration, not the editable current
    # configs/ file, which may validly change after this experiment.
    assert sha256(NEW / "config.toml") == guard["configs/glossary_cpu_pilot.toml"]


def test_selected_cohorts_partition_all_eligible_audio_ids():
    old_ids = read_json(OLD / "selected_ids.json")
    new_ids = read_json(NEW / "selected_ids.json")
    selection = read_json(NEW / "selection_audit.json")
    protocol = read_json(NEW / "protocol.json")
    assert len(old_ids) == len(set(old_ids)) == 200
    assert len(new_ids) == len(set(new_ids)) == 725
    assert not set(old_ids) & set(new_ids)
    assert set(old_ids) | set(new_ids) == set(selection["eligible_audio_sha256"])
    assert len(selection["eligible_audio_sha256"]) == selection["eligible_count"] == 925
    assert selection["count"] == protocol["selection"]["count"] == len(new_ids)
    assert selection["prior_count"] == len(old_ids)
    assert selection["groups"] == protocol["selection"]["groups"] == {
        "mixed": 107, "mandarin": 453, "english": 165,
    }
    assert selection["selected_ids_sha256"] == sha256(NEW / "selected_ids.json")
    assert selection["reference_not_used_for_selection"] is True
    assert selection["vocabulary_or_predictions_used_for_selection"] is False
    assert selection["official_test_opened"] is False


def test_frozen_provenance_agrees_with_original_preservation_records():
    protocol = read_json(NEW / "protocol.json")
    guard = recorded_hashes()
    input_sources = {
        "source_manifest": "experiments/glossary/data/ascend-validation-737e9800/manifest.jsonl",
        "prior_selected_ids": (OLD / "selected_ids.json").relative_to(ROOT).as_posix(),
        "prior_protocol": (OLD / "protocol.json").relative_to(ROOT).as_posix(),
    }
    for field, original in input_sources.items():
        assert protocol["input_sha256"][field] == guard[original]
    # Historical source hashes are cross-checked only between frozen records.
    # Do not compare them with current .py files or load ignored run/data files.
    for source in ("src/speaker_lab/glossary_metrics.py",
                   "src/speaker_lab/glossary_experiment.py",
                   "scripts/analyze_glossary_comparison.py"):
        assert protocol["source_sha256"][source] == guard[source]
    selection = read_json(NEW / "selection_audit.json")
    assert selection["sampling_source_code_sha256"] == protocol["source_sha256"][
        "scripts/freeze_ascend_remaining.py"]
    assert selection["source_manifest_sha256"] == protocol["input_sha256"]["source_manifest"]
    assert selection["manifest_sha256"] == protocol["input_sha256"]["runtime_manifest"]


def test_six_event_review_preserves_scores_and_discloses_unfinished_listening():
    protocol = read_json(NEW / "protocol.json")
    review = read_json(AUDIT / "six_event_audit.json")
    guard = recorded_hashes()
    assert protocol["six_event_review"]["artifact"] == (
        AUDIT / "six_event_audit.json").relative_to(ROOT).as_posix()
    assert protocol["six_event_review"]["sha256"] == sha256(AUDIT / "six_event_audit.json")
    assert protocol["six_event_review"]["text_and_scoring_verified"] is True
    assert protocol["six_event_review"]["listened"] is False
    assert review["primary_scores_unchanged"] is True
    assert review["reviewer_prediction_aware"] is True
    assert review["model_consensus_is_not_truth"] is True
    assert "no listening" in review["scope"]
    assert review["source_sha256"]["scorer"] == guard["src/speaker_lab/glossary_metrics.py"]
    assert review["source_sha256"]["correct"] == sha256(NEW / "correct.json")
    assert len(review["events"]) == 6
    assert {event["id"] for event in review["events"]} == {
        "00621", "00062", "00957", "01010", "00928", "00904",
    }
    for event in review["events"]:
        assert event["text_and_scoring_verified"] is True
        assert event["audio_bytes_verified"] is True
        assert event["listened"] is False
        assert event["heard_text"] is None
        assert event["acoustic_adjudication"] == "unresolved"
        assert set(event["predictions"]) == {"none", "personal", "unrelated"}


def test_replication_keeps_main_rules_and_predeclared_analysis_boundary():
    old_protocol = read_json(OLD / "protocol.json")
    protocol = read_json(NEW / "protocol.json")
    plan = read_json(NEW / "analysis_plan.json")
    assert protocol["normalization"] == old_protocol["normalization"] == "nfkc_han_latin_digits_v1"
    assert "no aliases" in plan["term_matching"]
    assert "no vocabulary or scoring changes" in protocol["prohibitions"]
    assert protocol["execution"]["expected_timed_inferences"] == 725 * 3
    assert protocol["dataset"]["evaluation_split"] == "validation"
    assert protocol["dataset"]["test_used"] is False
    assert plan["uncertainty"] == {
        "method": "paired clip percentile bootstrap stratified by language; retain each language count",
        "replicates": 1000, "seed": 20261018, "confidence": 0.95,
        "unit": "clip, not speaker/session", "multiple_comparison_adjustment": False,
    }
    assert "all 2175 predictions" in plan["analysis_start_gate"]
    assert "no interim treatment-effect analysis" in plan["analysis_start_gate"]
    assert "not independent validation" in plan["cohorts"]["pooled925"]
