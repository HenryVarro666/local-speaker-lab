"""Analysis integrity checks; no ASR, network, GPU, or audio decoder is needed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_glossary_comparison.py"
SPEC = importlib.util.spec_from_file_location("analyze_glossary_comparison_tests", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def compare(reference, before, after, terms):
    return analysis.compare_mentions(analysis.analyze_text(reference, before, terms),
                                     analysis.analyze_text(reference, after, terms))


def test_missed_reference_occurrence_is_recovered_and_correct_one_can_break():
    result = compare("检查 cache 和 API", "检查 cash 和 API", "检查 cache 和 app", ["cache", "API"])
    assert result["reference_term_transitions"] == {
        "miss_to_hit": 1, "hit_to_miss": 1, "stable_hit": 0, "stable_miss": 0}
    assert result["baseline_wrong_terms_corrected"] == result["baseline_correct_terms_broken"] == 1
    assert {event["term"]: event["transition"] for event in result["term_transition_events"]} == {
        "cache": "miss_to_hit", "api": "hit_to_miss"}


def test_reordered_terms_do_not_receive_bag_of_words_credit():
    result = compare("cache API", "cache API", "API cache", ["cache", "API"])
    assert result["baseline_correct_terms_broken"] == 2
    assert result["baseline_wrong_terms_corrected"] == 0
    assert result["false_terms_gained"] == 2


def test_repeated_mentions_have_distinct_reference_spans():
    result = compare("cache cache", "cache", "cache cache", ["cache"])
    assert result["reference_term_transitions"]["miss_to_hit"] == 1
    assert result["reference_term_transitions"]["stable_hit"] == 1
    assert result["false_terms_gained"] == 0


def test_longest_phrase_fragment_is_not_labeled_an_acoustic_insertion():
    before = analysis.analyze_text("social media", "social", ["social", "social media"])
    after = analysis.analyze_text("social media", "social media", ["social", "social media"])
    assert before["score"]["term_false_mentions"] == 1
    assert before["false_mentions"][0]["kind"] == "fragment_of_reference_term"
    assert before["false_mentions"][0]["reference_term_containers"][0]["term"] == "social media"
    result = analysis.compare_mentions(before, after)
    assert result["baseline_wrong_terms_corrected"] == 1
    assert result["false_terms_lost"] == 1
    assert result["false_term_loss_events"][0]["kind"] == "fragment_of_reference_term"


def test_false_substitution_stays_stable_while_new_insertion_is_gained():
    result = compare("检查 group work", "检查 good work", "检查 good work cache", ["good", "work", "cache"])
    assert result["false_terms_stable"] == 1
    assert result["false_terms_gained"] == 1
    assert result["false_terms_lost"] == 0
    assert result["false_term_gain_events"][0]["kind"] == "insertion"
    before = analysis.analyze_text("检查 group work", "检查 good work", ["good", "work"])
    assert before["false_mentions"][0]["kind"] == "substitution"
    assert before["false_mentions"][0]["aligned_reference_tokens"] == ["group"]


def test_false_mentions_are_multisets_and_insertions_have_reference_anchors():
    result = compare("继续", "继续 cache cache", "继续 cache", ["cache"])
    assert result["false_terms_stable"] == 1
    assert result["false_terms_lost"] == 1
    moved = compare("继续", "cache 继续", "继续 cache", ["cache"])
    assert moved["false_terms_gained"] == moved["false_terms_lost"] == 1


def test_no_partial_latin_word_or_alias_match():
    result = analysis.analyze_text("cached A P I", "cached A P I", ["cache", "API"])
    assert result["score"]["term_ref_mentions"] == result["score"]["term_hyp_mentions"] == 0


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_run(tmp_path, *, clips=None, terms=None, controls=None, directory="run"):
    directory = tmp_path / directory
    directory.mkdir()
    terms = terms or ["cache"]
    controls = controls or ["cairn"]
    clips = clips or [
        {"id": "positive", "reference": "检查 cache", "group": "mixed",
         "none": "检查 cash", "personal": "检查 cache", "unrelated": "检查 cairn"},
        {"id": "negative", "reference": "今天继续", "group": "mandarin",
         "none": "今天继续", "personal": "今天 cache 继续", "unrelated": "今天继续"},
    ]
    manifest, correct, unrelated = directory / "manifest.jsonl", directory / "correct.json", directory / "unrelated.json"
    rows, predictions, audio_hashes = [], [], {}
    for clip in clips:
        audio = directory / (clip["id"] + ".wav")
        # Hash-only analysis must not decode samples or import soundfile.
        audio.write_bytes(("frozen audio " + clip["id"]).encode())
        audio_hashes[clip["id"]] = sha(audio)
        rows.append({"id": clip["id"], "reference": clip["reference"], "audio": audio.name,
                     "group": clip["group"], "split": "validation"})
        for condition in analysis.CONDITIONS:
            predictions.append({"id": clip["id"], "condition": condition, "text": clip[condition],
                                "seconds": 2.0 if condition == "personal" else 1.0, "duration_seconds": 2.0})
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    write_json(correct, terms)
    write_json(unrelated, controls)
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in predictions), encoding="utf-8")
    metadata = {"status": "complete", "manifest_sha256": sha(manifest), "personal_sha256": sha(correct),
                "unrelated_sha256": sha(unrelated), "audio_sha256": audio_hashes,
                "contexts": analysis.make_contexts(terms, controls), "config": {"model": "fixed-model"}}
    write_json(directory / "run.json", metadata)
    return directory, manifest, correct, unrelated


def test_complete_run_has_covered_uncovered_cohorts_transitions_and_union_fp(tmp_path):
    paths = fixture_run(tmp_path)
    result = analysis.analyze_run(*paths, bootstrap_replicates=24)
    assert result["coverage"]["all"] == {"clips": 2, "positive_clips": 1, "negative_clips": 1, "reference_mentions": 1}
    assert result["summary"]["all"]["personal"]["term_recall"] == 1
    assert result["summary"]["all"]["personal"]["term_precision"] == .5
    assert result["summary"]["all"]["personal"]["no_term_false_mentions"] == 1
    assert result["summary"]["all"]["unrelated"]["all_candidates"]["term_false_mentions"] == 1
    assert result["paired"]["all"]["personal_vs_none"]["baseline_wrong_terms_corrected"] == 1
    assert result["paired"]["all"]["personal_vs_none"]["false_terms_gained"] == 1
    assert result["paired"]["all"]["unrelated_vs_none"]["all_candidates"]["false_terms_gained"] == 1
    assert result["summary"]["coverage:covered"]["personal"]["term_precision"] == 1
    assert result["summary"]["coverage:uncovered"]["personal"]["term_recall"] is None
    assert result["summary"]["group:mixed/coverage:covered"]["none"]["clips"] == 1
    assert result["summary"]["group:mandarin/coverage:covered"]["none"]["clips"] == 0
    assert result["summary"]["all"]["personal"]["rtf"] == 1


def test_bootstrap_is_paired_stratified_reproducible_and_micro_averaged(tmp_path):
    clips = [
        {"id": "long", "reference": "one two three four five six seven eight nine cache", "group": "english",
         "none": "one two three four five six seven eight nine cash",
         "personal": "one two three four five six seven eight nine cache", "unrelated": "one two three four five six seven eight nine cash"},
        {"id": "short", "reference": "cache", "group": "mixed", "none": "cash", "personal": "cache", "unrelated": "cash"},
    ]
    paths = fixture_run(tmp_path, clips=clips)
    first = analysis.analyze_run(*paths, bootstrap_replicates=31, bootstrap_seed=17)
    second = analysis.analyze_run(*paths, bootstrap_replicates=31, bootstrap_seed=17)
    a = first["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]
    b = second["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]
    assert a == b
    assert a["strata"] == {"english": 1, "mixed": 1}
    metric = a["metrics"]["mer_delta_percentage_points"]
    assert metric["estimate"] == pytest.approx(-200 / 11)
    assert metric["lower"] == metric["upper"] == metric["estimate"]
    assert metric["valid_replicates"] == 31


def test_undefined_ratios_have_no_bootstrap_values(tmp_path):
    clips = [{"id": "emptyterms", "reference": "今天继续", "group": "mandarin",
              "none": "今天继续", "personal": "今天继续", "unrelated": "今天继续"}]
    result = analysis.analyze_run(*fixture_run(tmp_path, clips=clips), bootstrap_replicates=10)
    metric = result["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]["metrics"]["term_recall_delta_percentage_points"]
    assert metric == {"estimate": None, "lower": None, "upper": None, "valid_replicates": 0}


def test_incomplete_run_is_rejected_before_prediction_analysis(tmp_path):
    paths = fixture_run(tmp_path)
    metadata = json.loads((paths[0] / "run.json").read_text(encoding="utf-8"))
    metadata["status"] = "started"
    write_json(paths[0] / "run.json", metadata)
    (paths[0] / "predictions.jsonl").write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze_run(*paths)


def test_missing_pair_and_changed_glossary_are_rejected(tmp_path):
    paths = fixture_run(tmp_path)
    predictions = (paths[0] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    (paths[0] / "predictions.jsonl").write_text("\n".join(predictions[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete paired"):
        analysis.analyze_run(*paths)
    write_json(paths[2], ["changed"])
    with pytest.raises(ValueError, match="personal_sha256"):
        analysis.analyze_run(*paths)


def test_optional_old_run_compares_only_identical_audio_overlap(tmp_path):
    paths = fixture_run(tmp_path)
    old_clip = [{"id": "positive", "reference": "检查 cache", "group": "mixed",
                 "none": "检查 cash", "personal": "检查 cache", "unrelated": "检查 cairn"}]
    old = fixture_run(tmp_path, clips=old_clip, directory="old")
    result = analysis.analyze_run(*paths, old_run_dir=old[0], bootstrap_replicates=0)
    check = result["old_baseline_consistency"]
    assert check["overlap_clips"] == check["raw_identical_clips"] == 1
    assert check["new_only_clips"] == 1
    assert check["old_baseline_rescored"] == check["new_baseline_same_overlap"]


def test_sparse_bootstrap_reports_undefined_recall_replicates(tmp_path):
    clips = [
        {"id": "one", "reference": "检查 cache", "group": "mixed",
         "none": "检查 cash", "personal": "检查 cache", "unrelated": "检查 cash"},
        {"id": "two", "reference": "继续", "group": "mixed",
         "none": "继续", "personal": "继续", "unrelated": "继续"},
    ]
    result = analysis.analyze_run(*fixture_run(tmp_path, clips=clips), bootstrap_replicates=128)
    metric = result["paired"]["all"]["personal_vs_none"]["bootstrap_ci"]["metrics"]["term_recall_delta_percentage_points"]
    assert 0 < metric["valid_replicates"] < 128
    assert metric["estimate"] == metric["lower"] == metric["upper"] == 100


@pytest.mark.parametrize("bad_seconds", [True, float("nan"), float("inf"), -1])
def test_invalid_timing_cannot_produce_misleading_statistics(tmp_path, bad_seconds):
    paths = fixture_run(tmp_path)
    path = paths[0] / "predictions.jsonl"
    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    predictions[0]["seconds"] = bad_seconds
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid timing"):
        analysis.analyze_run(*paths)


def test_present_but_empty_cached_scores_cannot_be_marked_verified(tmp_path):
    paths = fixture_run(tmp_path)
    write_json(paths[0] / "scores.json", {"slices": {}})
    with pytest.raises(ValueError, match="missing required"):
        analysis.analyze_run(*paths, bootstrap_replicates=0)


def test_empty_cached_coverage_cohort_timings_are_valid_nulls(tmp_path):
    paths = fixture_run(tmp_path)
    first = analysis.analyze_run(*paths, bootstrap_replicates=0)
    write_json(paths[0] / "scores.json", {"slices": first["summary"]})
    result = analysis.analyze_run(*paths, bootstrap_replicates=0)
    assert result["independent_verification"]["saved_scores_verified"] is True
    assert result["summary"]["group:mandarin/coverage:covered"]["none"]["warm_p50_seconds"] is None


@pytest.mark.parametrize("cohort,value", [("all", None), ("all", float("nan")),
                                           ("all", float("inf")), ("all", True),
                                           ("group:mandarin/coverage:covered", 0)])
def test_cached_undefined_or_nonfinite_timing_mismatches_are_rejected(tmp_path, cohort, value):
    paths = fixture_run(tmp_path)
    first = analysis.analyze_run(*paths, bootstrap_replicates=0)
    first["summary"][cohort]["none"]["warm_p50_seconds"] = value
    write_json(paths[0] / "scores.json", {"slices": first["summary"]})
    with pytest.raises(ValueError, match="Saved timing mismatch"):
        analysis.analyze_run(*paths, bootstrap_replicates=0)


def test_old_run_cannot_hide_a_completely_missing_clip(tmp_path):
    paths = fixture_run(tmp_path)
    old = fixture_run(tmp_path, directory="old")
    path = old[0] / "predictions.jsonl"
    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions if row["id"] != "negative"), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete paired"):
        analysis.analyze_run(*paths, old_run_dir=old[0], bootstrap_replicates=0)


def test_cli_refuses_overwrite_and_never_mutates_frozen_inputs(tmp_path):
    paths = fixture_run(tmp_path)
    inputs = [*paths[1:], paths[0] / "run.json", paths[0] / "predictions.jsonl"]
    before = {path: sha(path) for path in inputs}
    args = ["--run-dir", str(paths[0]), "--manifest", str(paths[1]), "--correct", str(paths[2]),
            "--unrelated", str(paths[3]), "--bootstrap-replicates", "4"]
    assert analysis.main(args) == 0
    output = paths[0] / "comparison.json"
    digest = sha(output)
    with pytest.raises(SystemExit):
        analysis.main(args)
    assert sha(output) == digest
    assert {path: sha(path) for path in inputs} == before
