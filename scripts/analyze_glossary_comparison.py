#!/usr/bin/env python3
"""Analyze a COMPLETE fixed-audio, three-arm glossary run without ML dependencies.

Reference term mentions use stable normalized token spans, the scorer's
left-to-right longest matching, and its deterministic minimum-edit alignment.
This distinguishes recovery of a previously missed occurrence from a new miss
of a previously correct occurrence. Repeated words retain edit-alignment
ambiguity; no timestamps or semantic aliases are inferred.

False mentions are matched across arms by literal term plus aligned reference
slots / insertion boundaries, as a multiset within each clip. A substituted
candidate is not called an insertion. The classification is descriptive of the
edit alignment, not a claim about an acoustic hallucination's cause.

Bootstrap intervals resample paired CLIPS within language groups, retaining each
group's sample count. They are percentile 95% intervals with a fixed seed, not
speaker/session-cluster intervals. Undefined ratios are omitted per metric and
their valid-replicate count is reported. Raw primary scoring is never changed.

CLI writes a new comparison.json by default, refusing to overwrite an existing
file. It never modifies predictions, metadata, glossaries, samples, or models.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

# Also run directly from a checkout that has not been installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_lab.glossary_experiment import (  # noqa: E402
    file_hash, load_manifest, load_terms, make_contexts, prediction_index, read_jsonl,
)
from speaker_lab.glossary_metrics import aggregate_scores, normalize_tokens, score_pair  # noqa: E402


CONDITIONS = ("none", "personal", "unrelated")
FALSE_KINDS = ("insertion", "substitution", "substitution_and_insertion",
               "fragment_of_reference_term", "other_alignment")
COUNT_FIELDS = ("S", "D", "I", "N", "term_ref_mentions", "term_hyp_mentions",
                "term_correct_mentions", "term_false_mentions")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def term_mentions(tokens: list[str], terms: list[str]) -> list[dict]:
    """Select nonoverlapping longest matches, with reference-stable span IDs."""
    patterns = {tuple(normalize_tokens(term)) for term in terms}
    if () in patterns:
        raise ValueError("Empty normalized term")
    patterns = sorted(patterns, key=lambda term: (-len(term), term))
    result, start = [], 0
    while start < len(tokens):
        for term in patterns:
            end = start + len(term)
            if tuple(tokens[start:end]) == term:
                result.append({"start": start, "end": end, "term": " ".join(term)})
                start = end
                break
        else:
            start += 1
    return result


def _alignment(reference: list[str], hypothesis: list[str]) -> tuple[dict, dict]:
    """Return independent counts and per-hypothesis-token alignment operations."""
    table = [list(range(len(hypothesis) + 1))]
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            row.append(min(table[i - 1][j - 1] + (ref != hyp),
                           table[i - 1][j] + 1, row[j - 1] + 1))
        table.append(row)
    counts = {"S": 0, "D": 0, "I": 0, "N": len(reference)}
    operations = {}
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            same = reference[i - 1] == hypothesis[j - 1]
            operations[j - 1] = {"op": "M" if same else "S", "ref": i - 1, "boundary": None}
            counts["S"] += not same
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            counts["D"] += 1
            i -= 1
        else:
            # i is the reference cursor immediately before this inserted token.
            operations[j - 1] = {"op": "I", "ref": None, "boundary": i}
            counts["I"] += 1
            j -= 1
    return counts, operations


def analyze_text(reference: str, hypothesis: str, terms: list[str]) -> dict:
    """Public-score-compatible details for individual reference occurrences."""
    ref, hyp = normalize_tokens(reference), normalize_tokens(hypothesis)
    score = score_pair(reference, hypothesis, terms)
    counts, operations = _alignment(ref, hyp)
    if any(counts[field] != score[field] for field in counts):
        raise RuntimeError("Analysis alignment differs from the primary scorer")
    refs = term_mentions(ref, terms)
    spans = {(mention["start"], mention["end"]) for mention in refs}
    hits, false = set(), []
    for mention in term_mentions(hyp, terms):
        start, end = mention["start"], mention["end"]
        first = operations[start]["ref"]
        span = (first, first + end - start) if first is not None else None
        if span in spans and all(
            operations[position]["op"] == "M"
            and operations[position]["ref"] == first + position - start
            for position in range(start, end)
        ):
            hits.add(span)
            continue
        tags = {operations[position]["op"] for position in range(start, end)}
        containers = [m for m in refs if first is not None
                      and m["start"] <= first and first + end - start <= m["end"]
                      and (m["start"], m["end"]) != span]
        contiguous_match = first is not None and all(
            operations[position]["op"] == "M"
            and operations[position]["ref"] == first + position - start
            for position in range(start, end)
        )
        if contiguous_match and containers:
            kind = "fragment_of_reference_term"
        elif tags == {"I"}:
            kind = "insertion"
        elif "S" in tags and "I" in tags:
            kind = "substitution_and_insertion"
        elif "S" in tags:
            kind = "substitution"
        else:
            kind = "other_alignment"
        locations = [[operations[position]["op"], operations[position]["ref"],
                      operations[position]["boundary"]] for position in range(start, end)]
        false.append({
            **mention, "kind": kind, "reference_locations": locations,
            "reference_term_containers": containers,
            "aligned_reference_tokens": [ref[operations[position]["ref"]]
                                         if operations[position]["ref"] is not None else None
                                         for position in range(start, end)],
        })
    detailed_refs = [{**mention, "hit": (mention["start"], mention["end"]) in hits} for mention in refs]
    if len(hits) != score["term_correct_mentions"] or len(false) != score["term_false_mentions"]:
        raise RuntimeError("Mention analysis differs from the primary scorer")
    return {"score": score, "reference_mentions": detailed_refs, "false_mentions": false}


def _false_key(mention: dict) -> tuple:
    return mention["term"], mention["kind"], tuple(tuple(location) for location in mention["reference_locations"])


def compare_mentions(baseline: dict, condition: dict) -> dict:
    """Compare occurrences on the SAME reference; false mentions use multisets."""
    base = {(m["start"], m["end"], m["term"]): m["hit"] for m in baseline["reference_mentions"]}
    other = {(m["start"], m["end"], m["term"]): m["hit"] for m in condition["reference_mentions"]}
    if base.keys() != other.keys():
        raise ValueError("Cannot compare term mentions from different references/vocabularies")
    transitions = {"miss_to_hit": 0, "hit_to_miss": 0, "stable_hit": 0, "stable_miss": 0}
    events = []
    for key, before in base.items():
        after = other[key]
        name = ("stable_hit" if after else "hit_to_miss") if before else ("miss_to_hit" if after else "stable_miss")
        transitions[name] += 1
        if before != after:
            events.append({"start": key[0], "end": key[1], "term": key[2], "transition": name})
    before = Counter(_false_key(m) for m in baseline["false_mentions"])
    after = Counter(_false_key(m) for m in condition["false_mentions"])
    gained, lost, stable = after - before, before - after, before & after

    def describe(counter: Counter) -> list[dict]:
        return [{"term": key[0], "kind": key[1], "reference_locations": [list(x) for x in key[2]],
                 "count": count} for key, count in sorted(counter.items(), key=lambda pair: repr(pair[0]))]

    return {"reference_term_transitions": transitions,
            "baseline_wrong_terms_corrected": transitions["miss_to_hit"],
            "baseline_correct_terms_broken": transitions["hit_to_miss"],
            "term_transition_events": events,
            "false_terms_gained": sum(gained.values()), "false_terms_lost": sum(lost.values()),
            "false_terms_stable": sum(stable.values()),
            "false_term_gain_events": describe(gained), "false_term_loss_events": describe(lost)}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _metric_deltas(base: dict, other: dict, clips: int) -> dict:
    def difference(field_numerator, field_denominator, scale=100):
        before = _rate(base[field_numerator], base[field_denominator])
        after = _rate(other[field_numerator], other[field_denominator])
        return scale * (after - before) if before is not None and after is not None else None

    base = {**base, "edits": base["S"] + base["D"] + base["I"]}
    other = {**other, "edits": other["S"] + other["D"] + other["I"]}
    return {"mer_delta_percentage_points": difference("edits", "N"),
            "term_recall_delta_percentage_points": difference("term_correct_mentions", "term_ref_mentions"),
            "term_precision_delta_percentage_points": difference("term_correct_mentions", "term_hyp_mentions"),
            "false_terms_delta_per_100_clips": _rate(100 * (other["term_false_mentions"] - base["term_false_mentions"]), clips)}


def _quantile(values: list[float], probability: float):
    if not values:
        return None
    values = sorted(values)
    position = probability * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_comparison(cases: list[dict], condition: str, *, replicates: int, seed: int) -> dict:
    """Paired, language-stratified clip bootstrap; cached counts, no rescoring."""
    if replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    strata = defaultdict(list)
    for position, case in enumerate(cases):
        strata[case["group"]].append(position)
    strata = [strata[group] for group in sorted(strata)]
    rng, samples = random.Random(seed), defaultdict(list)
    base_point = aggregate_scores([case["conditions"]["none"]["score"] for case in cases])
    other_point = aggregate_scores([case["conditions"][condition]["score"] for case in cases])
    point = _metric_deltas(base_point, other_point, len(cases))
    for _ in range(replicates if cases else 0):
        chosen = [rng.choice(stratum) for stratum in strata for _ in stratum]
        base = {field: 0 for field in COUNT_FIELDS}
        other = {field: 0 for field in COUNT_FIELDS}
        for position in chosen:
            for target, arm in ((base, "none"), (other, condition)):
                counts = cases[position]["conditions"][arm]["score"]
                for field in COUNT_FIELDS:
                    target[field] += counts[field]
        for metric, value in _metric_deltas(base, other, len(chosen)).items():
            if value is not None:
                samples[metric].append(value)
    return {"method": "paired language-stratified clip percentile bootstrap",
            "confidence_level": 0.95, "requested_replicates": replicates, "seed": seed,
            "strata": {group: sum(case["group"] == group for case in cases)
                       for group in sorted({case["group"] for case in cases})},
            "limitation": "Does not model speaker/session correlation or model-run variability",
            "metrics": {metric: {"estimate": value, "lower": _quantile(samples[metric], .025),
                                 "upper": _quantile(samples[metric], .975),
                                 "valid_replicates": len(samples[metric])}
                        for metric, value in point.items()}}


def _summary(cases: list[dict], condition: str) -> dict:
    result = aggregate_scores([case["conditions"][condition]["score"] for case in cases])
    result["all_candidates"] = aggregate_scores([case["conditions"][condition]["all_candidates"]["score"] for case in cases])
    false = [mention for case in cases for mention in case["conditions"][condition]["false_mentions"]]
    result["false_term_mechanisms"] = {kind: sum(m["kind"] == kind for m in false) for kind in FALSE_KINDS}
    positive = [case for case in cases if case["conditions"]["none"]["score"]["term_ref_mentions"]]
    result["covered_clip_scores"] = aggregate_scores([case["conditions"][condition]["score"] for case in positive])
    times = [case["conditions"][condition]["seconds"] for case in cases]
    duration = sum(case["duration_seconds"] for case in cases)
    result.update(total_inference_seconds=sum(times), audio_seconds=duration,
                  warm_p50_seconds=statistics.median(times) if times else None,
                  warm_p95_seconds=sorted(times)[math.ceil(.95 * len(times)) - 1] if times else None,
                  rtf=_rate(sum(times), duration))
    return result


def _paired(cases: list[dict], condition: str, *, replicates: int, seed: int) -> dict:
    differences = [case["comparisons"][condition + "_vs_none"] for case in cases]
    transitions = {name: sum(row["reference_term_transitions"][name] for row in differences)
                   for name in ("miss_to_hit", "hit_to_miss", "stable_hit", "stable_miss")}
    baseline = aggregate_scores([case["conditions"]["none"]["score"] for case in cases])
    current = aggregate_scores([case["conditions"][condition]["score"] for case in cases])
    union_differences = [row["all_candidates"] for row in differences]
    union_transitions = {name: sum(row["reference_term_transitions"][name] for row in union_differences)
                         for name in ("miss_to_hit", "hit_to_miss", "stable_hit", "stable_miss")}
    return {"improved_clips": sum(row["edit_delta"] < 0 for row in differences),
            "same_edit_clips": sum(row["edit_delta"] == 0 for row in differences),
            "worsened_clips": sum(row["edit_delta"] > 0 for row in differences),
            "edit_delta": sum(row["edit_delta"] for row in differences),
            **_metric_deltas(baseline, current, len(cases)),
            "reference_term_transitions": transitions,
            "baseline_wrong_terms_corrected": transitions["miss_to_hit"],
            "baseline_correct_terms_broken": transitions["hit_to_miss"],
            "false_terms_gained": sum(row["false_terms_gained"] for row in differences),
            "false_terms_lost": sum(row["false_terms_lost"] for row in differences),
            "false_terms_stable": sum(row["false_terms_stable"] for row in differences),
            "all_candidates": {
                "reference_term_transitions": union_transitions,
                "baseline_wrong_terms_corrected": union_transitions["miss_to_hit"],
                "baseline_correct_terms_broken": union_transitions["hit_to_miss"],
                "false_terms_gained": sum(row["false_terms_gained"] for row in union_differences),
                "false_terms_lost": sum(row["false_terms_lost"] for row in union_differences),
                "false_terms_stable": sum(row["false_terms_stable"] for row in union_differences)},
            "normalized_text_changed_clips": sum(row["normalized_text_changed"] for row in differences),
            "raw_text_changed_clips": sum(row["raw_text_changed"] for row in differences),
            "bootstrap_ci": bootstrap_comparison(cases, condition, replicates=replicates, seed=seed)}


def _require_hash(metadata: dict, field: str, path: Path) -> None:
    if not isinstance(metadata.get(field), str) or file_hash(path) != metadata[field]:
        raise ValueError(f"Missing or mismatched {field}: {path}")


def _old_baseline(old_run_dir: Path, cases: list[dict], terms: list[str], unrelated: list[str], metadata: dict) -> dict:
    old = _read_json(old_run_dir / "run.json")
    if old.get("status") != "complete":
        raise ValueError("Old consistency-check run is not complete")
    raw = read_jsonl(old_run_dir / "predictions.jsonl")
    declared_hashes = old.get("audio_sha256")
    if not isinstance(declared_hashes, dict) or not declared_hashes or any(
        not isinstance(clip_id, str) or not isinstance(digest, str)
        for clip_id, digest in declared_hashes.items()
    ):
        raise ValueError("Old run requires a nonempty audio hash ID mapping")
    old_ids = set(declared_hashes)
    old_index = prediction_index(raw, [{"id": clip_id} for clip_id in old_ids])
    overlap = [case for case in cases if case["id"] in old_ids]
    for case in overlap:
        clip_id = case["id"]
        if old.get("audio_sha256", {}).get(clip_id) != metadata["audio_sha256"][clip_id]:
            raise ValueError(f"Old/new audio differs for overlap clip {clip_id}")
    return {"purpose": "Rescore only the overlapping old empty-context predictions with NEW fixed candidates",
            "old_run": str(old_run_dir.resolve()), "overlap_clips": len(overlap),
            "new_only_clips": len(cases) - len(overlap), "old_only_clips": len(old_ids) - len(overlap),
            "configuration_equal": old.get("config") == metadata.get("config"),
            "raw_identical_clips": sum(old_index[c["id"], "none"]["text"] == c["conditions"]["none"]["text"] for c in overlap),
            "normalized_identical_clips": sum(normalize_tokens(old_index[c["id"], "none"]["text"]) == normalize_tokens(c["conditions"]["none"]["text"]) for c in overlap),
            "old_baseline_rescored": aggregate_scores([score_pair(c["reference"], old_index[c["id"], "none"]["text"], terms) for c in overlap]),
            "new_baseline_same_overlap": aggregate_scores([c["conditions"]["none"]["score"] for c in overlap]),
            "old_union_rescored": aggregate_scores([score_pair(c["reference"], old_index[c["id"], "none"]["text"], terms + unrelated) for c in overlap]),
            "reference_scope": "Current frozen manifest references; overlapping audio hashes verified",
            "old_artifact_sha256": {name: file_hash(old_run_dir / name) for name in ("run.json", "predictions.jsonl")}}


def analyze_run(run_dir: Path, manifest: Path, correct: Path, unrelated: Path, *,
                old_run_dir: Path | None = None, bootstrap_replicates: int = 1000,
                bootstrap_seed: int = 20261018) -> dict:
    """Read completed artifacts and return an analysis; never mutate run inputs."""
    run_dir, manifest, correct, unrelated = map(lambda p: Path(p).resolve(), (run_dir, manifest, correct, unrelated))
    metadata = _read_json(run_dir / "run.json")
    if metadata.get("status") != "complete":
        raise ValueError("Refusing to analyze an incomplete run")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    _require_hash(metadata, "manifest_sha256", manifest)
    _require_hash(metadata, "personal_sha256", correct)
    _require_hash(metadata, "unrelated_sha256", unrelated)
    terms, controls = load_terms(correct), load_terms(unrelated)
    contexts = make_contexts(terms, controls)
    if metadata.get("contexts") != contexts:
        raise ValueError("Run contexts do not match supplied fixed glossaries")
    rows = load_manifest(manifest, check_audio=False)
    if set(metadata.get("audio_sha256", {})) != {row["id"] for row in rows}:
        raise ValueError("Audio hash IDs do not match manifest IDs")
    for row in rows:
        if file_hash(Path(row["audio_path"])) != metadata["audio_sha256"][row["id"]]:
            raise ValueError(f"Audio hash mismatch: {row['id']}")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    index = prediction_index(predictions, rows)
    cases = []
    for row in rows:
        case = {"id": row["id"], "group": row["group"], "split": row["split"],
                "reference": row["reference"], "conditions": {}, "comparisons": {}}
        durations = []
        for condition in CONDITIONS:
            prediction = index[row["id"], condition]
            seconds, duration = prediction.get("seconds"), prediction.get("duration_seconds")
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0
                    or isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Invalid timing/duration: {row['id']}/{condition}")
            durations.append(duration)
            detail = analyze_text(row["reference"], prediction["text"], terms)
            detail.update(text=prediction["text"], seconds=seconds,
                          all_candidates=analyze_text(row["reference"], prediction["text"], terms + controls))
            case["conditions"][condition] = detail
        if not all(math.isclose(durations[0], duration, rel_tol=1e-12) for duration in durations):
            raise ValueError(f"Conditions do not share audio duration: {row['id']}")
        case["duration_seconds"] = durations[0]
        for condition in ("personal", "unrelated"):
            baseline, current = case["conditions"]["none"], case["conditions"][condition]
            comparison = compare_mentions(baseline, current)
            comparison["all_candidates"] = compare_mentions(baseline["all_candidates"], current["all_candidates"])
            comparison.update(edit_delta=sum(current["score"][f] - baseline["score"][f] for f in ("S", "D", "I")),
                              raw_text_changed=baseline["text"] != current["text"],
                              normalized_text_changed=normalize_tokens(baseline["text"]) != normalize_tokens(current["text"]))
            case["comparisons"][condition + "_vs_none"] = comparison
        cases.append(case)
    covered = lambda case: bool(case["conditions"]["none"]["score"]["term_ref_mentions"])
    cohorts = {"all": cases, "coverage:covered": [c for c in cases if covered(c)],
               "coverage:uncovered": [c for c in cases if not covered(c)]}
    for group in sorted({case["group"] for case in cases}):
        subset = [case for case in cases if case["group"] == group]
        cohorts["group:" + group] = subset
        for name, value in (("covered", True), ("uncovered", False)):
            cohorts[f"group:{group}/coverage:{name}"] = [case for case in subset if covered(case) == value]
    summary, paired, coverage = {}, {}, {}
    for name, subset in cohorts.items():
        summary[name] = {condition: _summary(subset, condition) for condition in CONDITIONS}
        coverage[name] = {"clips": len(subset), "positive_clips": sum(covered(c) for c in subset),
                          "negative_clips": sum(not covered(c) for c in subset),
                          "reference_mentions": summary[name]["none"]["term_ref_mentions"]}
        seed = int.from_bytes(hashlib.sha256(f"{bootstrap_seed}:{name}".encode()).digest()[:8], "big")
        paired[name] = {condition + "_vs_none": _paired(subset, condition, replicates=bootstrap_replicates, seed=seed)
                        for condition in ("personal", "unrelated")}
    per_term = {}
    for term in terms:
        normalized = " ".join(normalize_tokens(term))
        per_term[term] = {"normalized": normalized, "conditions": {}}
        for condition in CONDITIONS:
            refs = [m for c in cases for m in c["conditions"][condition]["reference_mentions"] if m["term"] == normalized]
            false = [m for c in cases for m in c["conditions"][condition]["false_mentions"] if m["term"] == normalized]
            hits = sum(m["hit"] for m in refs)
            per_term[term]["conditions"][condition] = {"reference_mentions": len(refs), "correct_mentions": hits,
                                                      "false_mentions": len(false), "recall": _rate(hits, len(refs)),
                                                      "precision": _rate(hits, hits + len(false))}
    coverage["vocabulary"] = {"entries": len(terms),
                               "entries_with_reference_occurrence": sum(v["conditions"]["none"]["reference_mentions"] > 0 for v in per_term.values())}
    stored_verified = None
    if (run_dir / "scores.json").exists():
        stored = _read_json(run_dir / "scores.json")["slices"]
        required_slices = {"all"} | {"group:" + case["group"] for case in cases}
        if not isinstance(stored, dict) or not required_slices.issubset(stored):
            raise ValueError("Saved scores are missing required all/language slices")
        for name, arms in summary.items():
            if name not in stored:
                continue
            for condition, fresh in arms.items():
                for field in (*COUNT_FIELDS, "mer", "term_recall", "term_precision", "all_candidates",
                              "clips", "no_term_clips", "no_term_false_mentions", "no_term_false_clips",
                              "no_term_false_mentions_per_100_clips", "no_term_false_clip_rate"):
                    if fresh[field] != stored[name][condition][field]:
                        raise ValueError(f"Saved score mismatch: {name}/{condition}/{field}")
                for field in ("warm_p50_seconds", "warm_p95_seconds", "rtf"):
                    if field in stored[name][condition]:
                        before, after = fresh[field], stored[name][condition][field]
                        if before is None or after is None:
                            equal = before is None and after is None
                        else:
                            equal = (not isinstance(after, bool) and isinstance(after, (int, float)) and math.isfinite(after)
                                     and math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-12))
                        if not equal:
                            raise ValueError(f"Saved timing mismatch: {name}/{condition}/{field}")
        stored_verified = True
    repo = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), repo / "src/speaker_lab/glossary_metrics.py",
               repo / "src/speaker_lab/glossary_experiment.py", repo / "scripts/select_glossary_pilot.py",
               repo / "scripts/build_ascend_train_glossary.py", repo / "scripts/export_ascend_train_text.py"]
    result = {"schema_version": 2, "run": str(run_dir),
              "condition_aliases": {"personal": "correct fixed glossary (train-derived in this experiment)"},
              "normalization": "nfkc_han_latin_digits_v1", "model": metadata.get("config"),
              "versions": metadata.get("versions"),
              "analysis_parameters": {"bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": bootstrap_seed},
              "independent_verification": {"complete_paired_predictions": True, "frozen_input_audio_hashes_verified": True,
                                           "alignment_and_mentions_match_primary_scorer": True, "saved_scores_verified": stored_verified},
              "source_sha256_at_analysis": {str(path.relative_to(repo)): file_hash(path) for path in sources if path.exists()},
              "artifact_sha256": {name: file_hash(run_dir / name) for name in ("run.json", "predictions.jsonl", "scores.json") if (run_dir / name).exists()},
              "input_sha256": {"manifest": file_hash(manifest), "correct": file_hash(correct), "unrelated": file_hash(unrelated)},
              "context_sizes": {c: {"items": len(terms if c == "personal" else controls) if c != "none" else 0,
                                    "normalized_tokens": len(normalize_tokens(contexts[c])), "characters": len(contexts[c])} for c in CONDITIONS},
              "execution": {"timed_inferences": len(predictions), "unscored_warmup_calls": 1,
                            "unique_audio_seconds": sum(c["duration_seconds"] for c in cases),
                            "total_timed_inference_seconds": sum(p["seconds"] for p in predictions),
                            "model_load_seconds": metadata.get("model_load_seconds"), "warmup_seconds": metadata.get("warmup_seconds")},
              "coverage": coverage, "summary": summary, "paired": paired, "per_term": per_term, "clips": cases,
              "limitations": ["Literal references retain existing annotations; no post-hoc normalization changes",
                              "Alignment tie ambiguity applies to repeated/relocated words",
                              "False-mention gained/lost identities use aligned reference slots and insertion boundaries",
                              "Clip bootstrap does not model speaker/session correlation or repeated-run timing variability"]}
    selection_path = manifest.parent / "selection.json"
    if selection_path.exists():
        result["selection_metadata"] = _read_json(selection_path)
        result["selection_sha256"] = file_hash(selection_path)
    if old_run_dir is not None:
        result["old_baseline_consistency"] = _old_baseline(Path(old_run_dir).resolve(), cases, terms, controls, metadata)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--correct", required=True, type=Path)
    parser.add_argument("--unrelated", required=True, type=Path)
    parser.add_argument("--old-run-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path, help="New output; defaults to RUN_DIR/comparison.json")
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "comparison.json"
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    try:
        result = analyze_run(args.run_dir, args.manifest, args.correct, args.unrelated,
                             old_run_dir=args.old_run_dir, bootstrap_replicates=args.bootstrap_replicates,
                             bootstrap_seed=args.bootstrap_seed)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
