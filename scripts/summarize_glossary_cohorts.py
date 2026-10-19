#!/usr/bin/env python3
"""Verify two disjoint completed glossary cohorts and publish aggregate JSON.

This is a reporting adapter, not a new scorer. It invokes the unchanged paired
analyzer, including its recorded bootstrap parameters, and requires the saved
comparison to agree. Audio, references and predictions remain local. Pooled
results sum occurrence/edit counts and inference times; they are descriptive
because they include the development cohort. No pooled bootstrap is computed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
ANALYZER_PATH = REPO / "scripts/analyze_glossary_comparison.py"
SPEC = importlib.util.spec_from_file_location("glossary_cohort_analyzer", ANALYZER_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

NORMALIZATION = "nfkc_han_latin_digits_v1"
CORE_SOURCES = {
    "src/speaker_lab/glossary_metrics.py",
    "src/speaker_lab/glossary_experiment.py",
    "scripts/analyze_glossary_comparison.py",
}
MODEL_FIELDS = ("repo_id", "revision", "device", "dtype", "language", "max_new_tokens",
                "torch_threads", "seed", "do_sample", "num_beams")
VERIFIED_FIELDS = ("normalization", "analysis_parameters", "independent_verification",
                   "artifact_sha256", "input_sha256", "coverage",
                   "summary", "paired", "per_term", "clips")


def read_json(path: Path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_file(root: Path, name: str, digest: str) -> Path:
    """Only accept portable relative hash keys within their declared root."""
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "Hash key must be a portable relative path")
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()),
            "Hash path escapes its declared root")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"Invalid SHA256 for {name}")
    require(path.is_file() and analysis.file_hash(path) == digest,
            f"Hash mismatch: {name}")
    return path


def load_protocol(directory: Path) -> tuple[dict, dict]:
    protocol = read_json(directory / "protocol.json")
    require(isinstance(protocol, dict), "Protocol must be a JSON object")
    require(protocol.get("normalization") == NORMALIZATION, "Protocol normalization mismatch")
    require(protocol.get("dataset", {}).get("evaluation_split") == "validation",
            "Only validation cohorts are supported")
    portable = protocol.get("portable_files_sha256", {})
    require(isinstance(portable, dict)
            and {"correct.json", "unrelated.json", "selected_ids.json", "analysis_plan.json"}.issubset(portable),
            "Protocol is missing fixed glossary/ID hashes")
    for name, digest in portable.items():
        checked_file(directory, name, digest)
    hashes = {name: analysis.file_hash(directory / name)
              for name in ("protocol.json", *portable)}
    freeze = directory / "FREEZE.json"
    if freeze.exists():
        frozen = read_json(freeze).get("sha256", {})
        require("protocol.json" in frozen and set(portable).issubset(frozen),
                "FREEZE is missing required protocol hashes")
        for name, digest in frozen.items():
            checked_file(directory, name, digest)
            hashes[name] = digest
        hashes["FREEZE.json"] = analysis.file_hash(freeze)
    return protocol, hashes


def verify_sources(protocol: dict) -> dict:
    sources = protocol.get("source_sha256", {})
    require(CORE_SOURCES.issubset(sources), "New protocol lacks frozen scorer/analyzer sources")
    for name, digest in sources.items():
        checked_file(REPO, name, digest)
    return dict(sources)


def verify_cohort(run_dir: Path, manifest: Path, protocol_dir: Path,
                  protocol: dict, metadata: dict) -> tuple[dict, dict]:
    for name in ("predictions.jsonl", "scores.json", "comparison.json"):
        require((run_dir / name).is_file(), f"Missing completed-run artifact: {name}")
    # Strict JSON decoding happens before calling the legacy analyzer.
    stored_scores = read_json(run_dir / "scores.json")
    cached = read_json(run_dir / "comparison.json")
    require(isinstance(stored_scores, dict) and isinstance(cached, dict),
            "Completed-run artifacts must be JSON objects")
    parameters = cached.get("analysis_parameters", {})
    replicates, seed = parameters.get("bootstrap_replicates"), parameters.get("bootstrap_seed")
    require(type(replicates) is int and replicates >= 0 and type(seed) is int,
            "Invalid recorded bootstrap parameters")
    uncertainty = read_json(protocol_dir / "analysis_plan.json").get("uncertainty")
    if isinstance(uncertainty, dict):
        require(replicates == uncertainty.get("replicates") and seed == uncertainty.get("seed")
                and uncertainty.get("confidence") == .95, "Analysis differs from frozen bootstrap plan")
    require(metadata.get("normalization") == NORMALIZATION, "Run normalization mismatch")
    expected_manifest = protocol.get("input_sha256", {}).get("runtime_manifest")
    if expected_manifest is None:
        expected_manifest = protocol.get("reference_input_sha256", {}).get("manifest")
    require(expected_manifest == analysis.file_hash(manifest), "Protocol manifest hash mismatch")
    for field in ("revision", "device", "dtype", "language", "max_new_tokens", "torch_threads"):
        require(metadata.get("config", {}).get(field) == protocol.get("model", {}).get(field),
                f"Run model setting differs from protocol: {field}")
    require(metadata.get("seed") == protocol.get("model", {}).get("seed"), "Run seed mismatch")
    for field in ("do_sample", "num_beams"):
        require(metadata.get("generation_config", {}).get(field) == protocol.get("model", {}).get(field),
                f"Run decoding differs from protocol: {field}")
    if (protocol_dir / "config.toml").exists():
        require(metadata.get("config_sha256") == analysis.file_hash(protocol_dir / "config.toml"),
                "Frozen configuration hash mismatch")
    fresh = analysis.analyze_run(
        run_dir, manifest, protocol_dir / "correct.json", protocol_dir / "unrelated.json",
        bootstrap_replicates=replicates, bootstrap_seed=seed,
    )
    require(fresh["independent_verification"]["saved_scores_verified"] is True,
            "Saved primary scores were not verified")
    for field in VERIFIED_FIELDS:
        require(cached.get(field) == fresh[field], f"Saved comparison mismatch: {field}")
    # Historical reports may append a prose timing caveat; verify every numeric
    # execution field from the analyzer without copying that unstructured prose.
    require(isinstance(cached.get("execution"), dict), "Missing saved execution statistics")
    for field, value in fresh["execution"].items():
        require(cached["execution"].get(field) == value, f"Saved execution mismatch: {field}")
    comparison_sources = {name.replace("\\", "/"): digest
                          for name, digest in cached.get("source_sha256_at_analysis", {}).items()}
    for name in CORE_SOURCES:
        require(comparison_sources.get(name) == analysis.file_hash(REPO / name),
                f"Comparison source hash mismatch: {name}")
    ids = read_json(protocol_dir / "selected_ids.json")
    require(isinstance(ids, list) and all(isinstance(value, str) for value in ids),
            "Frozen selected IDs must be a string list")
    require(ids == [case["id"] for case in fresh["clips"]], "Manifest differs from frozen ID order")
    require(all(case["split"] == "validation" for case in fresh["clips"]), "Non-validation clip")
    groups = dict(sorted(Counter(case["group"] for case in fresh["clips"]).items()))
    require(all(re.fullmatch(r"[a-z][a-z0-9_-]*", group) for group in groups), "Invalid group label")
    selection = protocol.get("selection", {})
    require(selection.get("count") == len(ids) and selection.get("groups") == groups,
            "Protocol cohort count/language distribution mismatch")
    require(math.isclose(selection.get("total_audio_seconds", -1),
                         fresh["execution"]["unique_audio_seconds"], rel_tol=1e-12, abs_tol=1e-9),
            "Protocol audio duration mismatch")
    limits = (selection.get("min_duration_seconds"), selection.get("max_duration_seconds"))
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) for value in limits), "Invalid protocol duration limits")
    require(all(limits[0] <= case["duration_seconds"] <= limits[1] for case in fresh["clips"]),
            "Clip duration outside frozen eligibility limits")
    artifacts = {name: analysis.file_hash(run_dir / name)
                 for name in ("run.json", "predictions.jsonl", "scores.json", "comparison.json")}
    for optional in ("protocol_audit.json", "analysis_plan.json"):
        if (run_dir / optional).is_file():
            artifacts[optional] = analysis.file_hash(run_dir / optional)
    return fresh, artifacts


def reference_statistics(cases: list[dict]) -> dict:
    return {
        "clips": len(cases),
        "groups": dict(sorted(Counter(case["group"] for case in cases).items())),
        "reference_tokens": sum(case["conditions"]["none"]["score"]["N"] for case in cases),
        "reference_unknown_marker_clips": sum("[unk]" in case["reference"].lower() for case in cases),
        "reference_unknown_markers": sum(case["reference"].lower().count("[unk]") for case in cases),
    }


def per_term_summary(cases: list[dict], terms: list[str]) -> dict:
    result = {}
    for term in terms:
        normalized = " ".join(analysis.normalize_tokens(term))
        arms = {}
        for arm in analysis.CONDITIONS:
            refs = [mention for case in cases for mention in case["conditions"][arm]["reference_mentions"]
                    if mention["term"] == normalized]
            false = sum(mention["term"] == normalized for case in cases
                        for mention in case["conditions"][arm]["false_mentions"])
            hits = sum(mention["hit"] for mention in refs)
            arms[arm] = {"reference_mentions": len(refs), "correct_mentions": hits,
                         "false_mentions": false, "recall": hits / len(refs) if refs else None,
                         "precision": hits / (hits + false) if hits + false else None}
        result[term] = {"normalized": normalized, "conditions": arms}
    return result


def pooled_summary(analyses: list[dict], terms: list[str]) -> dict:
    cases = [case for result in analyses for case in result["clips"]]
    covered = lambda case: bool(case["conditions"]["none"]["score"]["term_ref_mentions"])
    slices = {"all": cases, "coverage:covered": [case for case in cases if covered(case)],
              "coverage:uncovered": [case for case in cases if not covered(case)]}
    for group in sorted({case["group"] for case in cases}):
        subset = [case for case in cases if case["group"] == group]
        slices["group:" + group] = subset
        for name, value in (("covered", True), ("uncovered", False)):
            slices[f"group:{group}/coverage:{name}"] = [case for case in subset if covered(case) == value]
    summary, paired, coverage = {}, {}, {}
    for name, subset in slices.items():
        summary[name] = {arm: analysis._summary(subset, arm) for arm in analysis.CONDITIONS}
        coverage[name] = {"clips": len(subset), "positive_clips": sum(covered(case) for case in subset),
                          "negative_clips": sum(not covered(case) for case in subset),
                          "reference_mentions": summary[name]["none"]["term_ref_mentions"]}
        paired[name] = {}
        for arm in ("personal", "unrelated"):
            values = analysis._paired(subset, arm, replicates=0, seed=0)
            values["bootstrap_ci"] = None
            paired[name][arm + "_vs_none"] = values
    per_term = per_term_summary(cases, terms)
    coverage["vocabulary"] = {
        "entries": len(terms),
        "entries_with_reference_occurrence": sum(item["conditions"]["none"]["reference_mentions"] > 0
                                                  for item in per_term.values()),
    }
    execution = {name: sum(result["execution"][name] for result in analyses)
                 for name in ("timed_inferences", "unscored_warmup_calls", "unique_audio_seconds",
                              "total_timed_inference_seconds")}
    return {"role": "Supplemental descriptive aggregate including development; not independent validation",
            "reference_statistics": reference_statistics(cases), "coverage": coverage,
            "summary": summary, "paired": paired, "per_term": per_term, "execution": execution,
            "bootstrap": "No pooled bootstrap; cohort intervals remain separate"}


def summarize_cohorts(old_run_dir: Path, old_manifest: Path, old_protocol_dir: Path,
                       new_run_dir: Path, new_manifest: Path, new_protocol_dir: Path) -> dict:
    directories = [Path(old_run_dir).resolve(), Path(new_run_dir).resolve()]
    manifests = [Path(old_manifest).resolve(), Path(new_manifest).resolve()]
    protocols = [Path(old_protocol_dir).resolve(), Path(new_protocol_dir).resolve()]
    # No reference or prediction is read until BOTH completion gates pass.
    metadata = [read_json(directory / "run.json") for directory in directories]
    require(all(isinstance(item, dict) and item.get("status") == "complete" for item in metadata),
            "Refusing incomplete cohort; both run statuses must be complete")
    require((protocols[1] / "FREEZE.json").is_file(), "New protocol must have a FREEZE hash record")
    loaded = [load_protocol(directory) for directory in protocols]
    specs = [item[0] for item in loaded]
    source_hashes = verify_sources(specs[1])
    for field in MODEL_FIELDS:
        require(specs[0].get("model", {}).get(field) == specs[1].get("model", {}).get(field),
                f"Cohort protocol model differs: {field}")
    for field in ("repo_id", "revision", "evaluation_split", "glossary_split"):
        require(specs[0].get("dataset", {}).get(field) == specs[1].get("dataset", {}).get(field),
                f"Cohort dataset differs: {field}")
    for name in ("correct.json", "unrelated.json"):
        require(analysis.file_hash(protocols[0] / name) == analysis.file_hash(protocols[1] / name),
                f"Cohort fixed vocabulary differs: {name}")
    for field in ("config", "config_sha256", "versions", "generation_config", "seed", "offline"):
        require(field in metadata[0] and metadata[0].get(field) == metadata[1].get(field),
                f"Cohort runtime provenance differs: {field}")
    ids = [set(item.get("audio_sha256", {})) for item in metadata]
    require(not ids[0] & ids[1], "Cohort ID intersection is forbidden")
    audio = [set(item.get("audio_sha256", {}).values()) for item in metadata]
    require(not audio[0] & audio[1], "Cohort audio-hash intersection is forbidden")
    results = [verify_cohort(*args) for args in zip(directories, manifests, protocols, specs, metadata)]
    require(results[0][0]["analysis_parameters"] == results[1][0]["analysis_parameters"],
            "Cohort bootstrap parameters differ")
    cohorts, provenance = {}, {}
    for name, role, (fresh, artifacts), (_, hashes), meta in zip(
            ("development", "remaining"),
            ("Previously inspected development cohort", "All remaining eligible validation clips"),
            results, loaded, metadata):
        cohorts[name] = {"role": role, "reference_statistics": reference_statistics(fresh["clips"]),
                         **{key: fresh[key] for key in ("coverage", "summary", "paired", "per_term", "execution")}}
        provenance[name] = {"protocol_files_sha256": hashes, "artifact_sha256": artifacts,
                            "input_sha256": fresh["input_sha256"], "config_sha256": meta["config_sha256"]}
    terms = analysis.load_terms(protocols[0] / "correct.json")
    cohorts["pooled"] = pooled_summary([result[0] for result in results], terms)
    selection = specs[1].get("selection", {})
    if "eligible_count" in selection:
        require(selection["eligible_count"] == cohorts["pooled"]["reference_statistics"]["clips"],
                "Pooled count differs from frozen eligible population")
    if "prior_excluded_count" in selection:
        require(selection["prior_excluded_count"] == cohorts["development"]["reference_statistics"]["clips"],
                "Development count differs from frozen exclusion count")
    return {
        "schema_version": 1, "normalization": NORMALIZATION,
        "condition_aliases": {"none": "empty context", "personal": "unchanged correct fixed glossary",
                              "unrelated": "unchanged unrelated fixed glossary"},
        "model": {field: specs[1]["model"][field] for field in MODEL_FIELDS},
        "versions": metadata[1]["versions"],
        "verification": {"both_runs_complete": True, "complete_unique_paired_predictions": True,
                         "cohorts_disjoint_by_id_and_audio_hash": True, "frozen_inputs_and_sources_verified": True,
                         "saved_scores_comparisons_and_ci_independently_recomputed": True,
                         "contains_raw_transcripts_or_audio_paths": False},
        "provenance": {"summary_tool_sha256": analysis.file_hash(Path(__file__)),
                       "frozen_source_sha256": source_hashes, "cohorts": provenance},
        "cohorts": cohorts,
        "limitations": [
            "Old development and new remaining cohorts have different language mixtures; absolute cross-cohort MER is not a treatment effect.",
            "Pooled rates use summed counts, never averages of cohort rates; development is included and adds no independent replication.",
            "Cohort intervals are the verified saved paired language-stratified clip bootstrap; they do not model speaker/session dependence.",
            "CPU inference timings were not collected under isolated system load and are not Ubuntu/GPU performance estimates.",
            "No vocabulary, references, scoring, decoding, audio, or language settings are changed by this reporting adapter.",
            "Strict term false mentions are alignment diagnostics, not adjudicated acoustic hallucinations.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in ("old", "new"):
        for field in ("run-dir", "manifest", "protocol-dir"):
            parser.add_argument(f"--{cohort}-{field}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing summary")
    try:
        result = summarize_cohorts(args.old_run_dir, args.old_manifest, args.old_protocol_dir,
                                   args.new_run_dir, args.new_manifest, args.new_protocol_dir)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Verified aggregate summary saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
