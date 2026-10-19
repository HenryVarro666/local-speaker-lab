"""Freeze a language-stratified ASCEND pilot without filtering on reference terms."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

DEFAULT_SEED = 20261018
DEFAULT_COUNTS = {"mixed": 12, "mandarin": 6, "english": 6}
GROUP_ORDER = ("mixed", "mandarin", "english")


def positive_int(value):
    """argparse count type: accept positive integers only."""
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("count must be a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return count


def select_rows(rows, counts=None, seed=DEFAULT_SEED):
    """Select fixed hash-order prefixes using only group, duration_seconds, id.

    Increasing per-group counts extends the same prefixes. Reference text,
    vocabulary, and recognition output have no role in this function.
    """
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(GROUP_ORDER):
        raise ValueError("counts must specify exactly mixed, mandarin, english")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0
           for count in counts.values()):
        raise ValueError("every group count must be a positive integer")
    eligible = [row for row in rows if 1 <= row["duration_seconds"] <= 12]
    chosen = []
    for group in GROUP_ORDER:
        count = counts[group]
        pool = [row for row in eligible if row["group"] == group]
        pool.sort(key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]))
        if len(pool) < count:
            raise ValueError(f"Not enough eligible {group} clips: {len(pool)} < {count}")
        chosen.extend(pool[:count])
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--personal", type=Path, required=True)
    parser.add_argument("--unrelated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mixed-count", type=positive_int, default=DEFAULT_COUNTS["mixed"])
    parser.add_argument("--mandarin-count", type=positive_int, default=DEFAULT_COUNTS["mandarin"])
    parser.add_argument("--english-count", type=positive_int, default=DEFAULT_COUNTS["english"])
    parser.add_argument("--vocabulary-scope", default="Previously frozen technical examples; not mined from ASCEND references")
    args = parser.parse_args(argv)
    # Keep the sampling helper independently testable without importing loaders
    # or opening any transcript/vocabulary files.
    from speaker_lab.glossary_experiment import file_hash, load_manifest, load_terms, make_contexts

    source = args.manifest.resolve()
    rows = load_manifest(source, check_audio=True)
    contexts = make_contexts(load_terms(args.personal), load_terms(args.unrelated))
    counts = {"mixed": args.mixed_count, "mandarin": args.mandarin_count, "english": args.english_count}
    chosen = select_rows(rows, counts, seed=args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.personal, output / "personal.json")
    shutil.copyfile(args.unrelated, output / "unrelated.json")
    with (output / "manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in chosen:
            item = {key: row[key] for key in ("id", "reference", "split", "group")}
            item["audio"] = os.path.relpath(row["audio_path"], output).replace("\\", "/")
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = {
        "source_manifest": str(source), "source_manifest_sha256": file_hash(source),
        "seed": args.seed, "min_duration_seconds": 1, "max_duration_seconds": 12,
        "requested_group_counts": counts,
        "sampling_source_code": str(Path(__file__).resolve()),
        "sampling_source_code_sha256": file_hash(Path(__file__).resolve()),
        "selection": "per-language sha256(seed:id), no lexical or model-error filtering",
        "count": len(chosen), "groups": dict(Counter(r["group"] for r in chosen)),
        "total_audio_seconds": sum(r["duration_seconds"] for r in chosen),
        "ids": [r["id"] for r in chosen],
        "personal_sha256": file_hash(output / "personal.json"),
        "unrelated_sha256": file_hash(output / "unrelated.json"),
        "manifest_sha256": file_hash(output / "manifest.jsonl"),
        "contexts": contexts,
        "vocabulary_scope": args.vocabulary_scope,
    }
    (output / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("count", "groups", "total_audio_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
