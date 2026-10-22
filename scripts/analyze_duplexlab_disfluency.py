"""Audit prior coverage regressions without new inference or reference edits."""

import json
from collections import Counter
from pathlib import Path
from speaker_lab.duplex.coverage import has_filler
from speaker_lab.duplex.disfluency import alignment_edits, text_features
from speaker_lab.duplex.protocol import file_hash, write_json
from speaker_lab.glossary_metrics import score_pair

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT / "runs/duplexlab/coverage-v1"
    data = ROOT / "data/duplexlab/coverage-v1"
    out = ROOT / "runs/duplexlab/disfluency-v1"
    out.mkdir(parents=True, exist_ok=True)
    public = ROOT / "experiments/duplexlab/reports/2026-10-20-disfluency"
    public.mkdir(parents=True, exist_ok=True)
    meta = {
        r["clip_id"]: r
        for r in map(json.loads, (data / "test.jsonl").read_text().splitlines())
    }
    groups = {}
    details = {
        cid: {
            "clip_id": cid,
            "features": text_features(r["reference"]),
            "reference": r["reference"],
            "audio_sha256": r["audio_sha256"],
            "predictions": {},
        }
        for cid, r in meta.items()
    }
    for seed in [42, 43, 44]:
        groups[str(seed)] = {}
        for arm in ["random", "filtered"]:
            records = json.loads((old / f"test-{arm}-{seed}/summary.json").read_text())[
                "per_clip"
            ]
            a = Counter()
            kinds = Counter()
            tokens = Counter()
            for r in records:
                row = meta[r["clip_id"]]
                edits = alignment_edits(row["reference"], r["hypothesis"])
                assert all(edits[k] == r[k] for k in ["S", "D", "I", "N"])
                for t in edits["deleted"]:
                    token = t["token"]
                    cat = (
                        "filler"
                        if token
                        in set("嗯呃啊额唔诶欸")
                        | {"uh", "um", "hmm", "erm", "er", "ah", "eh"}
                        else "single_ascii_letter"
                        if len(token) == 1 and token.isascii() and token.isalpha()
                        else "other"
                    )
                    kinds[cat] += 1
                    tokens[token] += 1
                a.update(
                    {k: edits[k] for k in ["S", "D", "I", "N", "deleted_filler_count"]}
                )
                if has_filler(row["reference"]):
                    a.update(
                        {f"filler_group_{k}": edits[k] for k in ["S", "D", "I", "N"]}
                    )
                assert file_hash(data / row["audio_relpath"]) == row["audio_sha256"]
                details[r["clip_id"]]["predictions"][f"{arm}-{seed}"] = {
                    "hypothesis": r["hypothesis"],
                    **edits,
                }
            groups[str(seed)][arm] = {
                **a,
                "deleted_token_categories": dict(kinds),
                "deleted_tokens": dict(tokens),
            }
    consistent = []
    for cid, info in details.items():
        delta = [
            sum(
                info["predictions"][f"filtered-{s}"][k]
                - info["predictions"][f"random-{s}"][k]
                for k in ["S", "D", "I"]
            )
            for s in [42, 43, 44]
        ]
        if all(x > 0 for x in delta):
            consistent.append(
                {
                    "clip_id": cid,
                    "delta_errors": delta,
                    "delta_deletions": [
                        info["predictions"][f"filtered-{s}"]["D"]
                        - info["predictions"][f"random-{s}"]["D"]
                        for s in [42, 43, 44]
                    ],
                    "features": info["features"],
                }
            )
    consistent.sort(key=lambda x: (-sum(x["delta_errors"]), x["clip_id"]))
    assert len(consistent) == 8
    acronym = {}
    cid = "ascend-train-05012"
    for s in [42, 43, 44]:
        acronym[str(s)] = {}
        for arm in ["random", "filtered"]:
            info = details[cid]
            count = score_pair(
                info["reference"].replace("U S T C", "USTC"),
                info["predictions"][f"{arm}-{s}"]["hypothesis"].replace(
                    "U S T C", "USTC"
                ),
                terms=[],
            )
            acronym[str(s)][arm] = {k: count[k] for k in ["S", "D", "I", "N", "mer"]}
    write_json(
        out / "PRIOR_REGRESSIONS_PRIVATE.json",
        {
            "groups": groups,
            "clips": details,
            "consistent": consistent,
            "human_listening": "pending",
        },
    )
    # Full deleted token lists are kept private; public groups are counts only.
    public_groups = {
        s: {
            a: {k: v for k, v in x.items() if k != "deleted_tokens"}
            for a, x in arms.items()
        }
        for s, arms in groups.items()
    }
    result = {
        "source_study": "coverage-v1",
        "source_comparison_sha256": file_hash(old / "COMPARISON-test.json"),
        "seeds": public_groups,
        "consistent_regressions": consistent,
        "USTC_alias_diagnostic_only": acronym,
        "all_1800_primary_alignment_counts_match": True,
        "human_listening": "pending",
        "primary_references_or_scores_changed": False,
        "alignment_ties": "diagonal, then deletion, then insertion; ambiguous repeated words are not time-aligned acoustic proof",
    }
    write_json(public / "PRIOR_ANALYSIS.json", result)
    print(
        json.dumps(
            {"groups": public_groups, "consistent": consistent, "alias": acronym},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
