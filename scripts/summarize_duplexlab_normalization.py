"""Aggregate strict scores and lexical deletion diagnostics from completed runs."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from speaker_lab.duplex.disfluency import alignment_edits, text_features
from speaker_lab.duplex.protocol import write_json, read_json
from speaker_lab.duplex.deletion import optimal_operations
from speaker_lab.glossary_metrics import normalize_tokens
from speaker_lab.duplex.multiseed import compare_seeds
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT / "runs/duplexlab/normalization-v1"
    data = ROOT / "data/duplexlab/normalization-v1"
    result = {}
    for role in ["dev", "test"]:
        meta = {
            r["clip_id"]: r
            for r in map(json.loads, (data / f"{role}.jsonl").read_text().splitlines())
        }
        result[role] = {}
        for seed in [42, 43, 44]:
            result[role][str(seed)] = {}
            for arm in ["random", "selected", "uniform", "filtered"]:
                scores = json.loads(
                    (runs / f"{role}-{arm}-{seed}/summary.json").read_text()
                )["per_clip"]
                bins = defaultdict(Counter)
                for row in scores:
                    ref = meta[row["clip_id"]]["reference"]
                    f = text_features(ref)
                    e = alignment_edits(ref, row["hypothesis"])
                    assert all(e[k] == row[k] for k in ["S", "D", "I", "N"])
                    count = {
                        k: e[k] for k in ["S", "D", "I", "N", "deleted_filler_count"]
                    }
                    count.update(clips=1, ref_filler_count=f["filler_count"])
                    single = sum(
                        len(x["token"]) == 1
                        and x["token"].isascii()
                        and x["token"].isalpha()
                        for x in e["deleted"]
                    )
                    count.update(
                        deleted_single_ascii_letter_count=single,
                        deleted_other_count=e["D"] - e["deleted_filler_count"] - single,
                    )

                    def b(n):
                        return str(n) if n < 2 else "2+"

                    sw = f["switch_count"]
                    en = f["max_en_span"]
                    groups = [
                        "overall",
                        f"frequency|{b(f['filler_count'])}",
                        f"repetition|{b(f['repeat_count'])}",
                        f"switches|{sw if sw < 2 else '2-3' if sw < 4 else '4+'}",
                        f"en_span|{'0' if en == 0 else '1-2' if en < 3 else '3+'}",
                    ]
                    groups += [
                        f"language|{meta[row['clip_id']]['language_group']}",
                        f"dev_panel|{meta[row['clip_id']].get('dev_panel', 'test')}",
                        "filler_present" if f["filler_count"] else "filler_absent",
                        "source_unk" if "[unk]" in ref.lower() else "source_no_unk",
                    ]
                    for g in groups:
                        bins[g].update(count)
                values = {}
                for k, c in bins.items():
                    v = dict(c)
                    v["mer"] = (c["S"] + c["D"] + c["I"]) / c["N"] if c["N"] else None
                    v["filler_deletion_rate"] = (
                        c["deleted_filler_count"] / c["ref_filler_count"]
                        if c["ref_filler_count"]
                        else None
                    )
                    values[k] = v
                result[role][str(seed)][arm] = values
    deletion_comparisons = {}
    stable = {}
    private = []
    for role in ["dev", "test"]:
        metadata = list(
            map(json.loads, (data / f"{role}.jsonl").read_text().splitlines())
        )
        loaded = {
            s: {
                a: read_json(runs / f"{role}-{a}-{s}/summary.json")
                for a in ["random", "selected", "uniform", "filtered"]
            }
            for s in [42, 43, 44]
        }
        deletion_comparisons[role] = {}
        for contrast, (left, right) in {
            "joint": ("random", "filtered"),
            "selection": ("random", "selected"),
            "normalization": ("selected", "uniform"),
            "weights": ("uniform", "filtered"),
        }.items():
            paired = {}
            for seed, pair in loaded.items():
                paired[seed] = {}
                for name, arm in [("random", left), ("filtered", right)]:
                    summary = deepcopy(pair[arm])
                    for row in summary["per_clip"]:
                        row["S"] = 0
                        row["I"] = 0
                        row["mer"] = row["D"] / row["N"]
                    paired[seed][name] = summary
            diagnostic = compare_seeds(paired, metadata)
            deletion_comparisons[role][contrast] = dict(
                metric="D/N only; S and I intentionally excluded for this secondary diagnostic",
                mean_deletion_rate=diagnostic["mean_mer"],
                mean_delta=diagnostic["mean_delta"],
                hierarchical_delta_interval=diagnostic["hierarchical_delta_interval"],
                sd_delta=diagnostic["sd_delta"],
                per_seed={
                    s: {
                        "rates": {
                            a: r["arms"][a]["mer"] for a in ["random", "filtered"]
                        },
                        "D": {a: r["arms"][a]["D"] for a in ["random", "filtered"]},
                        "N": r["arms"]["random"]["N"],
                        "delta": r["delta"],
                    }
                    for s, r in diagnostic["per_seed"].items()
                },
                interval_unit=diagnostic["interval_unit"],
                bootstrap_seed=diagnostic["bootstrap_seed"],
                bootstrap_replicates=diagnostic["bootstrap_replicates"],
            )
        stable[role] = {}
        for arm in ["random", "selected", "uniform", "filtered"]:
            sets = {
                s: {
                    row["clip_id"]: {
                        v["index"]
                        for v in alignment_edits(row["reference"], row["hypothesis"])[
                            "deleted"
                        ]
                    }
                    for row in loaded[s][arm]["per_clip"]
                }
                for s in [42, 43, 44]
            }
            stable[role][arm] = sum(
                len(set.intersection(*(sets[s][cid] for s in [42, 43, 44])))
                for cid in sets[42]
            )
        if role == "test":
            indexed = {
                s: {
                    a: {r["clip_id"]: r for r in loaded[s][a]["per_clip"]}
                    for a in loaded[s]
                }
                for s in loaded
            }
            for meta in metadata:
                cid = meta["clip_id"]
                ref = meta["reference"]
                per = {
                    a: {
                        s: alignment_edits(ref, indexed[s][a][cid]["hypothesis"])
                        for s in [42, 43, 44]
                    }
                    for a in ["random", "selected", "uniform", "filtered"]
                }
                private.append(
                    dict(
                        clip_id=cid,
                        reference=ref,
                        hypotheses={
                            a: {
                                s: indexed[s][a][cid]["hypothesis"]
                                for s in [42, 43, 44]
                            }
                            for a in per
                        },
                        alignments=per,
                    )
                )
    extra_cases = []
    for row in private:
        deleted = {
            a: {
                seed: {x["index"] for x in row["alignments"][a][seed]["deleted"]}
                for seed in [42, 43, 44]
            }
            for a in ["uniform", "filtered"]
        }
        extra = set.intersection(*deleted["filtered"].values()) - set.intersection(
            *deleted["uniform"].values()
        )
        if extra:
            extra_cases.append(
                dict(
                    clip_id=row["clip_id"],
                    tokens=[
                        dict(
                            index=i,
                            token=normalize_tokens(row["reference"])[i],
                            uniform_deleted_seeds=[
                                seed
                                for seed in [42, 43, 44]
                                if i in deleted["uniform"][seed]
                            ],
                            uniform_matched_all_paths_all_seeds=all(
                                optimal_operations(
                                    row["reference"], row["hypotheses"]["uniform"][seed]
                                )[i]
                                == {"M"}
                                for seed in [42, 43, 44]
                            ),
                            forced_all_paths_all_seeds=all(
                                optimal_operations(
                                    row["reference"],
                                    row["hypotheses"]["filtered"][seed],
                                )[i]
                                == {"D"}
                                for seed in [42, 43, 44]
                            ),
                        )
                        for i in sorted(extra)
                    ],
                    delta_D=[
                        row["alignments"]["filtered"][seed]["D"]
                        - row["alignments"]["uniform"][seed]["D"]
                        for seed in [42, 43, 44]
                    ],
                    delta_errors=[
                        sum(
                            row["alignments"]["filtered"][seed][k]
                            - row["alignments"]["uniform"][seed][k]
                            for k in ["S", "D", "I"]
                        )
                        for seed in [42, 43, 44]
                    ],
                )
            )
    write_json(
        runs / "NEW_STABLE_EXTRA.json",
        dict(
            status="posttest_diagnostic_no_training_changes",
            contrast="weighted minus normalized uniform CE",
            cases=extra_cases,
            clips=len(extra_cases),
            positions=sum(len(r["tokens"]) for r in extra_cases),
            human_listening="pending",
        ),
    )
    consistent = {}
    for name, (left, right) in {
        "joint": ("random", "filtered"),
        "selection": ("random", "selected"),
        "normalization": ("selected", "uniform"),
        "weights": ("uniform", "filtered"),
    }.items():
        cases = []
        for row in private:
            dd = [
                row["alignments"][right][seed]["D"] - row["alignments"][left][seed]["D"]
                for seed in [42, 43, 44]
            ]
            de = [
                sum(
                    row["alignments"][right][seed][k] - row["alignments"][left][seed][k]
                    for k in ["S", "D", "I"]
                )
                for seed in [42, 43, 44]
            ]
            if all(v > 0 for v in dd) or all(v > 0 for v in de):
                cases.append(
                    dict(
                        clip_id=row["clip_id"],
                        delta_D=dd,
                        delta_errors=de,
                        more_D_all_seeds=all(v > 0 for v in dd),
                        more_errors_all_seeds=all(v > 0 for v in de),
                    )
                )
        consistent[name] = dict(
            more_D_all_seeds_clips=sum(c["more_D_all_seeds"] for c in cases),
            more_errors_all_seeds_clips=sum(c["more_errors_all_seeds"] for c in cases),
            cases=cases,
        )
    write_json(
        runs / "CONSISTENT_REGRESSIONS.json",
        dict(
            status="posttest_diagnostic_no_training_changes",
            test_clips=len(private),
            contrasts=consistent,
        ),
    )
    write_json(runs / "NEW_DELETIONS_PRIVATE.json", private)
    write_json(runs / "DELETION_COMPARISONS.json", deletion_comparisons)
    write_json(runs / "STABLE_DELETIONS.json", stable)
    write_json(runs / "DISFLUENCY_GROUPS.json", result)
    print(
        "All24 SFT evaluation batches grouped; strict counts and lexical deletion diagnostics verified"
    )


if __name__ == "__main__":
    main()
