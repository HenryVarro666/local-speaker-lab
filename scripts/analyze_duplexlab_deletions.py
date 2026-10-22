"""Reproduce all prior seed-paired deletions and distinguish alignment uncertainty."""

import json
from collections import Counter
from pathlib import Path
from speaker_lab.duplex.deletion import optimal_operations
from speaker_lab.duplex.disfluency import alignment_edits, FILLERS
from speaker_lab.duplex.protocol import file_hash, write_json
from speaker_lab.glossary_metrics import normalize_tokens, score_pair

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT / "runs/duplexlab/disfluency-v1"
    out = ROOT / "runs/duplexlab/deletion-v1"
    out.mkdir(exist_ok=True)
    scores = {
        a: {
            s: {
                x["clip_id"]: x
                for x in json.loads((old / f"test-{a}-{s}/summary.json").read_text())[
                    "per_clip"
                ]
            }
            for s in (42, 43, 44)
        }
        for a in ("random", "filtered")
    }
    notes = {
        "03611": "head filler candidate; random also deletes in two seeds; listening pending",
        "05996": "large cross-language mistranscription; deletion placement is uncertain",
        "06064": "tail particle candidate; random also deletes in seed44; listening pending",
        "05988": "UNK source plus Leica transliteration/segmentation; not verified lexical loss",
        "08446": "tail particle; random substitutes final particle in all seeds; D increase alone does not mean MER regression",
        "04255": "repeated English short phrase; minimum-edit position ambiguity",
        "06633": "source G math versus GMAT and adjacent word alignment; listening pending",
        "08903": "one fewer repeated character in every candidate seed; identity of deleted copy ambiguous; acoustic truth pending",
        "04662": "function word substitution affects deletion placement; no extra D total",
        "04706": "tail interjection orthography; random also deletes in two seeds; listening pending",
        "07082": "alphago versus alpha go plus Chinese/English spelling; representation diagnostic, not verified swallowed word",
    }
    added = {seed: Counter() for seed in [42, 43, 44]}
    added_kinds, resolved_kinds = Counter(), Counter()
    rows = []
    public = []
    totals = {a: Counter() for a in scores}
    stable_counts = {a: 0 for a in scores}
    for cid in scores["random"][42]:
        ref = scores["random"][42][cid]["reference"]
        tokens = normalize_tokens(ref)
        ops = {
            a: {
                s: alignment_edits(ref, scores[a][s][cid]["hypothesis"])
                for s in (42, 43, 44)
            }
            for a in scores
        }
        paths = {
            a: {
                s: optimal_operations(ref, scores[a][s][cid]["hypothesis"])
                for s in (42, 43, 44)
            }
            for a in scores
        }
        ds = {
            a: {s: {x["index"] for x in ops[a][s]["deleted"]} for s in (42, 43, 44)}
            for a in scores
        }
        for a in scores:
            stable_counts[a] += len(set.intersection(*ds[a].values()))
            for s in (42, 43, 44):
                assert all(
                    ops[a][s][k] == scores[a][s][cid][k] for k in ("S", "D", "I", "N")
                )
                totals[a].update({k: ops[a][s][k] for k in ("S", "D", "I", "N")})
        for seed in [42, 43, 44]:
            new = ds["filtered"][seed] - ds["random"][seed]
            fixed = ds["random"][seed] - ds["filtered"][seed]
            added[seed].update(
                added_positions=len(new),
                resolved_positions=len(fixed),
                added_forced_across_optimal_paths=sum(
                    paths["filtered"][seed][i] == {"D"} for i in new
                ),
            )

            def kind(index):
                token = tokens[index]
                return (
                    "filler"
                    if token in FILLERS
                    else "single_ascii_letter"
                    if len(token) == 1 and token.isascii() and token.isalpha()
                    else "source_UNK_token"
                    if token == "unk" and "[unk]" in ref.lower()
                    else "other"
                )

            added_kinds.update(kind(i) for i in new)
            resolved_kinds.update(kind(i) for i in fixed)
        extra = []
        for i in sorted(
            set.intersection(*ds["filtered"].values())
            - set.intersection(*ds["random"].values())
        ):
            extra.append(
                dict(
                    index=i,
                    token=tokens[i],
                    random_deleted_seeds=[
                        s for s in (42, 43, 44) if i in ds["random"][s]
                    ],
                    candidate_forced_all_optimal_paths=all(
                        paths["filtered"][s][i] == {"D"} for s in (42, 43, 44)
                    ),
                    random_exact_match_all_paths_all_seeds=all(
                        paths["random"][s][i] == {"M"} for s in (42, 43, 44)
                    ),
                )
            )
        row = dict(
            clip_id=cid,
            reference=ref,
            hypotheses={
                a: {s: scores[a][s][cid]["hypothesis"] for s in (42, 43, 44)}
                for a in scores
            },
            alignment=ops,
            stable_extra=extra,
            delta_D=[
                ops["filtered"][s]["D"] - ops["random"][s]["D"] for s in (42, 43, 44)
            ],
            delta_errors=[
                sum(
                    ops["filtered"][s][k] - ops["random"][s][k] for k in ("S", "D", "I")
                )
                for s in (42, 43, 44)
            ],
        )
        rows.append(row)
        if extra:
            public.append(
                {
                    k: row[k]
                    for k in ("clip_id", "stable_extra", "delta_D", "delta_errors")
                }
                | {"interpretation": notes[cid.rsplit("-", 1)[1]]}
            )
    sensitivity = {}
    for row in rows:
        if row["clip_id"].endswith("07082"):
            alt = row["reference"].replace("alphago", "alpha go")
            sensitivity["alphago_boundary_case_07082"] = {
                a: {
                    str(seed): {
                        k: score_pair(alt, row["hypotheses"][a][seed], terms=[])[k]
                        for k in ["S", "D", "I", "N"]
                    }
                    for seed in [42, 43, 44]
                }
                for a in scores
            }
    write_json(out / "PRIOR_DELETIONS_PRIVATE.json", rows)
    summary = dict(
        status="verified_from_all_1920_paired_alignments",
        totals={a: dict(t) for a, t in totals.items()},
        stable_deleted_positions=stable_counts,
        extra_stable_positions=sum(len(r["stable_extra"]) for r in public),
        extra_stable_clips=len(public),
        cases=public,
        added_deletion_accounting=dict(
            per_seed={
                str(seed): dict(c, net_D=c["added_positions"] - c["resolved_positions"])
                for seed, c in added.items()
            },
            added_total=sum(c["added_positions"] for c in added.values()),
            resolved_total=sum(c["resolved_positions"] for c in added.values()),
            added_forced_total=sum(
                c["added_forced_across_optimal_paths"] for c in added.values()
            ),
            added_by_kind=dict(added_kinds),
            resolved_by_kind=dict(resolved_kinds),
            note="Additional accounting of the same observed old test after final training; no change to frozen selection/objective/new-test conditions",
        ),
        boundary_sensitivity=sensitivity,
        stable_extra_forced_positions=sum(
            x["candidate_forced_all_optimal_paths"]
            for r in public
            for x in r["stable_extra"]
        ),
        stable_extra_forced_and_random_exact_positions=sum(
            x["candidate_forced_all_optimal_paths"]
            and x["random_exact_match_all_paths_all_seeds"]
            for r in public
            for x in r["stable_extra"]
        ),
        human_listening="pending",
        primary_normalization_changed=False,
        source_sha256={
            f"{a}-{s}": file_hash(old / f"test-{a}-{s}/summary.json")
            for a in scores
            for s in (42, 43, 44)
        },
    )
    write_json(out / "PRIOR_ANALYSIS.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
