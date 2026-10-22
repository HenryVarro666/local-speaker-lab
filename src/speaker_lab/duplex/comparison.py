"""Paired micro error-rate comparisons with explicitly clip-level intervals."""
from __future__ import annotations

from pathlib import Path
import random

from .metrics import score_run
from .protocol import read_json, verify_protocol


def compare_summaries(summaries: dict, seed: int = 20261020, replicates: int = 1000) -> dict:
    first = next(iter(summaries))
    indexed = {}
    for name, summary in summaries.items():
        rows = summary["per_clip"]
        indexed[name] = {r["clip_id"]: r for r in rows}
        if len(indexed[name]) != len(rows) or not rows:
            raise ValueError("Missing or duplicate predictions")
        if summary["normalization"] != summaries[first]["normalization"] or set(indexed[name]) != set(indexed[first]):
            raise ValueError("Cannot compare different normalization or unpaired clips")
        for cid, row in indexed[name].items():
            base = indexed[first][cid]
            if row["reference"] != base["reference"] or row["N"] != base["N"]:
                raise ValueError("Cannot compare changed references")
    ids = sorted(indexed[first])
    runs, comparisons = {}, {}
    def errors(row):
        return row["S"] + row["D"] + row["I"]
    for name, rows in indexed.items():
        n = sum(r["N"] for r in rows.values())
        if n <= 0:
            raise ValueError("No reference tokens")
        runs[name] = {"clips": len(rows), "N": n, **{k: sum(r[k] for r in rows.values()) for k in ["S", "D", "I"]},
                      "error_rate": sum(errors(r) for r in rows.values()) / n}
        if name == first:
            continue
        pairs = [{"clip_id": cid, "N": rows[cid]["N"], "baseline_edits": errors(indexed[first][cid]),
                  "candidate_edits": errors(rows[cid])} for cid in ids]
        rng, deltas = random.Random(seed), []
        for _ in range(replicates):
            sampled = rng.choices(pairs, k=len(pairs))
            total = sum(r["N"] for r in sampled)
            if total:
                deltas.append(sum(r["candidate_edits"] - r["baseline_edits"] for r in sampled) / total)
        deltas.sort()
        comparisons[name] = {"improved": sum(r["candidate_edits"] < r["baseline_edits"] for r in pairs),
                             "worse": sum(r["candidate_edits"] > r["baseline_edits"] for r in pairs),
                             "unchanged": sum(r["candidate_edits"] == r["baseline_edits"] for r in pairs),
                             "delta": runs[name]["error_rate"] - runs[first]["error_rate"],
                             "delta_interval": [deltas[max(0, int(.025 * len(deltas)) - 1)], deltas[min(len(deltas) - 1, int(.975 * len(deltas)) - 1)]],
                             "pairs": pairs}
    return {"schema_version": 1, "baseline": first, "normalization": summaries[first]["normalization"],
            "runs": runs, "comparisons": comparisons, "bootstrap_seed": seed, "bootstrap_replicates": replicates,
            "interval_unit": "clip; not speaker-cluster or population-generalization interval"}


def compare_runs(run_dirs: dict[str, Path], seed: int = 20261020, replicates: int = 1000) -> dict:
    signatures, summaries = {}, {}
    for name, run in run_dirs.items():
        state = read_json(run / "run.json")
        protocol, data = Path(state["protocol_dir"]), Path(state["data_root"])
        records = verify_protocol(protocol, data)
        signatures[name] = {r["clip_id"]: (r["audio_sha256"], r["reference"], r["language_group"]) for r in records}
        summaries[name] = score_run(run, protocol, data)
    if any(s != next(iter(signatures.values())) for s in signatures.values()):
        raise ValueError("Run input audio/reference signatures differ")
    return compare_summaries(summaries, seed, replicates)
