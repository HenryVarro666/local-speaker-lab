"""Reference-agreement selection with matched speaker/difficulty/filler coverage."""

from __future__ import annotations

from collections import Counter
import math
import re
import time

from .datasets import stratum

_FILLER = re.compile(r"(?<![a-z])(?:uh|um|hmm|erm|er|ah|eh)(?![a-z])", re.I)


def has_filler(text: str) -> bool:
    return any(c in text for c in "嗯呃啊额唔诶欸") or bool(_FILLER.search(text))


def difficulty(row: dict) -> str:
    value = row["selection_mer"]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid base-reference disagreement")
    return "easy" if value == 0 else "medium" if value <= 0.2 else "hard"


def features(row: dict) -> list[str]:
    lang = row["language_group"]
    return [
        f"duration|{stratum(row)}",
        f"speaker|{lang}|{row['original_speaker_id']}",
        f"difficulty|{lang}|{difficulty(row)}",
        f"filler|{lang}|{int(has_filler(row['reference']))}",
    ]


def feature_marginals(rows: list[dict]) -> dict:
    return dict(sorted(Counter(key for row in rows for key in features(row)).items()))


def coverage_select(
    pool: list[dict],
    random_rows: list[dict],
    time_limit: float = 120,
    *,
    feature_weights=None,
    objective=None,
) -> tuple[list[dict], dict]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    objective = objective or (lambda row: row["selection_mer"])
    weights = feature_weights or (lambda row: dict.fromkeys(features(row), 1))

    def marginals(records):
        total = Counter()
        for row in records:
            total.update(weights(row))
        return dict(sorted(total.items()))

    rows = sorted(pool, key=lambda r: r["clip_id"])
    ids, pcm = set(), set()
    for r in rows:
        if (
            r["split"] != "train"
            or r["origin"] != "real"
            or r["audit_decision"] != "pass"
            or r["clip_id"] in ids
            or r["pcm_sha256"] in pcm
        ):
            raise ValueError(
                "Coverage candidates must be unique audited real train records"
            )
        ids.add(r["clip_id"])
        pcm.add(r["pcm_sha256"])
        features(r)
    wanted = {r["clip_id"] for r in random_rows}
    if not wanted or len(wanted) != len(random_rows) or not wanted <= ids:
        raise ValueError("Random arm must be a unique subset of train candidates")
    indexed = {r["clip_id"]: r for r in rows}
    if any(r != indexed[r["clip_id"]] for r in random_rows):
        raise ValueError("Random records differ from candidate metadata")
    targets = marginals(random_rows)
    row_weights = [weights(r) for r in rows]
    keys = sorted({k for w in row_weights for k in w})
    table = np.array([[w.get(k, 0) for w in row_weights] for k in keys], dtype=float)
    lower = [targets.get(k, 0) for k in keys]
    upper = list(lower)
    duration = np.array([r["duration_seconds"] for r in rows])
    seconds = sum(r["duration_seconds"] for r in random_rows)
    table = np.vstack([table, duration])
    lower.append(seconds * 0.99)
    upper.append(seconds * 1.01)
    cost = np.array(
        [objective(r) + 1e-7 * i / len(rows) for i, r in enumerate(rows)]
    )
    random_cost = sum(cost[i] for i, r in enumerate(rows) if r["clip_id"] in wanted)
    table = np.vstack([table, cost])
    lower.append(-np.inf)
    upper.append(random_cost + 1e-9)
    started = time.perf_counter()
    result = milp(
        cost,
        integrality=np.ones(len(rows)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(csc_matrix(table), lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    chosen = random_rows
    fallback = True
    if (
        result.status in [0, 1]
        and result.x is not None
        and np.max(np.abs(result.x - np.rint(result.x))) < 1e-5
    ):
        candidate = [r for r, x in zip(rows, result.x) if x > 0.5]
        vector = np.array(
            [int(r["clip_id"] in {v["clip_id"] for v in candidate}) for r in rows]
        )
        actual = table @ vector
        if np.all(actual >= np.array(lower) - 1e-7) and np.all(
            actual <= np.array(upper) + 1e-7
        ):
            chosen, fallback = candidate, False
    assert marginals(chosen) == targets
    gap = abs(sum(r["duration_seconds"] for r in chosen) - seconds) / seconds
    if gap > 0.010000001:
        raise ValueError("Solver violated duration budget")
    return sorted(chosen, key=lambda r: r["clip_id"]), {
        "solver_status": int(result.status),
        "solver_message": result.message,
        "solver_seconds": time.perf_counter() - started,
        "mip_gap": getattr(result, "mip_gap", None),
        "fallback_random": fallback,
        "constraints_verified": True,
        "marginals": targets,
        "duration_difference_fraction": gap,
        "random_objective": sum(objective(r) for r in random_rows),
        "coverage_objective": sum(objective(r) for r in chosen),
        "overlap": len(wanted & {r["clip_id"] for r in chosen}),
    }
