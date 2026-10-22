from collections import Counter
import copy

import pytest


def test_filler_proxy_uses_ascii_boundaries_and_explicit_chinese_markers():
    from speaker_lab.duplex.coverage import has_filler

    for text in ["嗯好的", "uh yes", "这个UM可以", "hmm, yes", "Erm"]:
        assert has_filler(text)
    for text in ["human", "summer", "thermal", "like apples", "hello"]:
        assert not has_filler(text)


def pool():
    rows = []
    for i in range(48):
        rows.append(
            {
                "clip_id": f"c{i:02}",
                "original_speaker_id": 1 + i % 2,
                "language_group": "mixed",
                "duration_seconds": 2 + i % 3 * 0.01,
                "reference": "嗯 yes" if i % 4 < 2 else "yes hello",
                "selection_mer": [0.0, 0.1, 0.3, 0.8][i % 4],
                "split": "train",
                "origin": "real",
                "audit_decision": "pass",
                "pcm_sha256": f"pcm{i}",
                "audio_sha256": f"audio{i}",
            }
        )
    return rows


def test_coverage_selection_preserves_all_marginals_and_budget():
    from speaker_lab.duplex.coverage import coverage_select, feature_marginals

    rows = pool()
    random = rows[12:36]
    chosen, audit = coverage_select(rows, random)
    assert len(chosen) == len(random)
    assert feature_marginals(chosen) == feature_marginals(random)
    assert (
        abs(
            sum(r["duration_seconds"] for r in chosen)
            - sum(r["duration_seconds"] for r in random)
        )
        / sum(r["duration_seconds"] for r in random)
        <= 0.01
    )
    assert (
        sum(r["selection_mer"] for r in chosen)
        <= sum(r["selection_mer"] for r in random) + 1e-8
    )
    assert audit["constraints_verified"]
    assert {r["clip_id"] for r in chosen} == {
        r["clip_id"] for r in coverage_select(list(reversed(rows)), random)[0]
    }


@pytest.mark.parametrize(
    "change", ["test", "review", "duplicate", "nonfinite", "unknown_random"]
)
def test_coverage_selection_rejects_leakage_or_invalid_candidates(change):
    from speaker_lab.duplex.coverage import coverage_select

    rows = pool()
    random = copy.deepcopy(rows[12:36])
    if change == "test":
        rows[0]["split"] = "test"
    elif change == "review":
        rows[0]["audit_decision"] = "needs_review"
    elif change == "duplicate":
        rows[0]["pcm_sha256"] = rows[1]["pcm_sha256"]
    elif change == "nonfinite":
        rows[0]["selection_mer"] = float("inf")
    else:
        random[0]["clip_id"] = "not-in-pool"
    with pytest.raises(ValueError):
        coverage_select(rows, random)


def test_marginals_retain_hard_cases_fillers_and_speakers():
    from speaker_lab.duplex.coverage import feature_marginals

    m = feature_marginals(pool())
    assert m["difficulty|mixed|hard"] == 24
    assert m["difficulty|mixed|easy"] == 12
    assert m["filler|mixed|1"] == 24
    assert Counter(v for k, v in m.items() if k.startswith("speaker|")) == {24: 2}
