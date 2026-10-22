import pytest


def test_paired_comparison_is_micro_aggregated_and_zero_change_has_zero_interval():
    from speaker_lab.duplex.comparison import compare_summaries
    rows = [{"clip_id": "short", "N": 1, "S": 1, "D": 0, "I": 0, "reference": "hi"},
            {"clip_id": "long", "N": 9, "S": 0, "D": 0, "I": 0, "reference": "a b c d e f g h i"}]
    result = compare_summaries({"base": {"normalization": "v1", "per_clip": rows},
                                "same": {"normalization": "v1", "per_clip": rows}}, replicates=100)
    assert result["runs"]["base"]["error_rate"] == .1
    assert result["comparisons"]["same"]["delta_interval"] == [0., 0.]
    assert result["comparisons"]["same"]["unchanged"] == 2


@pytest.mark.parametrize("change", ["missing", "reference", "normalization"])
def test_comparison_refuses_unpaired_or_different_references(change):
    from speaker_lab.duplex.comparison import compare_summaries
    base = {"normalization": "v1", "per_clip": [{"clip_id": "a", "N": 1, "S": 0, "D": 0, "I": 0, "reference": "hello"}]}
    other = {"normalization": "v1", "per_clip": [dict(base["per_clip"][0])]}
    if change == "missing":
        other["per_clip"] = []
    elif change == "reference":
        other["per_clip"][0]["reference"] = "different"
    else:
        other["normalization"] = "v2"
    with pytest.raises(ValueError):
        compare_summaries({"base": base, "other": other})
