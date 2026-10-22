import copy
import pytest


def test_counts_distinguish_filler_presence_from_frequency_and_ignore_unknown():
    from speaker_lab.duplex.disfluency import text_features

    f = text_features("嗯嗯 uh um [UNK] human summer thermal")
    assert f["filler_count"] == 4
    assert text_features("嗯 yes")["filler_count"] == 1
    assert text_features("human summer thermal like")["filler_count"] == 0
    assert text_features("[UNK]")["max_en_span"] == 0


def test_repetition_counts_adjacent_motifs_without_double_counting():
    from speaker_lab.duplex.disfluency import text_features

    assert text_features("就就就")["repeat_count"] == 2
    assert text_features("没有没有")["repeat_count"] == 1
    assert text_features("I I think")["repeat_count"] == 1
    assert text_features("然后然后然后")["repeat_count"] == 2
    assert text_features("好 [UNK] 好")["repeat_count"] == 0


def test_switch_spans_use_english_words_and_han_characters():
    from speaker_lab.duplex.disfluency import text_features

    f = text_features("我们 need to find 答案 okay")
    assert f["switch_count"] == 3
    assert f["max_en_span"] == 3
    assert f["max_zh_span"] == 2
    assert text_features("hello world")["switch_count"] == 0
    assert text_features("hello [UNK] 中文")["switch_count"] == 0


def test_deleted_tokens_reproduce_strict_counts_and_identify_filler_deletion():
    from speaker_lab.duplex.disfluency import alignment_edits
    from speaker_lab.glossary_metrics import score_pair

    for ref, hyp in [
        ("嗯 yes 嗯", "yes"),
        ("U S T C呃", "USTC呃"),
        ("就就就这样", "就这样"),
        ("ab c", "x c"),
    ]:
        edits = alignment_edits(ref, hyp)
        score = score_pair(ref, hyp, terms=[])
        assert {k: edits[k] for k in ["S", "D", "I", "N"]} == {
            k: score[k] for k in ["S", "D", "I", "N"]
        }
        assert len(edits["deleted"]) == score["D"]
    assert alignment_edits("嗯 yes 嗯", "yes")["deleted_filler_count"] == 2
    assert alignment_edits("U S T C呃", "USTC呃")["deleted_filler_count"] == 0


def fixture_pool():
    texts = [
        "嗯 yes",
        "嗯嗯 yes",
        "没有没有 yes",
        "嗯 i i think 好",
        "嗯嗯 need to go 好",
        "正常 yes text",
    ]
    return [
        dict(
            clip_id=f"c{i:03}",
            original_speaker_id=1 + i % 2,
            language_group="mixed",
            duration_seconds=2 + i % 3 * 0.01,
            reference=texts[i % 6],
            selection_mer=0.1 + i % 7 * 0.01,
            split="train",
            origin="real",
            audit_decision="pass",
            pcm_sha256=f"p{i}",
            audio_sha256=f"a{i}",
        )
        for i in range(72)
    ]


def test_selection_preserves_frequency_repetition_switch_and_existing_coverage():
    from speaker_lab.duplex.disfluency import disfluency_select, feature_marginals

    rows = fixture_pool()
    random = rows[18:54]
    chosen, audit = disfluency_select(rows, random)
    assert len(chosen) == len(random)
    assert feature_marginals(chosen) == feature_marginals(random)
    assert (
        sum(r["selection_mer"] for r in chosen)
        <= sum(r["selection_mer"] for r in random) + 1e-8
    )
    assert audit["duration_difference_fraction"] <= 0.010000001
    m = feature_marginals(chosen)
    assert m["filler_total|mixed"] > m["filler|mixed|1"]
    assert m["repeat_total|mixed"] > 0
    assert any(k.startswith("en_span|mixed|3+") for k in m)
    assert [r["clip_id"] for r in chosen] == [
        r["clip_id"] for r in disfluency_select(list(reversed(rows)), random)[0]
    ]


def test_selector_rejects_test_metadata_or_changed_random_source():
    from speaker_lab.duplex.disfluency import disfluency_select

    rows = fixture_pool()
    random = copy.deepcopy(rows[18:54])
    rows[0]["split"] = "test"
    with pytest.raises(ValueError):
        disfluency_select(rows, random)
