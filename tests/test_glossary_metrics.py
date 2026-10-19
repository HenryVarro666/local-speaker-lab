import pytest

from speaker_lab.glossary_metrics import aggregate_scores, normalize_tokens, score_pair


def test_normalization_preserves_mixed_tokens_and_does_not_merge_aliases():
    assert normalize_tokens("这个 ＡＰＩ，uses Qwen3—cache；2026！") == [
        "这", "个", "api", "uses", "qwen", "3", "cache", "2026"
    ]
    assert normalize_tokens("API") != normalize_tokens("A P I")
    assert normalize_tokens("résumé") == ["résumé"]


def test_case_and_nfkc_are_equivalent_without_duplicate_dictionary_mentions():
    result = score_pair("用这个API", "用这个ａｐｉ！", ["API", "api", "ＡＰＩ"])
    assert result["mer"] == 0
    assert result["N"] == 4
    assert result["term_ref_mentions"] == result["term_hyp_mentions"] == 1
    assert result["term_correct_mentions"] == 1
    assert result["term_precision"] == result["term_recall"] == 1


def test_latin_term_boundaries_do_not_match_substrings():
    result = score_pair("capital cached caches", "capital cached caches", ["API", "cache"])
    assert result["term_ref_mentions"] == result["term_hyp_mentions"] == 0
    assert result["term_recall"] is result["term_precision"] is None


def test_wrong_glossary_term_is_both_miss_and_false_mention():
    result = score_pair("检查 cache", "检查 Redis", ["cache", "Redis"])
    assert (result["S"], result["D"], result["I"], result["N"]) == (1, 0, 0, 3)
    assert result["term_ref_mentions"] == result["term_hyp_mentions"] == 1
    assert result["term_correct_mentions"] == 0
    assert result["term_false_mentions"] == 1
    assert result["term_recall"] == result["term_precision"] == 0


def test_inserted_unspoken_term_is_a_false_mention():
    result = score_pair("今天结束", "今天 API 结束", ["API"])
    assert result["I"] == 1
    assert result["term_ref_mentions"] == result["term_correct_mentions"] == 0
    assert result["term_hyp_mentions"] == result["term_false_mentions"] == 1
    assert result["term_recall"] is None
    assert result["term_precision"] == 0


def test_reordered_mentions_are_not_credited_from_set_membership():
    result = score_pair("cache API", "API cache", ["cache", "API"])
    assert result["S"] == 2
    assert result["term_correct_mentions"] == 0
    assert result["term_false_mentions"] == 2


def test_duplicate_hypothesis_mention_cannot_reuse_reference_mention():
    result = score_pair("用 API", "用 API API", ["API"])
    assert result["term_ref_mentions"] == result["term_correct_mentions"] == 1
    assert result["term_hyp_mentions"] == 2
    assert result["term_false_mentions"] == 1


def test_phrase_mentions_take_longest_match_and_do_not_overlap():
    terms = ["machine", "learning", "machine learning", "learning API", "API"]
    result = score_pair("machine learning API", "machine learning API", terms)
    assert result["term_ref_mentions"] == result["term_correct_mentions"] == 2
    assert result["term_false_mentions"] == 0


def test_broken_phrase_does_not_count_as_correct():
    result = score_pair("machine learning", "machine deep learning", ["machine learning"])
    assert result["I"] == 1
    assert result["term_ref_mentions"] == 1
    assert result["term_hyp_mentions"] == result["term_correct_mentions"] == 0
    assert result["term_recall"] == 0


def test_empty_and_punctuation_only_references_keep_insertions():
    result = score_pair("，！？", "API", ["API"])
    assert (result["N"], result["I"], result["S"], result["D"]) == (0, 1, 0, 0)
    assert result["mer"] is None
    assert result["term_false_mentions"] == 1
    empty = score_pair("", "...", [])
    assert empty["mer"] is empty["term_recall"] is empty["term_precision"] is None


def test_deletions_and_insertions_are_counted():
    assert score_pair("今天用 API", "用 API", ["API"])["D"] == 2
    assert score_pair("API", "今天用 API", ["API"])["I"] == 3


def test_aggregation_uses_micro_counts_and_explicit_negative_clip_denominators():
    rows = [
        score_pair("API", "", ["API"]),
        score_pair("今天继续检查 cache", "今天继续检查 cache", ["API", "cache"]),
        score_pair("你好", "你好 API API", ["API"]),
        score_pair("好", "好", ["API"]),
    ]
    result = aggregate_scores(rows)
    assert result["mer"] == pytest.approx(3 / 11)
    assert result["term_recall"] == pytest.approx(1 / 2)
    assert result["term_precision"] == pytest.approx(1 / 3)
    assert result["clips"] == 4
    assert result["no_term_clips"] == 2
    assert result["no_term_false_mentions"] == 2
    assert result["no_term_false_clips"] == 1
    assert result["no_term_false_mentions_per_100_clips"] == 100
    assert result["no_term_false_clip_rate"] == 0.5


def test_empty_aggregation_rates_are_undefined():
    result = aggregate_scores([])
    assert result["clips"] == result["N"] == result["no_term_clips"] == 0
    for field in ("mer", "term_recall", "term_precision", "no_term_false_mentions_per_100_clips", "no_term_false_clip_rate"):
        assert result[field] is None


@pytest.mark.parametrize("term", ["", "   ", "，！"])
def test_empty_normalized_glossary_terms_are_rejected(term):
    with pytest.raises(ValueError, match="Glossary terms"):
        score_pair("你好", "你好", [term])
