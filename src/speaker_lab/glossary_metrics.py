"""Dependency-free, deterministic scoring for short Mandarin-English ASR clips.

``normalize_tokens`` applies NFKC and lowercase, emits one token per Han
character, one per consecutive Latin-letter word, and one per digit run.
Punctuation/whitespace separate tokens; no acronym, number, spelling, translation,
or semantic aliases are merged. Other alphabetic scripts are preserved as word
tokens, rather than silently discarded. Thus ``Qwen3`` is ``["qwen", "3"]``
and ``API`` differs from ``A P I``. References and hypotheses use the same rules.

``score_pair`` returns integer ``S``, ``D``, ``I``, ``N`` (reference token count),
``mer`` = (S+D+I)/N, and integer ``term_ref_mentions``, ``term_hyp_mentions``,
``term_correct_mentions``, ``term_false_mentions``. ``term_recall`` and
``term_precision`` divide correct mentions by reference/hypothesis mentions.
All undefined rates are ``None``; an empty reference still records insertions.

Terms are exact normalized token sequences, deduplicated after normalization.
Mentions are selected left-to-right, taking the longest term at each position,
without overlaps. Empty normalized terms are rejected. A hypothesis mention is
correct only when every token is an exact match to the corresponding contiguous
reference mention in one deterministic minimum-edit alignment. Backtracking
prefers a diagonal (match/substitution), then deletion, then insertion on ties.
This alignment convention avoids bag-of-terms credit for arbitrary mentions;
ambiguous/repeated text still has the usual edit-alignment ambiguity.

``aggregate_scores`` micro-aggregates these fields from score dictionaries (extra
metadata is ignored). It additionally returns ``clips``, ``no_term_clips``,
``no_term_false_mentions``, ``no_term_false_clips``,
``no_term_false_mentions_per_100_clips``, and ``no_term_false_clip_rate``.
"""

from __future__ import annotations

import unicodedata


def _is_han(char: str) -> bool:
    name = unicodedata.name(char, "")
    return name.startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-"))


def normalize_tokens(text: str) -> list[str]:
    """Normalize text without merging aliases or translating recognized words."""
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    pending: list[str] = []
    kind: str | None = None

    def flush() -> None:
        if pending:
            tokens.append("".join(pending))
            pending.clear()

    for char in text:
        if _is_han(char):
            flush()
            kind = None
            tokens.append(char)
        elif char.isdecimal():
            if kind != "digits":
                flush()
            pending.append(char)
            kind = "digits"
        elif char.isalpha():
            if kind != "word":
                flush()
            pending.append(char)
            kind = "word"
        elif unicodedata.category(char).startswith("M") and kind == "word":
            # Preserve decomposed diacritics in Latin and other word tokens.
            pending.append(char)
        else:
            flush()
            kind = None
    flush()
    return tokens


def _term_patterns(terms: list[str]) -> list[tuple[str, ...]]:
    patterns = {tuple(normalize_tokens(term)) for term in terms}
    if () in patterns:
        raise ValueError("Glossary terms must contain at least one normalized token")
    return sorted(patterns, key=lambda item: (-len(item), item))


def _mentions(tokens: list[str], patterns: list[tuple[str, ...]]) -> list[tuple[int, int]]:
    mentions = []
    start = 0
    while start < len(tokens):
        for pattern in patterns:
            end = start + len(pattern)
            if tuple(tokens[start:end]) == pattern:
                mentions.append((start, end))
                start = end
                break
        else:
            start += 1
    return mentions


def _alignment(reference: list[str], hypothesis: list[str]) -> tuple[dict, dict[int, int]]:
    """Return S/D/I/N and exact hypothesis-to-reference token matches."""
    n, m = len(reference), len(hypothesis)
    costs = [list(range(m + 1))]
    for i, token in enumerate(reference, 1):
        row = [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(
                costs[i - 1][j - 1] + (token != other),
                costs[i - 1][j] + 1,
                row[j - 1] + 1,
            ))
        costs.append(row)

    counts = {"S": 0, "D": 0, "I": 0, "N": n}
    exact: dict[int, int] = {}
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            if reference[i - 1] == hypothesis[j - 1]:
                exact[j - 1] = i - 1
            else:
                counts["S"] += 1
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            counts["D"] += 1
            i -= 1
        else:
            counts["I"] += 1
            j -= 1
    return counts, exact


def _with_rates(counts: dict) -> dict:
    result = dict(counts)
    result["mer"] = (counts["S"] + counts["D"] + counts["I"]) / counts["N"] if counts["N"] else None
    result["term_recall"] = counts["term_correct_mentions"] / counts["term_ref_mentions"] if counts["term_ref_mentions"] else None
    result["term_precision"] = counts["term_correct_mentions"] / counts["term_hyp_mentions"] if counts["term_hyp_mentions"] else None
    return result


def score_pair(reference: str, hypothesis: str, terms: list[str]) -> dict:
    """Score one clip; see the module docstring for normalization and schema."""
    ref, hyp = normalize_tokens(reference), normalize_tokens(hypothesis)
    patterns = _term_patterns(terms)
    ref_mentions = set(_mentions(ref, patterns))
    hyp_mentions = _mentions(hyp, patterns)
    counts, exact = _alignment(ref, hyp)
    correct = 0
    for start, end in hyp_mentions:
        ref_start = exact.get(start)
        if ref_start is None:
            continue
        ref_end = ref_start + end - start
        if (ref_start, ref_end) in ref_mentions and all(
            exact.get(position) == ref_start + position - start for position in range(start, end)
        ):
            correct += 1
    counts.update(
        term_ref_mentions=len(ref_mentions),
        term_hyp_mentions=len(hyp_mentions),
        term_correct_mentions=correct,
        term_false_mentions=len(hyp_mentions) - correct,
    )
    return _with_rates(counts)


def aggregate_scores(rows: list[dict]) -> dict:
    """Micro-aggregate per-clip scores; undefined ratios remain ``None``."""
    fields = ("S", "D", "I", "N", "term_ref_mentions", "term_hyp_mentions",
              "term_correct_mentions", "term_false_mentions")
    result = _with_rates({field: sum(row[field] for row in rows) for field in fields})
    no_term = [row for row in rows if row["term_ref_mentions"] == 0]
    false_mentions = sum(row["term_false_mentions"] for row in no_term)
    false_clips = sum(row["term_false_mentions"] > 0 for row in no_term)
    result.update(
        clips=len(rows),
        no_term_clips=len(no_term),
        no_term_false_mentions=false_mentions,
        no_term_false_clips=false_clips,
        no_term_false_mentions_per_100_clips=100 * false_mentions / len(no_term) if no_term else None,
        no_term_false_clip_rate=false_clips / len(no_term) if no_term else None,
    )
    return result
