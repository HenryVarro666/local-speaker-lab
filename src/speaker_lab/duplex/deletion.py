"""Deletion diagnostics and train-only lexical retention proxies."""

from __future__ import annotations
from collections import Counter
import re
import unicodedata
from speaker_lab.glossary_metrics import normalize_tokens
from .disfluency import feature_weights as old_features


def optimal_operations(reference, hypothesis):
    """Operations possible for each reference position across ALL minimum-edit paths."""
    a, b = normalize_tokens(reference), normalize_tokens(hypothesis)
    n, m = len(a), len(b)
    f = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            f[i][j] = min(
                f[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                f[i - 1][j] + 1,
                f[i][j - 1] + 1,
            )
    g = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n:
                g[i][j] = m - j
            elif j == m:
                g[i][j] = n - i
            else:
                g[i][j] = min(
                    g[i + 1][j + 1] + (a[i] != b[j]), 1 + g[i + 1][j], 1 + g[i][j + 1]
                )
    ops = [set() for _ in a]
    for i in range(n):
        for j in range(m + 1):
            if f[i][j] + 1 + g[i + 1][j] == f[n][m]:
                ops[i].add("D")
            if j < m and f[i][j] + (a[i] != b[j]) + g[i + 1][j + 1] == f[n][m]:
                ops[i].add("M" if a[i] == b[j] else "S")
    return ops


def token_spans(text):
    pieces = [unicodedata.normalize("NFKC", c).lower() for c in text]
    normalized = "".join(pieces)
    if normalized != unicodedata.normalize("NFKC", text).lower():
        raise ValueError(
            "Cross-character Unicode composition needs explicit offset review"
        )
    mapping = [i for i, p in enumerate(pieces) for _ in p]
    cursor, result = 0, []
    for token in normalize_tokens(text):
        start = normalized.index(token, cursor)
        end = start + len(token)
        result.append(dict(token=token, start=mapping[start], end=mapping[end - 1] + 1))
        cursor = end
    return result


def retention_spans(reference, hypothesis):
    spans = token_spans(reference)
    tokens = [x["token"] for x in spans]
    ops = optimal_operations(reference, hypothesis)
    repeated = set()
    for i in range(len(tokens)):
        for width in range(1, 5):
            if (
                len(tokens[i : i + width]) == width
                and tokens[i : i + width] == tokens[i + width : i + 2 * width]
            ):
                repeated.update(range(i, i + 2 * width))
    unknown = [(m.start(), m.end()) for m in re.finditer(r"\[UNK\]", reference, re.I)]
    result = []
    for i, span in enumerate(spans):
        token = span["token"]
        raw = reference[span["start"] : span["end"]]
        eligible = (
            token.isalpha()
            and token != "unk"
            and not (raw.isascii() and (len(raw) == 1 or raw.isupper()))
        )
        eligible &= not any(span["start"] < e and span["end"] > s for s, e in unknown)
        if eligible and (ops[i] == {"D"} or i in repeated):
            result.append(
                dict(
                    span,
                    weight=2.0,
                    reason="forced_base_deletion"
                    if ops[i] == {"D"}
                    else "adjacent_repeat_proxy",
                )
            )
    return result


def offset_weights(offsets, labels, spans):
    if len(offsets) != len(labels):
        raise ValueError("Label/offset length mismatch")
    return [
        0.0
        if label == -100
        else max(
            [1.0]
            + [s["weight"] for s in spans if start < s["end"] and end > s["start"]]
        )
        for (start, end), label in zip(offsets, labels)
    ]


def feature_weights(row):
    result = old_features(row)
    language = row["language_group"]
    d = row["base_deletion_count"]
    result[f"base_deletions|{language}|{d if d < 2 else '2+'}"] = 1
    result[f"tail_deletion|{language}|{int(row['base_tail_deletion_count'] > 0)}"] = 1
    result[f"source_unk|{language}|{int('[unk]' in row['reference'].lower())}"] = 1
    return result


def feature_marginals(rows):
    total = Counter()
    for row in rows:
        total.update(feature_weights(row))
    return dict(sorted(total.items()))
