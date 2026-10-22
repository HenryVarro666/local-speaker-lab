"""Lexical disfluency proxies; no human acoustic or semantic annotation implied."""

from __future__ import annotations

from collections import Counter
import re

from speaker_lab.glossary_metrics import _is_han, normalize_tokens
from .coverage import coverage_select, features as coverage_features

FILLERS = set("嗯呃啊额唔诶欸") | {"uh", "um", "hmm", "erm", "er", "ah", "eh"}


def _tokens(text):
    tokens = []
    for part in re.split(r"(\[UNK\])", text, flags=re.I):
        tokens.extend(["<unk>"] if part.lower() == "[unk]" else normalize_tokens(part))
    return tokens


def text_features(text: str) -> dict:
    tokens = _tokens(text)
    repeat, i = 0, 0
    while i < len(tokens):
        best = (0, 0, 0)
        for size in range(1, 5):
            unit = tokens[i : i + size]
            if len(unit) != size or "<unk>" in unit:
                continue
            copies = 1
            while tokens[i + copies * size : i + (copies + 1) * size] == unit:
                copies += 1
            if copies > 1 and (copies * size, -size) > (best[0], -best[1]):
                best = (copies * size, size, copies - 1)
        if best[0]:
            repeat += best[2]
            i += best[0]
        else:
            i += 1
    switches, last, length = 0, None, 0
    longest = {"en": 0, "zh": 0}
    for token in tokens:
        lang = (
            "zh"
            if len(token) == 1 and _is_han(token)
            else "en"
            if re.fullmatch(r"[a-z]+", token)
            else None
        )
        if lang is None:
            last, length = None, 0
            continue
        if last is not None and lang != last:
            switches += 1
        length = length + 1 if lang == last else 1
        longest[lang] = max(longest[lang], length)
        last = lang
    return {
        "filler_count": sum(t in FILLERS for t in tokens),
        "repeat_count": repeat,
        "switch_count": switches,
        "max_en_span": longest["en"],
        "max_zh_span": longest["zh"],
    }


def feature_weights(row: dict) -> dict:
    lang = row["language_group"]
    f = text_features(row["reference"])
    result = dict.fromkeys(coverage_features(row), 1)
    for key, prefix in [
        ("filler_count", "filler_frequency"),
        ("repeat_count", "repetition"),
    ]:
        count = f[key]
        bucket = str(count) if count < 2 else "2+"
        result[f"{prefix}|{lang}|{bucket}"] = 1
    switches = f["switch_count"]
    result[
        f"switches|{lang}|{switches if switches < 2 else '2-3' if switches < 4 else '4+'}"
    ] = 1
    for key, prefix, limit in [
        ("max_en_span", "en_span", 2),
        ("max_zh_span", "zh_span", 5),
    ]:
        n = f[key]
        bucket = "0" if n == 0 else f"1-{limit}" if n <= limit else f"{limit + 1}+"
        result[f"{prefix}|{lang}|{bucket}"] = 1
    for key, prefix in [
        ("filler_count", "filler_total"),
        ("repeat_count", "repeat_total"),
    ]:
        if f[key]:
            result[f"{prefix}|{lang}"] = f[key]
    return result


def feature_marginals(rows: list[dict]) -> dict:
    total = Counter()
    for row in rows:
        total.update(feature_weights(row))
    return dict(sorted(total.items()))


def disfluency_select(pool, random_rows, time_limit=120):
    return coverage_select(
        pool, random_rows, time_limit, feature_weights=feature_weights
    )


def alignment_edits(reference: str, hypothesis: str) -> dict:
    """Same minimum-edit/tie convention as the frozen primary scorer; expose deletions."""
    ref, hyp = normalize_tokens(reference), normalize_tokens(hypothesis)
    n, m = len(ref), len(hyp)
    dp = [list(range(m + 1))]
    for i, token in enumerate(ref, 1):
        row = [i]
        for j, other in enumerate(hyp, 1):
            row.append(
                min(
                    dp[i - 1][j - 1] + (token != other),
                    dp[i - 1][j] + 1,
                    row[j - 1] + 1,
                )
            )
        dp.append(row)
    result = dict(S=0, D=0, I=0, N=n, deleted=[])
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            result["S"] += int(ref[i - 1] != hyp[j - 1])
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            result["D"] += 1
            result["deleted"].append({"index": i - 1, "token": ref[i - 1]})
            i -= 1
        else:
            result["I"] += 1
            j -= 1
    result["deleted"].reverse()
    result["deleted_filler_count"] = sum(
        t["token"] in FILLERS for t in result["deleted"]
    )
    return result
