from __future__ import annotations

import math
from collections import defaultdict

SPEAKERS = ("SPEAKER_00", "SPEAKER_01")


def activity_intervals(turns: list[dict], sample_count: int, sr: int = 16000) -> tuple[list[dict], dict]:
    """Sweep true (possibly overlapping) diarization into disjoint sample intervals."""
    cleaned = []
    first = {}
    for turn in turns:
        start, end = float(turn["start"]), float(turn["end"])
        label = str(turn["speaker"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Diarization returned nonfinite timestamps.")
        begin, stop = max(0, round(start * sr)), min(sample_count, round(end * sr))
        if stop > begin:
            first[label] = min(first.get(label, begin), begin)
            cleaned.append((begin, stop, label))
    if len(first) != 2:
        raise ValueError(f"Expected exactly two speakers; diarization returned {len(first)}. Check the input or backend.")
    labels = sorted(first, key=lambda label: (first[label], label))
    mapping = dict(zip(labels, SPEAKERS))
    events = defaultdict(lambda: defaultdict(int))
    events[0]
    events[sample_count]
    for start, end, label in cleaned:
        events[start][mapping[label]] += 1
        events[end][mapping[label]] -= 1
    counts = defaultdict(int)
    intervals = []
    positions = sorted(events)
    for index, start in enumerate(positions[:-1]):
        for speaker, change in events[start].items():
            counts[speaker] += change
        end = positions[index + 1]
        active = tuple(speaker for speaker in SPEAKERS if counts[speaker] > 0)
        if intervals and intervals[-1]["speakers"] == active:
            intervals[-1]["end"] = end
        else:
            intervals.append({"start": start, "end": end, "speakers": active})
    return intervals, mapping


def split_interval(start: int, end: int, max_samples: int):
    if max_samples < 1:
        raise ValueError("Chunk size must be positive.")
    while start < end:
        stop = min(start + max_samples, end)
        yield start, stop
        start = stop


def overlap_jobs(intervals: list[dict], sample_count: int, chunk_samples: int, context_samples: int) -> list[dict]:
    jobs = []
    for interval in intervals:
        if len(interval["speakers"]) != 2:
            continue
        for start, end in split_interval(interval["start"], interval["end"], chunk_samples):
            jobs.append({
                "id": f"overlap_{len(jobs):05d}", "start": start, "end": end,
                "input_start": max(0, start - context_samples),
                "input_end": min(sample_count, end + context_samples),
                "region_start": interval["start"], "region_end": interval["end"],
            })
    return jobs


def speaker_spans(intervals: list[dict], speaker: str) -> list[tuple[int, int]]:
    spans = []
    for interval in intervals:
        if speaker in interval["speakers"]:
            if spans and spans[-1][1] == interval["start"]:
                spans[-1] = (spans[-1][0], interval["end"])
            else:
                spans.append((interval["start"], interval["end"]))
    return spans
