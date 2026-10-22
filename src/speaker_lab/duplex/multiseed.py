"""Paired training-seed and speaker-cluster resampling without pseudo-replication."""
from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean, stdev

from .comparison import compare_summaries


def compare_seeds(arms: dict, records: list[dict], seed: int = 20261020, replicates: int = 5000) -> dict:
    if len(arms) < 2 or replicates < 20:
        raise ValueError('Need multiple seeds and at least 20 bootstrap replicates')
    if any(set(v) != {'random', 'filtered'} for v in arms.values()):
        raise ValueError('Each seed needs exactly the paired random and filtered arms')
    flattened = {f'{s}:{arm}': summary for s, v in arms.items() for arm, summary in v.items()}
    # Existing validator checks identical clips, reference text, N and normalization.
    compare_summaries(flattened, replicates=20)
    indexed = {s: {a: {r['clip_id']: r for r in summary['per_clip']} for a, summary in v.items()}
               for s, v in arms.items()}
    first = next(iter(indexed.values()))['random']
    if len({r['clip_id'] for r in records}) != len(records) or set(first) != {r['clip_id'] for r in records}:
        raise ValueError('Speaker metadata does not match paired predictions')
    groups = defaultdict(list)
    for row in records:
        if row.get('original_speaker_id') is None:
            raise ValueError('Missing official speaker metadata')
        groups[row['original_speaker_id']].append(row['clip_id'])
    if len(groups) < 2:
        raise ValueError('Need multiple speakers')

    def score(rows, ids):
        n = sum(rows[c]['N'] for c in ids)
        if n <= 0:
            raise ValueError('No reference tokens in a speaker group')
        counts = {k: sum(rows[c][k] for c in ids) for k in ['S', 'D', 'I']}
        return {**counts, 'N': n, 'errors': sum(counts.values()), 'mer': sum(counts.values()) / n}

    per_seed = {}
    for s, pair in indexed.items():
        totals = {a: score(rows, first) for a, rows in pair.items()}
        speakers = {str(g): {a: score(rows, ids) for a, rows in pair.items()} for g, ids in groups.items()}
        languages = {lang: {a: score(rows, [r['clip_id'] for r in records if r['language_group'] == lang])
                            for a, rows in pair.items()} for lang in sorted({r['language_group'] for r in records})}
        per_seed[str(s)] = {'arms': totals, 'speakers': speakers, 'languages': languages,
                           'delta': totals['filtered']['mer'] - totals['random']['mer'],
                           'speaker_macro_delta': mean(v['filtered']['mer'] - v['random']['mer'] for v in speakers.values())}
    rng, deltas = random.Random(seed), []
    seeds, speakers = sorted(indexed), sorted(groups)
    for _ in range(replicates):
        # Resample crossed factors independently, preserving arm pairing and whole speakers.
        drawn_seeds = rng.choices(seeds, k=len(seeds))
        ids = [cid for g in rng.choices(speakers, k=len(speakers)) for cid in groups[g]]
        deltas.append(mean(score(indexed[s]['filtered'], ids)['mer'] - score(indexed[s]['random'], ids)['mer'] for s in drawn_seeds))
    deltas.sort()
    observed = [r['delta'] for r in per_seed.values()]
    return {'schema_version': 1, 'unique_clips': len(first), 'unique_reference_tokens': sum(r['N'] for r in first.values()),
            'unique_speakers': len(groups), 'training_seeds': seeds, 'per_seed': per_seed,
            'mean_delta': mean(observed), 'sd_delta': stdev(observed), 'range_delta': [min(observed), max(observed)],
            'mean_speaker_macro_delta': mean(r['speaker_macro_delta'] for r in per_seed.values()),
            'mean_mer': {a: mean(r['arms'][a]['mer'] for r in per_seed.values()) for a in ['random', 'filtered']},
            'hierarchical_delta_interval': [deltas[max(0, int(.025 * len(deltas)) - 1)], deltas[int(.975 * len(deltas)) - 1]],
            'bootstrap_seed': seed, 'bootstrap_replicates': replicates,
            'interval_unit': 'paired crossed resampling of training seeds and whole speakers; clips within speaker retained',
            'limitations': 'Only three training seeds and few public-data speakers; exploratory interval, not population proof.'}
