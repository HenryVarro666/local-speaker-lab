"""Retrospective text-observable error analysis; never edits reference scores."""
from collections import Counter
import json
from pathlib import Path
import re

from speaker_lab.duplex.protocol import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/duplexlab/h100-continuation-v1'
OUT = ROOT / 'runs/duplexlab/multiseed-v1'
TAGS = {
    '01588': 'Chinese filler rendered as English',
    '01664': 'Chinese filler substitution; repeated syllables omitted in both arms',
    '01679': 'Mixed reference rendered as mostly English',
    '01701': 'English span replaced with Chinese words',
    '01723': 'Disfluency and acronym spelling/tokenization differences',
    '01756': 'Pronoun variant and extra character',
    '01777': 'Chinese filler rendered as English',
    '02127': 'Similar English lexical item; source contains UNK',
    '02145': 'English final word deletion',
    '02302': 'Chinese filler rendered as English',
    '02347': 'Homophone orthography: pronoun character',
    '02376': 'Chinese/English lexical switching; source contains UNK',
}


def main():
    summaries = {n: json.loads((OLD / f'test-{n}/summary.json').read_text()) for n in ['base', 'random', 'filtered']}
    indexed = {n: {r['clip_id']: r for r in s['per_clip']} for n, s in summaries.items()}
    meta = {r['clip_id']: r for r in map(json.loads, (ROOT / 'data/duplexlab/continuation-v1/test.jsonl').read_text().splitlines())}
    cases, movement = [], {}
    for language in ['zh', 'mixed', 'en']:
        gross_more = gross_fewer = worse = better = equal = 0
        for cid, row in indexed['random'].items():
            if row['language_group'] != language:
                continue
            f = indexed['filtered'][cid]
            delta = sum(f[k] - row[k] for k in ['S', 'D', 'I'])
            worse += delta > 0
            better += delta < 0
            equal += delta == 0
            gross_more += max(delta, 0)
            gross_fewer += max(-delta, 0)
            if delta > 0 and language in ['zh', 'mixed']:
                cases.append({'clip_id': cid, 'speaker': meta[cid]['original_speaker_id'], 'language': language,
                              'delta_edits': delta, 'duration_seconds': meta[cid]['duration_seconds'],
                              'text_observation': TAGS[cid.split('-')[-1]], 'human_listening': 'pending',
                              'audio': str(ROOT / 'data/duplexlab/continuation-v1' / meta[cid]['audio_relpath']),
                              'predictions': {n: indexed[n][cid] for n in indexed}})
        movement[language] = {'worse': worse, 'better': better, 'equal_edits': equal,
                              'gross_more_edits': gross_more, 'gross_fewer_edits': gross_fewer, 'net_edits': gross_more - gross_fewer}
    if len(cases) != 12:
        raise ValueError('Old frozen scores no longer match the analyzed twelve cases')
    selection = json.loads((OLD / 'SELECTION.json').read_text())
    pool = {r['clip_id']: r for r in json.loads((OLD / 'pool-base/summary.json').read_text())['per_clip']}
    diagnostics = {}
    for arm, rows in selection['arms'].items():
        diagnostics[arm] = {'speaker_counts': dict(sorted(Counter(r['original_speaker_id'] for r in rows).items())),
                            'languages': {}}
        for lang in ['zh', 'mixed', 'en']:
            group = [r for r in rows if r['language_group'] == lang]
            scores = [pool[r['clip_id']] for r in group]
            diagnostics[arm]['languages'][lang] = {
                'clips': len(group), 'base_exact_edit_clips': sum(r['S'] + r['D'] + r['I'] == 0 for r in scores),
                'base_micro_mer': sum(r['S'] + r['D'] + r['I'] for r in scores) / sum(r['N'] for r in scores),
                'contains_filler_char': sum(any(c in r['reference'] for c in '嗯呃啊') for r in group),
                'contains_repeated_character_3plus': sum(bool(re.search(r'([\u4e00-\u9fff])\1{2,}', r['reference'])) for r in group),
                'contains_UNK': sum('[UNK]' in r['reference'] for r in group)}
    public = {'status': 'retrospective_text_analysis', 'case_count': len(cases), 'movement': movement,
              'text_category_counts': dict(Counter(r['text_observation'] for r in cases)),
              'training_diagnostics': diagnostics, 'human_listening': 'pending',
              'source_summary_sha256': {n: file_hash(OLD / f'test-{n}/summary.json') for n in summaries},
              'causal_limit': 'Selection favors base-reference agreement. Correlated coverage changes do not establish the cause of test errors.'}
    write_json(OUT / 'REGRESSIONS-private.json', {**public, 'cases': cases})
    write_json(OUT / 'REGRESSIONS-public.json', public)
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
