"""Render audited multi-seed aggregates and private replay without changing scores."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from speaker_lab.duplex.protocol import file_hash, write_json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ['source', 'old', 'data', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    src, dest = args.source, args.output
    dest.mkdir(parents=True, exist_ok=True)
    records = {f'{a}-{s}': load((args.old if s == 42 else src) / (f'train-{a}-record/training.json' if s == 42 else f'train-{a}-{s}-record/training.json'))
               for s in [42, 43, 44] for a in ['random', 'filtered']}
    if any(r['status'] != 'trained_and_reloaded' or not r['checkpoint_reloaded'] or r['optimizer_steps'] != 63 for r in records.values()):
        raise ValueError('All six checkpoints must have 63 steps and actual complete reloads')
    actual = load(src / 'ACTUAL_ARGUMENTS.json')
    for name, row in records.items():
        row['actual_training_arguments'] = actual[name]
    comparison = {role: load(src / f'COMPARISON-{role}.json') for role in ['primary', 'secondary', 'combined', 'pass_only']}
    cases = load(src / 'REGRESSIONS-private.json')['cases']
    secondary = {s: {a: {r['clip_id']: r for r in load((args.old / f'test-{a}' if s == 42 else src / f'secondary-{a}-{s}') / 'summary.json')['per_clip']}
                     for a in ['random', 'filtered']} for s in [42, 43, 44]}
    stability = []
    for case in cases:
        cid = case['clip_id']
        deltas = {str(s): sum(secondary[s]['filtered'][cid][k] - secondary[s]['random'][cid][k] for k in ['S', 'D', 'I']) for s in [42, 43, 44]}
        stability.append({'clip_id': cid, 'speaker': case['speaker'], 'language': case['language'],
                          'text_observation': case['text_observation'], 'delta_edits_by_seed': deltas,
                          'seeds_with_more_filtered_edits': sum(d > 0 for d in deltas.values())})
    lo, hi = comparison['primary']['hierarchical_delta_interval']
    conclusion = ('The paired seed/speaker interval includes zero: no reliable filtering advantage is established.' if lo <= 0 <= hi else
                  'The paired interval is above zero: filtered has higher error in this fixed pilot.' if lo > 0 else
                  'The paired interval is below zero: filtered has lower error in this fixed pilot.')
    summary = {'status': 'complete', 'research_conclusion': conclusion,
               'comparison': comparison, 'base': load(src / 'BASE.json'), 'data_policy': load(args.data / 'POLICY.json'),
               'regression_analysis': load(src / 'REGRESSIONS-public.json'),
               'prior_case_stability': {'scope': 'Descriptive secondary analysis of the twelve cases selected from seed42 errors; not an unbiased failure-rate estimate.',
                                        'cases': stability},
               'training': {n: {k: r.get(k) for k in ['status', 'checkpoint_reloaded', 'optimizer_steps', 'epoch', 'subprocess_seconds',
                                                    'actual_training_arguments', 'upstream_sha256', 'export_hash', 'checkpoint_hashes']} for n, r in records.items()},
               'source_hashes': load(src / 'SOURCE.json'),
               'reporting_source_hashes': {name: file_hash(Path(__file__).resolve().parent / name)
                                           for name in ['report_duplexlab_multiseed.py', 'verify_duplexlab_multiseed.py',
                                                        'prepare_duplexlab_multiseed.py', 'analyze_duplexlab_regressions.py']},
               'local_verification': load(src / 'LOCAL_VERIFICATION.json'),
               'timing_scope': 'WorkerRunner call to full batch return, including startup and model load; not streaming, TTS, or voice latency.',
               'limitations': ['Fixed training selections across three seeds, not three independently sampled datasets.',
                               'Primary300 has three new speaker IDs; secondary100 was previously examined.',
                               '91 primary clips have needs_review audit status; source references retained, human listening pending.',
                               'Public ASCEND exposure during pretrained model training is unknown.',
                               'Seed42 reuses the original upstream run; 43/44 add pre-load set_seed and explicit Trainer seed.']}
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if any(s in text for s in ['/Users/', '/mnt/', 'hypothesis', '"reference"']):
        raise ValueError('Public report contains private paths or full transcripts')
    (dest / 'SUMMARY.json').write_text(text + '\n', encoding='utf-8')
    write_json(dest / 'REGRESSIONS.json', summary['regression_analysis'])
    write_json(dest / 'REGRESSION_STABILITY.json', summary['prior_case_stability'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': 'duplexlab-multiseed-v1', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#316bd0', '#a266bd', '#12907b']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    primary = comparison['primary']
    for i, (s, row) in enumerate(primary['per_seed'].items()):
        axes[0].plot(['Random', 'Filtered'], [row['arms'][a]['mer'] * 100 for a in ['random', 'filtered']],
                     marker='o', color=colors[i], label=f'Seed {s}')
        values = [row['delta'] * 100] + [(row['languages'][g]['filtered']['mer'] - row['languages'][g]['random']['mer']) * 100 for g in ['zh', 'mixed', 'en']]
        axes[1].bar([x + (i - 1) * .24 for x in range(4)], values, width=.22, color=colors[i], label=s)
    axes[0].set(ylabel='MER (%)', title='New 300 clips / three new speakers')
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color='#65758b', linewidth=1)
    axes[1].set_xticks(range(4), ['Overall', 'Chinese', 'Mixed', 'English'])
    axes[1].set(ylabel='Filtered minus Random (percentage points)', title='Negative means fewer filtered errors')
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    for suffix in ['svg', 'png']:
        fig.savefig(dest / f'seed-effects.{suffix}', dpi=180, metadata={'Date': None} if suffix == 'svg' else None)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    combined = comparison['combined']
    speakers = list(combined['per_seed']['42']['speakers'])
    for i, (s, row) in enumerate(combined['per_seed'].items()):
        values = [(row['speakers'][g]['filtered']['mer'] - row['speakers'][g]['random']['mer']) * 100 for g in speakers]
        ax.bar([x + (i - 1) * .24 for x in range(len(speakers))], values, width=.22, color=colors[i], label=f'Seed {s}')
    ax.axhline(0, color='#65758b', linewidth=1)
    ax.set_xticks(range(len(speakers)), [g + (' (new)' if int(g) in [3, 4, 17] else ' (prior)') for g in speakers])
    ax.set(ylabel='Filtered minus Random (percentage points)', title='Speaker heterogeneity; prior100 is secondary, previously observed')
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    for suffix in ['svg', 'png']:
        fig.savefig(dest / f'speaker-effects.{suffix}', dpi=180, metadata={'Date': None} if suffix == 'svg' else None)
    plt.close(fig)

    style = '''<style>body{font:16px/1.6 system-ui;background:#f1f5f9;color:#172235;max-width:1100px;margin:auto;padding:24px}header,article{background:white;border:1px solid #d7e1eb;border-radius:12px;padding:20px;margin:16px 0}h1{font-size:28px}h2{font-size:18px}p{overflow-wrap:anywhere;white-space:pre-wrap}.models{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}section{border-top:3px solid #386bca}section:last-child{border-color:#12907b}audio{max-width:100%}select{font:inherit;padding:7px}article[data-category=worse]{border-left:4px solid #b75b66}article[data-category=better]{border-left:4px solid #12907b}@media(max-width:650px){body{padding:10px}.models{grid-template-columns:1fr}}</style>'''
    start = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + style
    def ecount(r):
        return r['S'] + r['D'] + r['I']
    def section(name, row):
        return f'<section><b>{name} · S/D/I={row["S"]}/{row["D"]}/{row["I"]}</b><p>{escape(row["hypothesis"])}</p></section>'
    html = start + '<title>DuplexLab regression review</title><header><h1>12 prior Chinese / mixed regressions</h1><p>Text observations only. Human listening pending; original references and scores retained. Audio may reveal source ambiguity.</p></header>'
    for r in cases:
        html += f'<article><h2>{r["clip_id"]} · speaker {r["speaker"]} · {r["language"]} · +{r["delta_edits"]} edits</h2><p>{escape(r["text_observation"])}</p><audio controls preload="none" src="{Path(r["audio"]).as_uri()}"></audio><p>Source: {escape(r["predictions"]["base"]["reference"])}</p><div class="models">'
        html += ''.join(section(n if n == 'base' else n + ' (42)', r['predictions'][n]) for n in ['base', 'random', 'filtered']) + '</div>'
        for seed in [43, 44]:
            pair = secondary[seed]
            cid = r['clip_id']
            delta = ecount(pair['filtered'][cid]) - ecount(pair['random'][cid])
            html += '<div class="models">' + section(f'Random ({seed})', pair['random'][cid]) + section(f'Filtered ({seed})', pair['filtered'][cid]) + f'<p>Seed {seed}: filtered minus random = {delta:+d} edits</p></div>'
        html += '</article>'
    replay = src / 'replay'
    replay.mkdir(exist_ok=True)
    (replay / 'regressions.html').write_text(html + '</html>', encoding='utf-8')
    indexed = {f'{a}-{s}': {r['clip_id']: r for r in load(src / f'primary-{a}-{s}/summary.json')['per_clip']}
               for a in ['random', 'filtered'] for s in [42, 43, 44]}
    indexed['base'] = {r['clip_id']: r for r in load(src / 'primary-base/summary.json')['per_clip']}
    rows = [json.loads(s) for s in (args.data / 'primary.jsonl').read_text().splitlines()]
    html = start + '<title>DuplexLab three-seed review</title><header><h1>New300 · three seeds</h1><p>Three new speaker IDs. Source labels retained; human listening pending. Filters compare strict edit counts.</p><label>Seed <select id="seed"><option>42</option><option>43</option><option>44</option></select></label> <label>Show <select id="filter"><option value="all">All</option><option value="worse">Filtered: more edits</option><option value="better">Filtered: fewer edits</option><option value="equal">Equal edits</option></select></label></header>'
    for r in rows:
        cid = r['clip_id']
        html += f'<article data-id="{cid}"><h2>{cid} · {r["language_group"]} · speaker {r["original_speaker_id"]} · audit {r["audit_decision"]}</h2><audio controls preload="none" src="{(args.data / r["audio_relpath"]).resolve().as_uri()}"></audio><p>Source: {escape(r["reference"])}</p>'
        for s in [42, 43, 44]:
            delta = ecount(indexed[f'filtered-{s}'][cid]) - ecount(indexed[f'random-{s}'][cid])
            category = 'worse' if delta > 0 else 'better' if delta < 0 else 'equal'
            html += f'<div class="seed-panel models" data-seed="{s}" data-category="{category}">'
            html += section('Base', indexed['base'][cid]) + section('Random', indexed[f'random-{s}'][cid]) + section('Filtered', indexed[f'filtered-{s}'][cid]) + '</div>'
        html += '</article>'
    html += '''<script>function update(){const s=document.querySelector('#seed').value,f=document.querySelector('#filter').value;for(const a of document.querySelectorAll('article')){let cat;for(const p of a.querySelectorAll('.seed-panel')){p.hidden=p.dataset.seed!==s;p.style.display=p.hidden?'none':'';if(!p.hidden)cat=p.dataset.category}a.dataset.category=cat;a.hidden=f!=='all'&&f!==cat}}document.querySelector('#seed').onchange=update;document.querySelector('#filter').onchange=update;update();</script></html>'''
    (replay / 'primary.html').write_text(html, encoding='utf-8')
    write_json(src / 'REPORT_ARTIFACTS.json', {'public_summary_sha256': file_hash(dest / 'SUMMARY.json'),
                                             'private_replay_clips': len(rows), 'regression_cards': len(cases)})
    print(str(dest))


if __name__ == '__main__':
    main()
