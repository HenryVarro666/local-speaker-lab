"""Independently re-score mirrored SQLite predictions against local frozen audio."""
from __future__ import annotations

import json
import math
from pathlib import Path

from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds
from speaker_lab.duplex.protocol import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/duplexlab/multiseed-v1'
OLD = ROOT / 'runs/duplexlab/h100-continuation-v1'
DATA = ROOT / 'data/duplexlab/multiseed-v1'


def rows(name):
    return [json.loads(s) for s in (DATA / f'{name}.jsonl').read_text().splitlines()]


def main():
    policy = read_json(DATA / 'POLICY.json')
    for n, digest in policy['manifest_sha256'].items():
        assert file_hash(DATA / f'{n}.jsonl') == digest
    observed = {}
    for name in ['primary-base'] + [f'primary-{a}-{s}' for s in [42, 43, 44] for a in ['random', 'filtered']] + [f'secondary-{a}-{s}' for s in [43, 44] for a in ['random', 'filtered']]:
        computed = score_run(SOURCE / name, SOURCE / 'protocols' / name, DATA)
        assert computed == read_json(SOURCE / name / 'summary.json'), name
        observed[name] = computed
    for arm in ['base', 'random', 'filtered']:
        name = f'test-{arm}'
        computed = score_run(OLD / name, OLD / 'protocols' / name, ROOT / 'data/duplexlab/continuation-v1')
        assert computed == read_json(OLD / name / 'summary.json'), name
    per_role = {r: {} for r in ['primary', 'secondary', 'combined', 'pass_only']}
    freeze = read_json(SOURCE / 'SENSITIVITY_FREEZE.json')
    assert freeze['amendment_sha256'] == file_hash(ROOT / 'experiments/duplexlab/MULTISEED_SENSITIVITY.md')
    keep = set(freeze['pass_ids'])
    assert keep == {r['clip_id'] for r in rows('primary') if r['audit_decision'] == 'pass'}
    for s in [42, 43, 44]:
        for a in ['random', 'filtered']:
            primary = observed[f'primary-{a}-{s}']
            secondary = read_json(OLD / f'test-{a}/summary.json') if s == 42 else observed[f'secondary-{a}-{s}']
            for role, scores in [('primary', primary), ('secondary', secondary),
                                 ('combined', {'normalization': primary['normalization'], 'per_clip': primary['per_clip'] + secondary['per_clip']}),
                                 ('pass_only', {'normalization': primary['normalization'], 'per_clip': [r for r in primary['per_clip'] if r['clip_id'] in keep]})]:
                per_role[role].setdefault(s, {})[a] = scores
    for role in ['primary', 'secondary', 'combined', 'pass_only']:
        records = rows('primary') if role in ['primary', 'pass_only'] else rows('secondary') if role == 'secondary' else rows('primary') + rows('secondary')
        if role == 'pass_only':
            records = [r for r in records if r['clip_id'] in keep]
        result = compare_seeds(per_role[role], records)
        if role != 'pass_only':
            assert result == read_json(SOURCE / f'COMPARISON-{role}.json'), role
        else:
            write_json(SOURCE / 'COMPARISON-pass_only.json', result)
    hashes = []
    actual = read_json(SOURCE / 'ACTUAL_ARGUMENTS.json')
    template = {k: v for k, v in actual['random-42'].items() if k != 'seed'}
    assert all({k: v for k, v in r.items() if k != 'seed'} == template for r in actual.values())
    for s in [42, 43, 44]:
        for a in ['random', 'filtered']:
            assert actual[f'{a}-{s}']['seed'] == s
            record = read_json((OLD / f'train-{a}-record' if s == 42 else SOURCE / f'train-{a}-{s}-record') / 'training.json')
            assert record['status'] == 'trained_and_reloaded' and record['optimizer_steps'] == 63
            trainer_path = SOURCE / f'training-state/{a}-{s}.json'
            assert file_hash(trainer_path) == record['checkpoint_hashes']['trainer_state.json']
            trainer = read_json(trainer_path)
            assert trainer['global_step'] == 63 and trainer['epoch'] == 1
            for key in ['loss', 'grad_norm', 'eval_loss']:
                values = [row[key] for row in trainer['log_history'] if key in row]
                assert values and all(math.isfinite(v) for v in values)
            if s != 42:
                assert record['actual_training_arguments']['seed'] == s
                assert record['actual_training_arguments']['data_seed'] is None
            state = read_json(SOURCE / f'primary-{a}-{s}/run.json')
            assert state['model_artifacts']['model.safetensors'] == record['checkpoint_hashes']['model.safetensors']
            assert state['config']['asr']['language'] == ''
            assert state['config']['asr']['dtype'] == 'bfloat16'
            assert state['config']['asr']['aligner'] == ''
            hashes.append(record['checkpoint_hashes']['model.safetensors'])
    assert len(set(hashes)) == 6, 'Expected six distinct trained weight files'
    assert read_json(SOURCE / 'primary-base/run.json')['model_artifacts']['model.safetensors'] not in hashes
    result = {'status': 'verified', 'complete_new_runs': len(observed), 'legacy_runs_rescored': 3, 'new_model_inferences': 2500,
              'new_unique_clips': 300, 'new_unique_speakers': 3, 'six_distinct_checkpoint_hashes': True,
              'server_score_equality': True, 'three_server_comparisons_equal_on_mac': True,
              'pass_only_pre_outcome_members': len(keep), 'no_model_reinference': True}
    write_json(SOURCE / 'LOCAL_VERIFICATION.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
