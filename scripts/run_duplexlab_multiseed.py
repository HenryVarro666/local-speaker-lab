"""Run the preregistered fixed-arm 42/43/44 experiment; physical GPU1 only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from speaker_lab.duplex import seeded_sft
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds
from speaker_lab.duplex.protocol import content_hash, file_hash, freeze_protocol, read_json, write_json
from speaker_lab.duplex.runner import run_offline
from speaker_lab.duplex.training import run_training


def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def model_revision(checkpoint, base):
    if checkpoint == base:
        return base.name
    return content_hash({p.name: file_hash(p) for p in checkpoint.iterdir() if p.suffix in {'.json', '.safetensors'}})


def wait_for_gpu():
    uuid = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    deadline = time.monotonic() + 3600
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if not any(uuid in line for line in apps.splitlines()):
            return
        if time.monotonic() > deadline:
            raise RuntimeError('GPU1 remained occupied; no external process was stopped')
        print('Waiting for external GPU1 work to finish', flush=True)
        time.sleep(20)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--phase', choices=['train', 'evaluate'], required=True)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('This experiment requires physical GPU1')
    root, model = args.root.resolve(), args.model.resolve()
    runs = root / 'runs/duplexlab/multiseed-v1'
    runs.mkdir(parents=True, exist_ok=True)
    old = root / 'runs/duplexlab/continuation-v1'
    data = root / 'data/duplexlab/multiseed-v1'
    policy = read_json(data / 'POLICY.json')
    for n, digest in policy['manifest_sha256'].items():
        if file_hash(data / f'{n}.jsonl') != digest:
            raise ValueError('Frozen manifest changed')
    if file_hash(root / 'experiments/duplexlab/MULTISEED_PROTOCOL.md') != policy['protocol_sha256']:
        raise ValueError('Analysis protocol changed')
    primary, secondary = load(data / 'primary.jsonl'), load(data / 'secondary.jsonl')

    if args.phase == 'train':
        import torch
        for seed in [43, 44]:
            for arm in ['random', 'filtered']:
                wait_for_gpu()
                config = read_json(old / f'train-{arm}.json')
                config.update(seed=seed, seed_adapter_sha256=file_hash(seeded_sft.__file__),
                              output_dir=str(root / f'checkpoints/duplexlab/multiseed-v1/{arm}-{seed}'),
                              record_dir=str(runs / f'train-{arm}-{seed}-record'))
                write_json(runs / f'train-{arm}-{seed}.json', config)
                result = run_training(config, old / 'export', dry_run=False)
                if result['status'] != 'checkpoint_saved_unverified':
                    raise RuntimeError('Training failed; preserve records')
                actual = torch.load(Path(result['checkpoint']) / 'training_args.bin', map_location='cpu', weights_only=False)
                observed = {k: getattr(actual, k) for k in ['seed', 'data_seed', 'bf16', 'per_device_train_batch_size',
                            'gradient_accumulation_steps', 'learning_rate', 'num_train_epochs', 'dataloader_num_workers']}
                if observed['seed'] != seed or observed['data_seed'] is not None:
                    raise ValueError('Actual trainer seed mismatch')
                result['actual_training_arguments'] = observed
                write_json(Path(config['record_dir']) / 'training.json', result)
                print(json.dumps({'seed': seed, 'arm': arm, 'status': result['status'], 'seconds': result['subprocess_seconds'], 'actual': observed}), flush=True)
        return

    def transcribe(name, records, checkpoint):
        wait_for_gpu()
        manifest = runs / f'{name}.jsonl'
        manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        revision = model_revision(checkpoint, model)
        config = runs / f'{name}.toml'
        config.write_text(f'''[project]
root = {json.dumps(str(root))}
[asr]
condition = {json.dumps(name)}
model = {json.dumps(str(checkpoint))}
model_revision = {json.dumps(revision)}
python = {json.dumps(sys.executable)}
device = "cuda:0"
dtype = "bfloat16"
language = ""
aligner = ""
attn_implementation = "sdpa"
max_new_tokens = 256
timeout_seconds = 7200
''', encoding='utf-8')
        protocol = runs / 'protocols' / name
        freeze_protocol(manifest, data, config, protocol)
        state = run_offline(protocol, data, runs / name)
        if state['status'] != 'complete':
            raise RuntimeError(f'Incomplete evaluation: {name}')
        score = score_run(runs / name, protocol, data)
        write_json(runs / name / 'summary.json', score)
        print(json.dumps({'run': name, 'status': state['status'], 'count': state['succeeded'], 'seconds': state['batch_subprocess_seconds'],
                          'overall': score['origins']['real']['overall']}), flush=True)
        return score

    base = transcribe('primary-base', primary, model)
    paired = {role: {} for role in ['primary', 'secondary', 'combined']}
    for seed in [42, 43, 44]:
        pair = {}
        for arm in ['random', 'filtered']:
            record_path = old / f'train-{arm}-record/training.json' if seed == 42 else runs / f'train-{arm}-{seed}-record/training.json'
            record = read_json(record_path)
            pair[arm] = transcribe(f'primary-{arm}-{seed}', primary, Path(record['checkpoint']))
            previous = read_json(old / f'test-{arm}/summary.json') if seed == 42 else transcribe(f'secondary-{arm}-{seed}', secondary, Path(record['checkpoint']))
            paired['secondary'].setdefault(seed, {})[arm] = previous
            paired['combined'].setdefault(seed, {})[arm] = {'normalization': pair[arm]['normalization'],
                                                        'per_clip': pair[arm]['per_clip'] + previous['per_clip']}
            if seed != 42:
                record.update(checkpoint_reloaded=True, status='trained_and_reloaded', evaluation_run=f'primary-{arm}-{seed}')
                write_json(record_path, record)
        paired['primary'][seed] = pair
    for role, records in [('primary', primary), ('secondary', secondary), ('combined', primary + secondary)]:
        result = compare_seeds(paired[role], records)
        write_json(runs / f'COMPARISON-{role}.json', result)
    write_json(runs / 'BASE.json', {'primary': base['origins']['real'], 'secondary': read_json(old / 'test-base/summary.json')['origins']['real']})


if __name__ == '__main__':
    main()
