"""Freeze three previously unused ASCEND speakers before any new inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from speaker_lab.duplex.audio import audit_clip, find_exact_duplicates
from speaker_lab.duplex.datasets import sample_source_records, validate_partitions
from speaker_lab.duplex.protocol import file_hash, write_json


def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    import pyarrow.parquet as pq
    import soundfile as sf
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    out = root / 'data/duplexlab/multiseed-v1'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'original').mkdir()
    prior = [r for role in ['train', 'dev', 'test', 'language_confirmation']
             for r in load(root / f'data/duplexlab/continuation-v1/{role}.jsonl')]
    prior += load(root / 'data/duplexlab/dev20-ascend-train-v1/manifest.jsonl')
    # Old frozen manifest lacks official speaker field; metadata correction is documented.
    correction = json.loads((root / 'data/duplexlab/continuation-v1/POLICY.json').read_text())['old_metadata_correction']
    for r in prior:
        if r.get('original_speaker_id') is None:
            r.update(correction[r['clip_id']])
    rows, audits, shards = [], [], {}
    for name in ['train-00001-of-00003.parquet', 'train-00002-of-00003.parquet', 'test-00000-of-00001.parquet']:
        path = args.shards / name
        shards[name] = file_hash(path)
        split = name.split('-')[0]
        for source in pq.read_table(path).to_pylist():
            if source['original_speaker_id'] not in [3, 4, 17] or not source['transcription'].strip():
                continue
            raw = source['audio']['bytes']
            info = sf.info(io.BytesIO(raw))
            if not 1 <= info.duration <= 12:
                continue
            cid = f'ascend-{split}-' + source['id']
            row = {'schema_version': 1, 'clip_id': cid, 'audio_relpath': f'original/{cid}.wav',
                   'audio_sha256': hashlib.sha256(raw).hexdigest(),
                   'source_id': 'CAiRE/ASCEND@737e9800ae31be9932ba8464c80366559bd28424',
                   'source_group_id': f"ascend-speaker-{source['original_speaker_id']}",
                   'original_speaker_id': source['original_speaker_id'], 'session_id': source['session_id'],
                   'topic': source['topic'], 'original_filename': Path(source['path']).name,
                   'source_shard': name, 'split': 'test', 'original_split': split,
                   'language_group': source['language'], 'reference': source['transcription'],
                   'origin': 'real', 'license_id': 'CC-BY-SA-4.0', 'label_status': 'source_reference',
                   'sample_rate': info.samplerate, 'channels': info.channels, 'frames': info.frames,
                   'duration_seconds': info.duration, 'split_unit': 'original_speaker_id'}
            (out / row['audio_relpath']).write_bytes(raw)
            audit = audit_clip(row, out, out / 'derived', {})
            audits.append(audit)
            row.update(audit_decision=audit['decision'], pcm_sha256=audit.get('pcm_sha256'),
                       silence_fraction=audit.get('silence_fraction'))
            if audit['decision'] != 'reject':
                rows.append(row)
    selected = [r for speaker in [3, 4, 17]
                for r in sample_source_records([r for r in rows if r['original_speaker_id'] == speaker],
                                                {'mixed': 50, 'zh': 35, 'en': 15}, 20261020)]
    validate_partitions({'prior': prior, 'new': selected})
    ids = {r['clip_id'] for r in selected}
    if find_exact_duplicates([a for a in audits if a['clip_id'] in ids]):
        raise ValueError('Duplicate test audio')
    # Only selected original WAVs are needed; preserve all candidate audit metadata.
    for path in (out / 'original').iterdir():
        if path.stem not in ids:
            path.unlink()
    secondary = load(root / 'data/duplexlab/continuation-v1/test.jsonl')
    for row in secondary:
        (out / row['audio_relpath']).write_bytes((root / 'data/duplexlab/continuation-v1' / row['audio_relpath']).read_bytes())
    for name, records in [('primary', selected), ('secondary', secondary)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write_json(out / 'AUDIT.json', audits)
    write_json(out / 'POLICY.json', {'schema_version': 1, 'seed': 20261020, 'speakers': [3, 4, 17],
               'quotas_per_speaker': {'mixed': 50, 'zh': 35, 'en': 15}, 'counts': {'primary': len(selected), 'secondary': len(secondary)},
               'needs_review': sum(r['audit_decision'] == 'needs_review' for r in selected),
               'shard_sha256': shards, 'manifest_sha256': {n: file_hash(out / f'{n}.jsonl') for n in ['primary', 'secondary']},
               'protocol_sha256': file_hash(root / 'experiments/duplexlab/MULTISEED_PROTOCOL.md'),
               'prior_unique_speakers': sorted({r['original_speaker_id'] for r in prior}),
               'source_reference_unchanged': True})
    print((out / 'POLICY.json').read_text())


if __name__ == '__main__':
    main()
