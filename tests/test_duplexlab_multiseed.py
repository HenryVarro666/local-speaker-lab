import json
import sys
import types

import pytest


def test_evaluation_revision_preserves_hub_snapshot_identity(tmp_path):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/run_duplexlab_multiseed.py'
    spec = importlib.util.spec_from_file_location('multiseed_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / ('a' * 40)
    base.mkdir()
    (base / 'config.json').write_text('{}')
    assert module.model_revision(base, base) == base.name
    checkpoint = tmp_path / 'checkpoint-63'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    assert len(module.model_revision(checkpoint, base)) == 64


def test_seed_adapter_sets_trainer_seed_and_keeps_default_data_seed(tmp_path, monkeypatch):
    from speaker_lab.duplex.seeded_sft import execute
    output = tmp_path / 'actual.json'
    upstream = tmp_path / 'upstream.py'
    upstream.write_text('''import json, sys
from transformers import TrainingArguments
def main():
    args = TrainingArguments(output_dir=sys.argv[1])
    with open(sys.argv[1], 'w') as f: json.dump(args, f)
''')
    calls = []
    fake = types.ModuleType('transformers')
    fake.TrainingArguments = lambda **kw: {'data_seed': None, **kw}
    fake.set_seed = calls.append
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    execute(upstream, 43, [str(output)])
    assert calls == [43]
    assert json.loads(output.read_text()) == {'seed': 43, 'data_seed': None, 'output_dir': str(output)}


def test_training_seed_is_an_adapter_argument_and_adapter_is_hash_bound(tmp_path):
    from speaker_lab.duplex.training import build_training_command
    from test_duplexlab_training import setup_export
    from speaker_lab.duplex.protocol import file_hash
    from speaker_lab.duplex import seeded_sft
    export, config = setup_export(tmp_path)
    config.update(seed=44, seed_adapter_sha256=file_hash(seeded_sft.__file__))
    command = build_training_command(config, export)
    assert command[1:3] == ['-m', 'speaker_lab.duplex.seeded_sft']
    assert command[command.index('--seed') + 1] == '44'
    assert command[command.index('--upstream') + 1] == config['script']
    config['seed_adapter_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='adapter'):
        build_training_command(config, export)


def sample_summary(errors):
    return {'normalization': 'v1', 'per_clip': [
        {'clip_id': 'a', 'reference': 'a', 'N': 1, 'S': errors[0], 'D': 0, 'I': 0},
        {'clip_id': 'b', 'reference': 'b', 'N': 1, 'S': errors[1], 'D': 0, 'I': 0}]}


def test_multi_seed_statistics_pair_speakers_and_do_not_multiply_unique_n():
    from speaker_lab.duplex.multiseed import compare_seeds
    arms = {s: {'random': sample_summary([1, 1]), 'filtered': sample_summary([0, 0])} for s in [42, 43, 44]}
    rows = [{'clip_id': 'a', 'original_speaker_id': 3, 'language_group': 'zh'},
            {'clip_id': 'b', 'original_speaker_id': 4, 'language_group': 'mixed'}]
    result = compare_seeds(arms, rows, replicates=100)
    assert result['unique_clips'] == 2
    assert result['unique_reference_tokens'] == 2
    assert result['unique_speakers'] == 2
    assert result['mean_delta'] == -1
    assert result['sd_delta'] == 0
    assert result['hierarchical_delta_interval'] == [-1., -1.]
    assert result['mean_speaker_macro_delta'] == -1


def test_multi_seed_statistics_refuse_reference_or_seed_input_changes():
    from speaker_lab.duplex.multiseed import compare_seeds
    arms = {s: {'random': sample_summary([1, 1]), 'filtered': sample_summary([0, 0])} for s in [42, 43, 44]}
    arms[44]['random']['per_clip'][0]['reference'] = 'changed'
    rows = [{'clip_id': 'a', 'original_speaker_id': 3, 'language_group': 'zh'},
            {'clip_id': 'b', 'original_speaker_id': 4, 'language_group': 'mixed'}]
    with pytest.raises(ValueError):
        compare_seeds(arms, rows)
