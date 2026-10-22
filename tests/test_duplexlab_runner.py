from pathlib import Path
import sys

import pytest

from duplex_helpers import make_clip, write_manifest


def setup_run(tmp_path):
    from speaker_lab.duplex.protocol import freeze_protocol
    root = tmp_path / "data"
    record = make_clip(root, origin="fixture")
    manifest = write_manifest(tmp_path / "manifest.jsonl", [record])
    config = tmp_path / "config.toml"
    config.write_text('[asr]\ncondition="fixture"\nmodel="fixture"\n'
                      'backend="duplex_fixture_backend:Backend"\nlanguage=""\naligner=""\n'
                      f'python="{Path(sys.executable).as_posix()}"\n', encoding="utf-8")
    protocol = tmp_path / "protocol"
    freeze_protocol(manifest, root, config, protocol)
    return root, protocol


def test_offline_run_uses_real_worker_and_resume_keeps_committed_result(tmp_path, monkeypatch):
    from speaker_lab.duplex.runner import run_offline
    root, protocol = setup_run(tmp_path)
    plugin = tmp_path / "plugins"
    plugin.mkdir()
    (plugin / "duplex_fixture_backend.py").write_text(
        "class Backend:\n"
        " def run(self, payload, options):\n"
        "  assert options['language'] == '' and options['aligner'] == ''\n"
        "  return {'segments': [{'job_id': j['id'], 'text': '明天 at four thirty'} for j in payload['jobs']]}\n",
        encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(plugin))
    output = tmp_path / "run"
    first = run_offline(protocol, root, output)
    assert first["status"] == "complete" and first["succeeded"] == 1
    (plugin / "duplex_fixture_backend.py").write_text("raise RuntimeError('must not rerun')\n")
    assert run_offline(protocol, root, output, resume=True)["succeeded"] == 1
    with pytest.raises(FileExistsError):
        run_offline(protocol, root, output)


def test_run_rejects_changed_protocol_on_resume_before_starting_model(tmp_path):
    from speaker_lab.duplex.runner import run_offline
    root, protocol = setup_run(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text('{"protocol_hash":"another-protocol"}')
    with pytest.raises(ValueError, match="protocol"):
        run_offline(protocol, root, output, resume=True)


def test_known_language_diagnostic_is_explicitly_forwarded_to_worker(tmp_path, monkeypatch):
    from speaker_lab.duplex.protocol import freeze_protocol
    from speaker_lab.duplex.runner import run_offline
    root, _ = setup_run(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(config.read_text().replace('language=""', 'language="English"'))
    protocol = tmp_path / "english-protocol"
    freeze_protocol(tmp_path / "manifest.jsonl", root, config, protocol)
    plugin = tmp_path / "plugins"
    plugin.mkdir()
    (plugin / "duplex_fixture_backend.py").write_text(
        "class Backend:\n"
        " def run(self, payload, options):\n"
        "  if options['language'] != 'English': raise ValueError('Language was not forwarded')\n"
        "  return {'segments': [{'job_id': j['id'], 'text': 'known language'} for j in payload['jobs']]}\n")
    monkeypatch.setenv("PYTHONPATH", str(plugin))
    result = run_offline(protocol, root, tmp_path / "known-language-run")
    assert result["status"] == "complete"


def test_qwen_rejects_unpinned_remote_model_before_loading(tmp_path):
    from speaker_lab.duplex.protocol import freeze_protocol
    from speaker_lab.duplex.runner import run_offline
    root, _ = setup_run(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(config.read_text().replace('model="fixture"', 'model="Qwen/Qwen3-ASR-0.6B"')
                      .replace('backend="duplex_fixture_backend:Backend"', 'backend="qwen"'))
    protocol = tmp_path / "unpinned"
    freeze_protocol(tmp_path / "manifest.jsonl", root, config, protocol)
    with pytest.raises(ValueError, match="local snapshot"):
        run_offline(protocol, root, tmp_path / "no-network-run")


def test_retry_keeps_original_failed_worker_log(tmp_path, monkeypatch):
    from speaker_lab.duplex.runner import run_offline
    root, protocol = setup_run(tmp_path)
    plugin = tmp_path / "plugins"
    plugin.mkdir()
    module = plugin / "duplex_fixture_backend.py"
    module.write_text("class Backend:\n def run(self,payload,options): raise ValueError('initial model failure')\n")
    monkeypatch.setenv("PYTHONPATH", str(plugin))
    output = tmp_path / "retry-run"
    assert run_offline(protocol, root, output)["status"] == "incomplete"
    first_log = next(output.rglob("worker.log"))
    original = first_log.read_bytes()
    module.write_text("class Backend:\n def run(self,payload,options):\n  return {'segments':[{'job_id':j['id'],'text':'recovered'} for j in payload['jobs']]}\n")
    assert run_offline(protocol, root, output, resume=True)["status"] == "complete"
    assert first_log.read_bytes() == original
    assert len(list(output.rglob("worker.log"))) == 2
