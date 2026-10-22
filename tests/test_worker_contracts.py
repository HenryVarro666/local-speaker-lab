"""Real process-boundary failures and plugin contracts, without model imports."""
import json
import os
from pathlib import Path
import sys

import pytest

from speaker_lab import worker
from speaker_lab.backends import load_backend
from speaker_lab.runner import WorkerRunner


ROOT = Path(__file__).resolve().parents[1]


def make_runner(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "contract_backend.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "class Backend:\n"
        "    def run(self, payload, options):\n"
        "        if payload.get('exit_early'):\n"
        "            os._exit(0)\n"
        "        if 'response' in payload:\n"
        "            path = Path(sys.argv[sys.argv.index('--response') + 1])\n"
        "            path.write_text(payload['response'], encoding='utf-8')\n"
        "            os._exit(0)\n"
        "        return {'value': payload.get('value', 1)}\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(plugin_dir) + os.pathsep + str(ROOT / "src"))
    python = getattr(sys, "_base_executable", sys.executable)
    config = {"_root": str(ROOT), "workers": {"diarize_python": python, "timeout_seconds": 10},
              "diarize": {"backend": "contract_backend:Backend"}}
    return WorkerRunner(config, tmp_path / "run")


def test_repeated_stage_cannot_reuse_stale_success(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch)
    assert runner.run("diarize", {"value": 3}) == {"value": 3}
    with pytest.raises(RuntimeError, match="code 0 without a response"):
        runner.run("diarize", {"exit_early": True})
    assert not (runner.run_dir / "work/diarize/response.json").exists()


def test_worker_preserves_virtualenv_interpreter_and_its_dependencies(tmp_path):
    """Resolving a venv's Python symlink must not select the base environment."""
    import subprocess
    import venv

    if os.name == "nt":
        pytest.skip("POSIX virtualenv interpreter symlink regression")
    environment = tmp_path / "model-environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    (site / "worker_environment_marker.py").write_text("VALUE = 73\n", encoding="utf-8")
    (site / "environment_backend.py").write_text(
        "import sys\nfrom worker_environment_marker import VALUE\n"
        "class Backend:\n"
        "    def run(self, payload, options):\n"
        "        return {'value': VALUE, 'prefix': sys.prefix}\n", encoding="utf-8")
    runner = WorkerRunner({
        "_root": str(ROOT),
        "workers": {"transcribe_python": str(python), "timeout_seconds": 15},
        "transcribe": {"backend": "environment_backend:Backend"},
    }, tmp_path / "venv-run")
    assert runner.run("transcribe", {}) == {"value": 73, "prefix": str(environment)}


@pytest.mark.parametrize("response,description", [
    ("not json", "invalid JSON"),
    ('{"ok": true, "result": {"score": NaN}}', "invalid JSON"),
    ("[]", "response envelope"),
    ('{"ok": 1, "result": {}}', "response envelope"),
    ('{"ok": true, "result": []}', "result must be an object"),
    ('{"ok": false, "error": "plugin error"}', "plugin error"),
])
def test_invalid_worker_responses_include_stage_log(tmp_path, monkeypatch, response, description):
    runner = make_runner(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match=description) as caught:
        runner.run("diarize", {"response": response})
    assert str(runner.run_dir / "work/diarize/worker.log") in str(caught.value)


@pytest.mark.parametrize("message", [
    None, [], {}, {"stage": 1, "backend": "module:Class"},
    {"stage": "diarize", "backend": ""},
    {"stage": "diarize", "backend": ["module:Class"]},
    {"stage": "diarize", "backend": "module:Class", "payload": []},
    {"stage": "diarize", "backend": "module:Class", "options": None},
])
def test_invalid_request_fails_before_plugin_import(monkeypatch, message):
    monkeypatch.setattr(worker, "load_backend", lambda *_: pytest.fail("Invalid request loaded plugin"))
    with pytest.raises((ValueError, TypeError), match="Worker request"):
        worker.execute(message)


@pytest.mark.parametrize("name", [None, [], "", " :Class", ":Class", "module:", "module:Class:bad"])
def test_invalid_plugin_target_fails_before_import(monkeypatch, name):
    monkeypatch.setattr("speaker_lab.backends.import_module", lambda *_: pytest.fail("Invalid target imported module"))
    with pytest.raises(ValueError):
        load_backend("diarize", name)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "10"])
def test_invalid_worker_timeout_is_rejected_before_spawn(tmp_path, monkeypatch, timeout):
    runner = make_runner(tmp_path, monkeypatch)
    runner.config["workers"]["timeout_seconds"] = timeout
    monkeypatch.setattr("speaker_lab.runner.subprocess.Popen", lambda *_args, **_kwargs: pytest.fail("Invalid timeout spawned worker"))
    with pytest.raises(ValueError, match="positive finite"):
        runner.run("diarize", {})


def test_worker_invalid_request_is_serialized_as_failure(tmp_path):
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text("[]", encoding="utf-8")
    assert worker.main(["--request", str(request), "--response", str(response)]) == 1
    assert json.loads(response.read_text(encoding="utf-8"))["error"] == "TypeError: Worker request must be an object."
