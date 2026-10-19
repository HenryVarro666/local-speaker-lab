"""Exercise credential handling through real local worker subprocesses, without models."""
import io
import json
import os
from pathlib import Path
import sys

import pytest

from speaker_lab.redaction import Redactor, RedactingStream
from speaker_lab.runner import WorkerRunner, write_json
from speaker_lab import worker


ROOT = Path(__file__).resolve().parents[1]


def test_nested_json_is_redacted_before_json_escaping(tmp_path, monkeypatch):
    token = 'secret-with-"quotes"-and-\\slashes'
    other = "custom-secret-without-prefix"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("PRIVATE_MODEL_TOKEN", other)
    redactor = Redactor({"nested": {"token_env": "PRIVATE_MODEL_TOKEN"}})
    path = tmp_path / "diagnostic.json"
    write_json(path, {"nested": [token, {"message": "failure " + other}], "count": 3,
                      "ordinary_transcript": "会议开始，讨论明天的安排。", "unknown_hf": "hf_123456789abc"},
               redactor=redactor)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["nested"] == ["[REDACTED]", {"message": "failure [REDACTED]"}]
    assert saved["count"] == 3
    assert saved["ordinary_transcript"] == "会议开始，讨论明天的安排。"
    assert saved["unknown_hf"] == "[REDACTED]"
    assert token not in path.read_text(encoding="utf-8")
    assert other not in path.read_text(encoding="utf-8")


def test_stream_withholds_split_token_even_when_backend_flushes(monkeypatch):
    token = "my-private-token-without-hf-prefix"
    monkeypatch.setenv("HF_TOKEN", token)
    output = io.StringIO()
    stream = RedactingStream(output, Redactor())
    stream.write("failure " + token[:9])
    stream.flush()
    assert output.getvalue() == ""
    stream.write(token[9:] + "\n")
    assert output.getvalue() == "failure [REDACTED]\n"
    stream.write("tail " + token)
    stream.finish()
    assert output.getvalue() == "failure [REDACTED]\ntail [REDACTED]"


def test_worker_stdout_stderr_response_and_traceback_are_redacted(tmp_path, monkeypatch, capsys):
    token = 'token-with-"quote"-and-\\slash'
    custom = "alternate-secret-no-hf-prefix"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("ALTERNATE_HF_TOKEN", custom)
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"options": {"token_env": "ALTERNATE_HF_TOKEN"}}), encoding="utf-8")

    def fail(request):
        sys.stdout.write(token[:6])
        sys.stdout.flush()
        sys.stdout.write(token[6:] + "\n")
        print("stderr " + custom, file=sys.stderr)
        raise RuntimeError("error " + token + " and " + custom)

    monkeypatch.setattr(worker, "execute", fail)
    assert worker.main(["--request", str(request), "--response", str(response)]) == 1
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
    assert custom not in captured.out + captured.err
    assert "[REDACTED]" in captured.out and "[REDACTED]" in captured.err
    result = json.loads(response.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["error"] == "RuntimeError: error [REDACTED] and [REDACTED]"
    assert token not in result["traceback"] and custom not in result["traceback"]


def make_fake_runner(tmp_path, monkeypatch, *, timeout=10, body=None):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "credential_backend.py").write_text(
        "import os, sys, time\n"
        "class Backend:\n"
        "    def run(self, payload, options):\n" +
        (body or
         "        print('python stdout ' + os.environ['HF_TOKEN'], flush=True)\n"
         "        os.write(2, ('native stderr ' + os.environ['MODEL_PRIVATE_TOKEN'] + '\\n').encode())\n"
         "        raise RuntimeError('download failed ' + os.environ['MODEL_PRIVATE_TOKEN'])\n"),
        encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(plugin_dir) + os.pathsep + str(ROOT / "src"))
    # On Windows a venv executable is a launcher; use its actual interpreter so
    # timeout tests kill the worker itself, as the Ubuntu venv executable does.
    python = getattr(sys, "_base_executable", sys.executable)
    config = {"_root": str(ROOT), "workers": {"diarize_python": python, "timeout_seconds": timeout},
              "diarize": {"backend": "credential_backend:Backend", "token_env": "MODEL_PRIVATE_TOKEN"}}
    return WorkerRunner(config, tmp_path / "run")


def test_real_worker_filters_native_output_before_it_reaches_log(tmp_path, monkeypatch):
    token, custom = "primary-private-token", "custom-private-token"
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("MODEL_PRIVATE_TOKEN", custom)
    runner = make_fake_runner(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match=r"download failed \[REDACTED\]") as caught:
        runner.run("diarize", {})
    assert token not in str(caught.value) and custom not in str(caught.value)
    stage = runner.run_dir / "work/diarize"
    log = (stage / "worker.log").read_text(encoding="utf-8")
    assert "python stdout [REDACTED]" in log
    assert "native stderr [REDACTED]" in log
    for name in ("request.json", "response.json", "worker.log"):
        content = (stage / name).read_text(encoding="utf-8")
        assert token not in content and custom not in content
    assert json.loads((stage / "response.json").read_text(encoding="utf-8"))["ok"] is False


def test_real_worker_timeout_preserves_redacted_log_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "timed-out-private-token")
    runner = make_fake_runner(tmp_path, monkeypatch, timeout=2, body=(
        "        os.write(2, ('before sleep ' + os.environ['HF_TOKEN']).encode())\n"
        "        time.sleep(30)\n"
        "        return {}\n"
    ))
    with pytest.raises(RuntimeError, match="diarize timed out"):
        runner.run("diarize", {})
    log = (runner.run_dir / "work/diarize/worker.log").read_text(encoding="utf-8")
    assert "timed-out-private-token" not in log
    assert "before sleep [REDACTED]" in log


def test_pipeline_failure_artifact_and_public_exception_hide_custom_token(tmp_path, monkeypatch):
    from speaker_lab import pipeline
    from speaker_lab.config import load_config

    token = 'custom-private-"quoted"-\\token'
    monkeypatch.setenv("AUDIO_MODEL_TOKEN", token)
    config = load_config(ROOT / "configs/default.toml")
    config["pipeline"]["output_dir"] = str(tmp_path / "runs")
    config["diarize"]["token_env"] = "AUDIO_MODEL_TOKEN"
    monkeypatch.setattr(pipeline, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "normalize_audio", lambda *args: pipeline.np.ones(16000, dtype="float32"))

    class FailingRunner:
        def __init__(self, config, run_dir):
            pass

        def run(self, stage, payload):
            raise RuntimeError("provider rejected " + token)

    monkeypatch.setattr(pipeline, "WorkerRunner", FailingRunner)
    with pytest.raises(RuntimeError, match=r"provider rejected \[REDACTED\]") as caught:
        pipeline.run_pipeline("recording.wav", "config.toml", summarize=False)
    assert token not in str(caught.value)
    assert caught.value.__suppress_context__ is True
    paths = list((tmp_path / "runs").glob("*/error.json"))
    assert len(paths) == 1
    artifact = json.loads(paths[0].read_text(encoding="utf-8"))
    assert artifact["error"] == "provider rejected [REDACTED]"
