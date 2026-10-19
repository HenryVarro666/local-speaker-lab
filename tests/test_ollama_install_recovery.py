"""Offline Ollama partial-install recovery and noninteractive sudo checks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ollama_install_recovery", ROOT / "scripts/deploy_and_accept.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    run.model, run.ollama_url = "qwen3:8b", "http://127.0.0.1:11434"
    observed = SimpleNamespace(commands=[], shim_paths=[], installed=False,
                               state="loaded", state_code=0, binary=True,
                               installer_error=None)
    monkeypatch.setattr(deploy.shutil, "which", lambda command: "/usr/bin/ollama" if observed.binary else None)

    def command(args, **kwargs):
        args = list(map(str, args))
        observed.commands.append((args, kwargs))
        if args == ["systemctl", "show", "ollama.service", "-p", "LoadState", "--value"]:
            return observed.state_code, observed.state
        if args[0] == "sh":
            observed.installed = True
            assert kwargs["sudo_keepalive"] is True
            assert not kwargs.get("pass_token")
            shim = Path(kwargs["env_overrides"]["PATH"].split(os.pathsep)[0]) / "sudo"
            observed.shim_paths.append(shim)
            assert shim.read_text(encoding="utf-8") == '#!/bin/sh\nexec /usr/bin/sudo -n "$@"\n'
            assert "HF_TOKEN" not in shim.read_text(encoding="utf-8")
            if observed.installer_error:
                raise RuntimeError(observed.installer_error)
        return 0, ""

    def request(path, data=None, timeout=20):
        if path == "/api/tags":
            return {"models": [{"name": "qwen3:8b"}]}
        if path == "/api/ps":
            return {"models": []}
        if path == "/api/show":
            return {}
        raise AssertionError(path)

    monkeypatch.setattr(run, "command", command)
    monkeypatch.setattr(run, "request", request)
    return run, observed


@pytest.mark.parametrize("binary, state, state_code", [
    (False, "not-found", 0),
    (True, "not-found", 0),
    (True, "not-found", 1),
    (False, "loaded", 0),
])
def test_missing_binary_or_service_repairs_official_install_and_removes_sudo_shim(installation, binary, state, state_code):
    run, observed = installation
    observed.binary, observed.state, observed.state_code = binary, state, state_code
    run.ollama_setup()
    assert observed.installed
    assert len(observed.shim_paths) == 1 and not observed.shim_paths[0].exists()
    assert any(args[:4] == ["sudo", "systemctl", "enable", "--now"] for args, _ in observed.commands)


def test_existing_binary_and_loaded_service_reuse_installation(installation):
    run, observed = installation
    run.ollama_setup()
    assert not observed.installed
    assert not any(args[0] in {"curl", "sh"} for args, _ in observed.commands)
    assert observed.commands[0][1]["check"] is False


@pytest.mark.parametrize("state, state_code", [("masked", 0), ("bad-setting", 0), ("error", 1), ("", 1), ("loaded", 1)])
def test_masked_invalid_or_inaccessible_service_is_never_overwritten(installation, state, state_code):
    run, observed = installation
    observed.binary, observed.state, observed.state_code = False, state, state_code
    with pytest.raises(RuntimeError, match="installer was not run"):
        run.ollama_setup()
    assert len(observed.commands) == 1
    assert not observed.installed


def test_installer_failure_removes_shim_and_report_hides_token(installation):
    run, observed = installation
    token = "opaque-install-credential"
    run.secrets.append(token)
    run.env["HF_TOKEN"] = token
    observed.state, observed.installer_error = "not-found", "installer failed " + token
    with pytest.raises(RuntimeError) as caught:
        run.step("ollama-setup", run.ollama_setup)
    run.finish("FAIL", str(caught.value))
    assert observed.shim_paths and not observed.shim_paths[0].exists()
    report = json.loads((run.log_dir / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["steps"][0]["status"] == "FAIL"
    assert report["error"] == "installer failed [REDACTED]"


def test_keepalive_uses_absolute_noninteractive_sudo_and_omits_token(tmp_path, monkeypatch, capsys):
    token = "keepalive-private-credential"
    monkeypatch.setenv("HF_TOKEN", token)
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    calls = []

    def subprocess_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="sudo rejected " + token)

    monkeypatch.setattr(deploy.subprocess, "run", subprocess_run)
    with pytest.raises(RuntimeError, match="could not be refreshed noninteractively"):
        run.sudo_refresh()
    assert calls[0][0] == ["/usr/bin/sudo", "-n", "-v"]
    assert "HF_TOKEN" not in calls[0][1]["env"]
    assert calls[0][1]["timeout"] == 10
    assert token not in capsys.readouterr().out
    assert token not in (run.log_dir / "deployment.log").read_text(encoding="utf-8")


def test_keepalive_failure_terminates_real_child_early_and_preserves_failure_report(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-command-token")
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    monkeypatch.setattr(deploy, "SUDO_REFRESH_SECONDS", 0.02)
    authorized, refreshed, later, children = [], [], [], []
    monkeypatch.setattr(deploy.Deployment, "sudo_validate", lambda self: authorized.append(True))
    real_popen = deploy.subprocess.Popen

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(deploy.subprocess, "Popen", popen)

    def refresh(self):
        refreshed.append(True)
        raise RuntimeError("sudo authorization expired")

    def execute(self, audio, install_driver=False):
        self.step("installer", lambda: self.command([
            getattr(sys, "_base_executable", sys.executable), "-c",
            "import time; print('installer started', flush=True); time.sleep(30)"
        ], timeout=60, sudo_keepalive=True, env_overrides={"HF_TOKEN": "private-command-token"}))
        later.append(True)

    monkeypatch.setattr(deploy.Deployment, "sudo_refresh", refresh)
    monkeypatch.setattr(deploy.Deployment, "execute", execute)
    started = time.monotonic()
    code = deploy.main(["--audio", "recording.wav", "--log-dir", str(tmp_path / "logs")])
    assert time.monotonic() - started < 10
    assert code == 1 and authorized == [True] and refreshed == [True] and not later
    assert len(children) == 1 and children[0].poll() is not None
    report = json.loads((tmp_path / "logs/deployment-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["steps"][0]["status"] == "FAIL"
    assert "sudo authorization expired" in report["error"]


def test_command_environment_override_cannot_reintroduce_token_without_opt_in(tmp_path, monkeypatch):
    token = "child-must-not-receive-credential"
    monkeypatch.setenv("HF_TOKEN", token)
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    _, output = run.command([sys.executable, "-c", "import os; print('present' if 'HF_TOKEN' in os.environ else 'absent')"],
                            timeout=30, env_overrides={"HF_TOKEN": token})
    assert output == "absent"


def test_keepalive_success_refreshes_during_real_child_without_changing_output(tmp_path, monkeypatch):
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    authorized, refreshed = [], []
    monkeypatch.setattr(deploy, "SUDO_REFRESH_SECONDS", 0.02)
    monkeypatch.setattr(run, "sudo_validate", lambda: authorized.append(True))
    monkeypatch.setattr(run, "sudo_refresh", lambda: refreshed.append(True))
    code, output = run.command([
        getattr(sys, "_base_executable", sys.executable), "-c",
        "import time; print('start', flush=True); time.sleep(0.4); print('done', flush=True)"
    ], timeout=10, sudo_keepalive=True)
    assert code == 0 and output == "start\ndone"
    assert authorized == [True] and refreshed
