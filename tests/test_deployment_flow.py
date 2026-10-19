"""Exercise complete deployment control flow with fake OS/GPU/network services.

These tests do not install packages, contact HF/Ollama, or claim real GPU quality.
"""
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deployment_flow", ROOT / "scripts/deploy_and_accept.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


@pytest.fixture
def simulated_machine(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "configs/default.toml").write_bytes((ROOT / "configs/default.toml").read_bytes())
    (root / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"fake input decoded by mocked ffmpeg")
    machine = SimpleNamespace(root=root, recording=recording, boot="boot-1", driver=True,
                              installed=False, ollama=False, model=False, resident=False,
                              calls=[], fail=None, token="hf_flowSecret123456789")
    # Patch this loaded module's sys binding, not global sys.version_info: the
    # simulated target is Ubuntu/Python 3.12 regardless of pytest's interpreter.
    machine.python = SimpleNamespace(version_info=(3, 12, 0, "final", 0),
                                     version="3.12.0 (simulated deployment target)",
                                     executable="/usr/bin/python3.12")
    monkeypatch.setattr(deploy, "sys", machine.python)
    monkeypatch.setattr(deploy, "ROOT", root)
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(deploy.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(deploy.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(deploy.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04"})
    monkeypatch.setattr(deploy.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def which(name):
        if name == "nvidia-smi" and not machine.driver or name == "ollama" and not machine.ollama:
            return None
        return "/usr/bin/" + name

    monkeypatch.setattr(deploy.shutil, "which", which)

    def proc_path(value):
        if str(value) == "/proc/sys/kernel/random/boot_id":
            return SimpleNamespace(read_text=lambda encoding: machine.boot)
        if str(value) == "/proc/driver/nvidia/version":
            return SimpleNamespace(read_text=lambda encoding: "NVIDIA UNIX Open Kernel Module 580.95.05")
        return Path(value)

    monkeypatch.setattr(deploy, "Path", proc_path)
    monkeypatch.setattr(deploy, "open", lambda *args, **kwargs: contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(deploy.getpass, "getpass", lambda prompt: machine.token)
    monkeypatch.setattr(deploy.Deployment, "sudo_validate", lambda self: machine.calls.append("sudo-validate"))

    def command(self, args, **kwargs):
        args = list(map(str, args))
        text = " ".join(args).replace("\\", "/")
        machine.calls.append(text)
        self.emit("$ " + text)
        if machine.fail and machine.fail in text:
            raise RuntimeError("Simulated failure: " + machine.token)
        if args[0] == "nvidia-smi":
            return 0, "0, NVIDIA GeForce RTX 5080, 580.95.05, 16303"
        if args == ["ubuntu-drivers", "devices"]:
            return 0, "driver : nvidia-driver-580-open - distro non-free recommended"
        if args[0] == "dpkg-query":
            return 0, "installed"
        if args[:3] == ["systemctl", "show", "ollama.service"]:
            return 0, "loaded" if machine.ollama else "not-found"
        if args[0] == "ffmpeg":
            with wave.open(args[-1], "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\0\0" * (16000 * 30))
        elif "scripts/install_ubuntu.sh" in text:
            for name in ("app", "diarize", "separate", "asr"):
                env = root / f".venvs/{name}"
                env.mkdir(parents=True, exist_ok=True)
                (env / "installed-versions.txt").write_text("mock-package==1.0\n", encoding="utf-8")
            cache = root / ".cache/downloaded-model"
            cache.parent.mkdir(exist_ok=True)
            if not cache.exists():
                cache.write_bytes(b"keep cached model")
            machine.installed = True
        elif args[0] == "curl":
            Path(args[-1]).write_text("# fake official installer", encoding="utf-8")
        elif args[0] == "sh":
            machine.ollama = True
        elif args[:2] == ["ollama", "pull"]:
            machine.model = True
        elif args[:2] == ["ollama", "ps"]:
            return 0, "NAME ID SIZE PROCESSOR\nqwen3:8b abc 6 GB 100% GPU"
        elif args[:2] == ["ollama", "stop"]:
            machine.resident = False
        elif "scripts/acceptance.py" in text:
            log_dir = Path(args[args.index("--log-dir") + 1])
            log_dir.mkdir()
            (log_dir / "acceptance-report.json").write_text(json.dumps({"automatic_checks": "PASS", "separation_jobs": 1, "run_dir": str(root / "runs/mock")}), encoding="utf-8")
        return 0, ""

    def request(self, path, data=None, timeout=20):
        machine.calls.append("HTTP " + path)
        if path == "/api/tags":
            return {"models": [{"name": "qwen3:8b", "digest": "same-installed-digest"}] if machine.model else []}
        if path == "/api/show":
            return {}
        if path == "/api/chat":
            machine.resident = True
            return {"done": True, "message": {"content": "正常"}}
        if path == "/api/ps":
            return {"models": [{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}] if machine.resident else []}
        raise AssertionError(path)

    monkeypatch.setattr(deploy.Deployment, "command", command)
    monkeypatch.setattr(deploy.Deployment, "request", request)

    def run(name, driver=False):
        log_dir = tmp_path / name
        args = ["--audio", str(recording), "--log-dir", str(log_dir)]
        if driver:
            args.append("--install-driver")
        code = deploy.main(args)
        return code, json.loads((log_dir / "deployment-report.json").read_text(encoding="utf-8"))

    machine.run = run
    return machine


def test_fresh_install_then_repeat_reuses_model_cache_and_preserves_old_report(simulated_machine):
    machine = simulated_machine
    code, first = machine.run("first")
    assert code == 0 and first["status"] == "PASS"
    assert first["evidence"]["python"] == "3.12.0"
    assert machine.installed and machine.ollama and machine.model
    first_calls = list(machine.calls)
    assert sum(line.startswith("ollama pull") for line in first_calls) == 1
    assert any(line.startswith("curl ") for line in first_calls)
    first_file = Path(first["log_dir"]) / "deployment-report.json"
    before = first_file.read_bytes()
    cached = machine.root / ".cache/downloaded-model"
    machine.calls.clear()
    code, repeat = machine.run("repeat")
    assert code == 0 and repeat["status"] == "PASS"
    assert cached.read_bytes() == b"keep cached model"
    assert first_file.read_bytes() == before
    assert not any(line.startswith(("ollama pull", "curl ", "sh ")) for line in machine.calls)
    assert any("scripts/install_ubuntu.sh" in line for line in machine.calls)
    assert [item["name"] for item in repeat["steps"]][-3:] == ["doctor", "clearvoice-cuda", "recording-acceptance"]
    assert repeat["quality_status"] == "PENDING_MANUAL_REVIEW"


def test_unsupported_simulated_python311_stops_before_system_work(simulated_machine):
    machine = simulated_machine
    machine.python.version_info = (3, 11, 0, "final", 0)
    machine.python.version = "3.11.0 (unsupported simulated target)"
    code, report = machine.run("unsupported-python")
    assert code == 1 and report["status"] == "FAIL"
    assert [step["name"] for step in report["steps"]] == ["platform"]
    assert "Run with Python 3.12" in report["error"]
    assert not machine.calls
    assert not machine.installed and not machine.ollama and not machine.model


def test_driver_install_same_boot_stops_then_new_boot_runs_full_acceptance(simulated_machine):
    machine = simulated_machine
    machine.driver = False
    code, installed = machine.run("install-driver", driver=True)
    assert code == 20 and installed["status"] == "REBOOT_REQUIRED"
    assert not machine.installed and not any("HF_TOKEN" in line for line in machine.calls)
    install = "sudo ubuntu-drivers install nvidia:580-open"
    assert machine.calls.count(install) == 1
    code, pending = machine.run("same-boot", driver=True)
    assert code == 20 and pending["status"] == "REBOOT_REQUIRED"
    assert machine.calls.count(install) == 1
    machine.boot, machine.driver = "boot-2", True
    code, recovered = machine.run("after-reboot", driver=True)
    assert code == 0 and recovered["status"] == "PASS"
    assert machine.calls.count(install) == 1
    assert not (machine.root / ".cache/deployment-driver-reboot.json").exists()


def test_first_failure_stops_after_hf_access_and_rerun_recovers(simulated_machine):
    machine = simulated_machine
    machine.fail = "hf_hub_download"
    code, failed = machine.run("failed")
    assert code == 1 and failed["status"] == "FAIL"
    assert failed["steps"][-1]["name"] == "hf-access"
    assert not machine.ollama
    assert machine.token not in json.dumps(failed)
    for log in Path(failed["log_dir"]).rglob("*"):
        if log.is_file() and log.suffix in (".json", ".md", ".log"):
            assert machine.token not in log.read_text(encoding="utf-8")
    machine.fail = None
    code, recovered = machine.run("recovered")
    assert code == 0 and recovered["status"] == "PASS"


def test_escaped_secrets_and_paths_are_redacted_before_json_encoding(tmp_path, monkeypatch):
    token = 'opaque"\\credential'
    monkeypatch.setenv("HF_TOKEN", token)
    root = tmp_path / "project"
    run = deploy.Deployment(root, tmp_path / "logs")
    run.report["project_root"] = "/home/" + token
    run.report["evidence"]["nested"] = [{"error": token}]
    run.finish("FAIL", "failed " + token)
    report = json.loads((run.log_dir / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["evidence"]["nested"][0]["error"] == "[REDACTED]"
    for name in ("deployment-report.json", "deployment-report.md"):
        assert token not in (run.log_dir / name).read_text(encoding="utf-8")
        assert "[REDACTED]" in (run.log_dir / name).read_text(encoding="utf-8")


def test_getpass_echo_fallback_is_rejected_before_reading_token(tmp_path, monkeypatch):
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    monkeypatch.setattr(deploy, "open", lambda *args, **kwargs: contextlib.nullcontext(), raising=False)
    def unsafe_getpass(prompt):
        warnings.warn("cannot disable echo", deploy.getpass.GetPassWarning)
        pytest.fail("fallback would read an unhidden token")
    monkeypatch.setattr(deploy.getpass, "getpass", unsafe_getpass)
    with pytest.raises(RuntimeError, match="Cannot disable terminal echo"):
        run.token_input()


@pytest.mark.parametrize("kind", ["existing-report", "invalid-checksum", "not-a-directory", "missing-arguments"])
def test_initialization_errors_create_separate_failure_report_without_overwriting(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(deploy, "ROOT", tmp_path / "project")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    previous = None
    if kind == "existing-report":
        previous = b'{"status":"PASS","preserve":true}'
        (log_dir / "deployment-report.json").write_bytes(previous)
    elif kind == "invalid-checksum":
        (log_dir / "source-sha256.txt").write_text("", encoding="utf-8")
    elif kind == "not-a-directory":
        log_dir = tmp_path / "ordinary-file"
        log_dir.write_text("keep", encoding="utf-8")
    args = [] if kind == "missing-arguments" else ["--audio", "recording.wav", "--log-dir", str(log_dir)]
    assert deploy.main(args) != 0
    reports = list((tmp_path / "deployment_logs").glob("initialization-failure-*/deployment-report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["steps"][0]["name"] == "initialization"
    if previous:
        assert (log_dir / "deployment-report.json").read_bytes() == previous


def test_second_interrupt_during_cleanup_cannot_discard_failure_report(tmp_path, monkeypatch):
    def execute(self, audio, install_driver=False):
        self.ollama_test_loaded, self.model = True, "qwen3:8b"
        raise KeyboardInterrupt
    def cleanup(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(deploy.Deployment, "execute", execute)
    monkeypatch.setattr(deploy.Deployment, "command", cleanup)
    logs = tmp_path / "logs"
    assert deploy.main(["--audio", "recording.wav", "--log-dir", str(logs)]) == 130
    report = json.loads((logs / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["cleanup_warning"] == "KeyboardInterrupt"


@pytest.mark.parametrize("package_state", ["unpacked", "half-configured", "not-installed"])
def test_driver_cli_success_without_installed_package_does_not_mark_reboot(tmp_path, monkeypatch, package_state):
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    monkeypatch.setattr(deploy, "Path", lambda value: SimpleNamespace(read_text=lambda encoding: "boot-id"))
    monkeypatch.setattr(deploy.shutil, "which", lambda name: None)
    def command(self, args, **kwargs):
        if args == ["ubuntu-drivers", "devices"]:
            return 0, "driver : nvidia-driver-580-open - distro non-free recommended"
        if args[0] == "dpkg-query":
            return 0, package_state
        assert args == ["sudo", "ubuntu-drivers", "install", "nvidia:580-open"]
        return 0, "installer incorrectly returned success"
    monkeypatch.setattr(deploy.Deployment, "command", command)
    with pytest.raises(RuntimeError, match="not fully installed"):
        run.driver_check(install_driver=True)
    assert not (tmp_path / ".cache/deployment-driver-reboot.json").exists()
