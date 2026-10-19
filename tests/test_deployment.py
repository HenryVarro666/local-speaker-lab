"""Offline deployment orchestration checks; no system or GPU changes are made."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deployment_script", ROOT / "scripts/deploy_and_accept.py")
deployment_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment_script)


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    return deployment_script.Deployment(project, tmp_path / "logs")


def read_report(log_dir):
    return json.loads((log_dir / "deployment-report.json").read_text(encoding="utf-8"))


def test_main_stops_at_first_failure_and_preserves_redacted_report(tmp_path, monkeypatch, capsys):
    token = "a-secret-token-without-an-hf-prefix"
    monkeypatch.setenv("HF_TOKEN", token)
    executed = []

    def execute(self, audio, install_driver=False):
        def fail():
            executed.append("first")
            self.emit("worker received " + token)
            raise RuntimeError("worker failed with " + token)

        self.step("first", fail)
        self.step("later", lambda: executed.append("later"))

    monkeypatch.setattr(deployment_script.Deployment, "execute", execute)
    logs = tmp_path / "logs"
    code = deployment_script.main(["--audio", str(tmp_path / "input.wav"), "--log-dir", str(logs)])
    assert code == 1
    assert executed == ["first"]
    report = read_report(logs)
    assert report["status"] == "FAIL"
    assert [step["status"] for step in report["steps"]] == ["FAIL"]
    assert "[REDACTED]" in report["error"]
    assert "[REDACTED]" in report["steps"][0]["error"]
    assert report["finished_at"]
    assert token not in capsys.readouterr().out
    for name in ("deployment.log", "01-first.log", "deployment-report.md", "deployment-report.json"):
        content = (logs / name).read_text(encoding="utf-8")
        assert token not in content
        assert "[REDACTED]" in content


@pytest.mark.parametrize("error, status, code", [
    (deployment_script.RebootRequired("Reboot before retrying"), "REBOOT_REQUIRED", 20),
    (KeyboardInterrupt(), "FAIL", 130),
])
def test_main_reports_reboot_or_interrupt_without_running_later_steps(tmp_path, monkeypatch, error, status, code):
    executed = []

    def execute(self, audio, install_driver=False):
        def stop():
            executed.append("stop")
            raise error
        self.step("stop", stop)
        executed.append("later")

    monkeypatch.setattr(deployment_script.Deployment, "execute", execute)
    logs = tmp_path / "logs"
    assert deployment_script.main(["--audio", "input.wav", "--log-dir", str(logs)]) == code
    assert executed == ["stop"]
    assert read_report(logs)["status"] == status


def test_main_sigterm_writes_failure_report_and_restores_signal_handler(tmp_path, monkeypatch):
    executed = []
    previous_handler = deployment_script.signal.getsignal(deployment_script.signal.SIGTERM)

    def execute(self, audio, install_driver=False):
        self.step("interrupted", lambda: deployment_script.signal.raise_signal(deployment_script.signal.SIGTERM))
        executed.append("later")

    monkeypatch.setattr(deployment_script.Deployment, "execute", execute)
    logs = tmp_path / "logs"
    assert deployment_script.main(["--audio", "input.wav", "--log-dir", str(logs)]) == 130
    assert executed == []
    report = read_report(logs)
    assert report["status"] == "FAIL" and report["steps"][0]["status"] == "FAIL"
    assert "Interrupted" in report["error"]
    assert deployment_script.signal.getsignal(deployment_script.signal.SIGTERM) == previous_handler


def test_actual_child_stdout_stderr_and_command_exception_are_redacted(tmp_path, monkeypatch, capsys):
    token = "child-secret-exact-value"
    monkeypatch.setenv("HF_TOKEN", token)
    run = deployment_script.Deployment(tmp_path, tmp_path / "logs")
    program = (
        "import os, sys; "
        "print('stdout=' + os.environ['HF_TOKEN'], flush=True); "
        "print('stderr=' + os.environ['HF_TOKEN'], file=sys.stderr, flush=True); "
        "sys.exit(3)"
    )
    with pytest.raises(RuntimeError, match="Command exited 3") as caught:
        run.step("child", lambda: run.command([sys.executable, "-c", program], timeout=30, pass_token=True))
    run.finish("FAIL", str(caught.value))
    assert token not in capsys.readouterr().out
    for name in ("deployment.log", "01-child.log", "deployment-report.md", "deployment-report.json"):
        assert token not in (run.log_dir / name).read_text(encoding="utf-8")
    child_log = (run.log_dir / "01-child.log").read_text(encoding="utf-8")
    assert "stdout=[REDACTED]" in child_log
    assert "stderr=[REDACTED]" in child_log
    assert read_report(run.log_dir)["status"] == "FAIL"


def test_child_commands_do_not_receive_hf_token_unless_explicitly_required(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-for-model-downloads-only")
    run = deployment_script.Deployment(tmp_path, tmp_path / "logs")
    code, output = run.command([
        sys.executable, "-c", "import os; print('present' if 'HF_TOKEN' in os.environ else 'absent')"
    ], timeout=30)
    assert code == 0
    assert output == "absent"


def test_posix_commands_keep_controlling_tty_and_use_independent_process_group(deployment, monkeypatch):
    launched = []

    def popen(args, **kwargs):
        launched.append((args, kwargs))
        return SimpleNamespace(stdout=io.StringIO("worker completed\n"), wait=lambda timeout: 0)

    # Replace this module's os reference, not the host's os.name: pathlib must
    # keep using the actual host platform while POSIX launch options are tested.
    monkeypatch.setattr(deployment_script, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(deployment_script.subprocess, "Popen", popen)
    assert deployment.command(["fake-worker"], timeout=5) == (0, "worker completed")
    options = launched[0][1]
    assert options["process_group"] == 0
    assert "start_new_session" not in options


def test_posix_terminate_kills_surviving_group_after_parent_has_already_exited(monkeypatch):
    killed, waited = [], []
    process = SimpleNamespace(pid=42000, poll=lambda: 0,
                              wait=lambda timeout: waited.append(timeout) or 0)
    monkeypatch.setattr(deployment_script, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: killed.append((pid, sig))))
    monkeypatch.setattr(deployment_script, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    deployment_script.Deployment.terminate(process)
    assert killed == [(42000, 15), (42000, 9)]
    assert waited == [5, 5]


def test_sudo_authentication_keeps_prompt_on_tty_and_captured_command_cannot_prompt(deployment, monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(("validate", args, kwargs))
        return SimpleNamespace(returncode=0)

    def popen(args, **kwargs):
        calls.append(("command", args, kwargs))
        return SimpleNamespace(stdout=io.StringIO("enabled\n"), wait=lambda timeout: 0)

    deployment.env["HF_TOKEN"] = "private-token"
    monkeypatch.setattr(deployment_script.subprocess, "run", run)
    monkeypatch.setattr(deployment_script.subprocess, "Popen", popen)
    deployment.command(["sudo", "systemctl", "enable", "--now", "ollama"], timeout=5)
    assert [call[0] for call in calls] == ["validate", "command"]
    assert calls[0][1] == ["sudo", "-v"]
    assert "stdout" not in calls[0][2] and "stdin" not in calls[0][2]
    assert calls[1][1] == ["sudo", "-n", "systemctl", "enable", "--now", "ollama"]
    assert all("HF_TOKEN" not in call[2]["env"] for call in calls)


def test_existing_log_directory_report_is_never_overwritten(deployment):
    report_path = deployment.log_dir / "deployment-report.json"
    before = report_path.read_bytes()
    with pytest.raises(RuntimeError, match="new --log-dir"):
        deployment_script.Deployment(deployment.root, deployment.log_dir)
    assert report_path.read_bytes() == before


@pytest.mark.parametrize("package, branch", [
    ("nvidia-driver-580-open", "580-open"),
    ("nvidia-driver-580-server-open", "580-server-open"),
])
def test_driver_installs_only_recommended_open_variant_and_waits_for_reboot(deployment, monkeypatch, package, branch):
    boot = ["boot-before-install"]
    commands = []
    available_smi = [False]

    def proc_path(value):
        if str(value) == "/proc/sys/kernel/random/boot_id":
            return SimpleNamespace(read_text=lambda encoding: boot[0])
        if str(value) == "/proc/driver/nvidia/version":
            return SimpleNamespace(read_text=lambda encoding: "NVIDIA UNIX Open Kernel Module 580.95.05")
        return Path(value)

    def command(args, **kwargs):
        commands.append(args)
        if args == ["ubuntu-drivers", "devices"]:
            return 0, (
                "driver : nvidia-driver-570 - distro non-free\n"
                f"driver : {package} - distro non-free recommended\n"
            )
        if args[0] == "nvidia-smi":
            return 0, "0, NVIDIA GeForce RTX 5080, 580.95.05, 16303"
        if args[0] == "dpkg-query":
            assert args[-1] == package
            return 0, "installed"
        assert args == ["sudo", "ubuntu-drivers", "install", "nvidia:" + branch]
        return 0, "installed"

    monkeypatch.setattr(deployment_script, "Path", proc_path)
    monkeypatch.setattr(deployment_script.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if available_smi[0] else None)
    monkeypatch.setattr(deployment, "command", command)
    with pytest.raises(deployment_script.RebootRequired, match="installation completed"):
        deployment.driver_check(install_driver=True)
    assert commands == [["ubuntu-drivers", "devices"], ["sudo", "ubuntu-drivers", "install", "nvidia:" + branch],
                        ["dpkg-query", "-W", "-f=${db:Status-Status}", package]]
    marker = deployment.root / ".cache/deployment-driver-reboot.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["package"] == package

    with pytest.raises(deployment_script.RebootRequired, match="will not be installed again"):
        deployment.driver_check(install_driver=True)
    assert len(commands) == 3

    boot[0] = "boot-after-install"
    available_smi[0] = True
    deployment.driver_check(install_driver=True)
    assert not marker.exists()
    assert deployment.report["evidence"]["gpu"]["name"] == "NVIDIA GeForce RTX 5080"
    assert len(commands) == 4


@pytest.mark.parametrize("devices, message", [
    ("driver : nvidia-driver-580 - distro non-free recommended", "unambiguous recommended"),
    ("driver : nvidia-driver-565-open - distro non-free recommended", "too old"),
    ("driver : nvidia-driver-580-open recommended\ndriver : nvidia-driver-590-open recommended", "unambiguous recommended"),
])
def test_driver_refuses_proprietary_old_or_ambiguous_recommendation(deployment, monkeypatch, devices, message):
    commands = []
    monkeypatch.setattr(deployment_script, "Path", lambda value: SimpleNamespace(read_text=lambda encoding: "boot-id"))
    monkeypatch.setattr(deployment_script.shutil, "which", lambda name: None)

    def command(args, **kwargs):
        commands.append(args)
        return 0, devices

    monkeypatch.setattr(deployment, "command", command)
    with pytest.raises(RuntimeError, match=message):
        deployment.driver_check(install_driver=True)
    assert commands == [["ubuntu-drivers", "devices"]]
    assert not (deployment.root / ".cache/deployment-driver-reboot.json").exists()


def test_ollama_success_requires_both_gpu_evidence_and_unloads(deployment, monkeypatch):
    deployment.model = "qwen3:8b"
    commands, requests = [], []
    stopped = False

    def request(path, data=None, timeout=20):
        requests.append((path, data))
        if path == "/api/chat":
            return {"done": True, "message": {"content": "正常"}}
        assert path == "/api/ps"
        return {"models": [] if stopped else [{"name": "qwen3:8b", "size": 6000000000, "size_vram": 6000000000}]}

    def command(args, **kwargs):
        nonlocal stopped
        commands.append(args)
        if args == ["ollama", "ps"]:
            return 0, "NAME ID SIZE PROCESSOR CONTEXT UNTIL\nqwen3:8b abc 6 GB 100% GPU 8192 5 minutes"
        assert args == ["ollama", "stop", "qwen3:8b"]
        stopped = True
        return 0, ""

    monkeypatch.setattr(deployment, "request", request)
    monkeypatch.setattr(deployment, "command", command)
    deployment.step("ollama-gpu", deployment.ollama_gpu)
    assert deployment.ollama_test_loaded is False
    assert commands == [["ollama", "ps"], ["ollama", "stop", "qwen3:8b"]]
    chat = requests[0][1]
    assert chat["stream"] is False and chat["think"] is False
    assert chat["options"]["num_ctx"] == 8192
    assert deployment.report["evidence"]["ollama"]["generation_done"] is True
    assert len([item for item in requests if item[0] == "/api/ps"]) == 2


@pytest.mark.parametrize("table, resident", [
    ("qwen3:8b abc 6 GB 40%/60% CPU/GPU 8192 5m", [{"name": "qwen3:8b", "size": 6000000000, "size_vram": 4000000000}]),
    ("qwen3:8b abc 6 GB 100% GPU 8192 5m", [{"name": "qwen3:8b", "size": 6000000000, "size_vram": 0}]),
    ("qwen3:8b abc 6 GB 100% GPU 8192 5m", [{"name": "qwen3:8b", "size": 6000000000, "size_vram": 4000000000}]),
    ("qwen3:8b abc 6 GB 100% GPU 8192 5m", [{"name": "qwen3:8b", "size": 6000000000, "size_vram": 7000000000}]),
    ("qwen3:8b abc 6 GB 100% GPU 8192 5m", [{"name": "qwen3:8b", "size": "6000000000", "size_vram": "6000000000"}]),
    ("qwen3:8b abc 6 GB 100% GPU 8192 5m", []),
])
def test_partial_gpu_or_missing_api_proof_fails_and_cleans_up(tmp_path, monkeypatch, table, resident):
    commands, completed = [], []

    def execute(self, audio, install_driver=False):
        self.model = "qwen3:8b"
        self.step("ollama-gpu", self.ollama_gpu)
        completed.append("later")

    def request(self, path, data=None, timeout=20):
        if path == "/api/chat":
            return {"done": True, "message": {"content": "正常"}}
        assert path == "/api/ps"
        return {"models": resident}

    def command(self, args, **kwargs):
        commands.append(args)
        return (0, table) if args == ["ollama", "ps"] else (0, "")

    monkeypatch.setattr(deployment_script.Deployment, "execute", execute)
    monkeypatch.setattr(deployment_script.Deployment, "request", request)
    monkeypatch.setattr(deployment_script.Deployment, "command", command)
    logs = tmp_path / "logs"
    assert deployment_script.main(["--audio", "input.wav", "--log-dir", str(logs)]) == 1
    assert completed == []
    assert commands[-1] == ["ollama", "stop", "qwen3:8b"]
    report = read_report(logs)
    assert report["status"] == "FAIL"
    assert report["steps"][0]["status"] == "FAIL"
    assert "100% GPU" in report["error"]


def test_ollama_request_error_still_attempts_cleanup_and_reports_cleanup_error(tmp_path, monkeypatch):
    token = "failure-cleanup-token"
    monkeypatch.setenv("HF_TOKEN", token)
    commands = []

    def execute(self, audio, install_driver=False):
        self.model = "qwen3:8b"
        self.step("ollama-gpu", self.ollama_gpu)

    def request(self, *args, **kwargs):
        raise OSError("server disconnected")

    def command(self, args, **kwargs):
        commands.append(args)
        raise RuntimeError("cleanup failed " + token)

    monkeypatch.setattr(deployment_script.Deployment, "execute", execute)
    monkeypatch.setattr(deployment_script.Deployment, "request", request)
    monkeypatch.setattr(deployment_script.Deployment, "command", command)
    logs = tmp_path / "logs"
    assert deployment_script.main(["--audio", "input.wav", "--log-dir", str(logs)]) == 1
    assert commands == [["ollama", "stop", "qwen3:8b"]]
    report = read_report(logs)
    assert "server disconnected" in report["error"]
    assert report["cleanup_warning"] == "cleanup failed [REDACTED]"


@pytest.mark.parametrize("seconds, passes", [(9.9, False), (10, True), (60, True), (120, True), (120.1, False)])
def test_prepare_audio_validates_whole_recording_duration_without_truncation(deployment, monkeypatch, seconds, passes):
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        with wave.open(str(args[-1]), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * round(seconds * 16000))
        return 0, ""

    monkeypatch.setattr(deployment, "command", command)
    if passes:
        deployment.prepare_audio(Path("recording with spaces.wav"))
        assert deployment.report["evidence"]["normalized_audio"]["duration_seconds"] == seconds
    else:
        with pytest.raises(RuntimeError, match="10–120s"):
            deployment.prepare_audio(Path("recording with spaces.wav"))
        assert "normalized_audio" not in deployment.report["evidence"]
    assert "-t" not in commands[0]
    assert commands[0][commands[0].index("-i") + 1] == Path("recording with spaces.wav")
