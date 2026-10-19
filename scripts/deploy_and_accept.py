#!/usr/bin/env python3
"""Repeatable Ubuntu 24.04 / RTX 5080 deployment and real-recording acceptance.

Use deploy_ubuntu.sh for system packages and ZIP extraction. This module uses
only Python 3.12's standard library, including before the app venv exists.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
import warnings
import wave

ROOT = Path(__file__).resolve().parents[1]
MANUAL_REVIEW = ["回听两条音轨，尤其重叠区间及人物身份", "校对双人转写及字词时间戳", "逐项核对纪要与引用片段是否一致"]
SUDO_REFRESH_SECONDS = 60


class RebootRequired(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Deployment:
    def __init__(self, root: Path, log_dir: Path, *, secrets=()):
        self.root, self.log_dir = root.resolve(), log_dir.resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if (self.log_dir / "deployment-report.json").exists():
            raise RuntimeError("Use a new --log-dir for each attempt; previous reports are preserved.")
        self.secrets = [os.environ["HF_TOKEN"]] if os.environ.get("HF_TOKEN") else []
        self.secrets.extend(secrets)
        self.env = os.environ.copy()
        # Old developer shells may still point at 0.1.3 sources or a different
        # HF cache. Keep new workers and the reported cache strategy consistent.
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            self.env.pop(name, None)
        self.env.update(HF_HOME=str(root / ".cache/huggingface"), HF_HUB_DOWNLOAD_TIMEOUT="120",
                        HF_HUB_DISABLE_TELEMETRY="1", PYANNOTE_METRICS_ENABLED="0",
                        GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1",
                        PYTHONPATH=str(root / "src"))
        self.report = {"schema_version": 1, "status": "RUNNING", "started_at": utc_now(),
                       "project_root": str(self.root), "log_dir": str(self.log_dir),
                       "steps": [], "evidence": {}, "manual_review_required": MANUAL_REVIEW,
                       "quality_status": "PENDING_MANUAL_REVIEW"}
        checksum_file = self.log_dir / "source-sha256.txt"
        if checksum_file.is_file():
            checksum = checksum_file.read_text(encoding="utf-8").split()
            if not checksum or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum[0]):
                raise ValueError("source-sha256.txt does not contain a valid SHA-256 digest.")
            self.report["evidence"]["source_zip_sha256"] = checksum[0]
        self.stage_log = None
        self.ollama_test_loaded = False
        self.write_report()

    def clean(self, value):
        value = str(value)
        for secret in self.secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{8,}", "[REDACTED]", value)

    def emit(self, message):
        message = self.clean(message)
        print(message, flush=True)
        with (self.log_dir / "deployment.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        if self.stage_log:
            with self.stage_log.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def write_report(self):
        # Clean values before JSON escaping: replacing serialized credentials
        # containing quotes/backslashes can otherwise leak or invalidate JSON.
        def clean_tree(value):
            if isinstance(value, str):
                return self.clean(value)
            if isinstance(value, dict):
                return {self.clean(key): clean_tree(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean_tree(item) for item in value]
            return value

        report = clean_tree(self.report)
        content = json.dumps(report, ensure_ascii=False, indent=2)
        target = self.log_dir / "deployment-report.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(content + "\n", encoding="utf-8")
        temporary.replace(target)
        lines = ["# 部署与验收报告", "", f"状态：**{report['status']}**", "",
                 f"开始：{report['started_at']}", f"项目：`{report['project_root']}`", f"日志：`{report['log_dir']}`", "",
                 "| 步骤 | 状态 | 用时（秒） | 日志 |", "| --- | --- | ---: | --- |"]
        for step in report["steps"]:
            lines.append(f"| {step['name']} | {step['status']} | {step.get('seconds', '')} | {step['log']} |")
        if report.get("error"):
            lines.extend(["", "失败原因：", "", "```text", report["error"], "```"])
        if report.get("cleanup_warning"):
            lines.extend(["", "清理警告：", "", "```text", report["cleanup_warning"], "```"])
        if report.get("failed_logs"):
            lines.extend(["", "## 失败日志", ""])
            lines.extend("- `" + item + "`" for item in report["failed_logs"])
        lines.extend(["", "## 验收证据", "", "```json", json.dumps(report["evidence"], ensure_ascii=False, indent=2), "```",
                      "", "自动 PASS 只表示部署、实际执行及产物检查通过。听感和语义准确率仍需人工确认。", ""])
        lines.extend("- " + item for item in MANUAL_REVIEW)
        if report.get("ui_command"):
            lines.extend(["", "## 启动界面", "", "```bash", report["ui_command"], "```",
                          "", "浏览器访问 http://127.0.0.1:7860；新终端须重新输入 HF_TOKEN，或使用已下载的缓存。"])
        markdown = self.log_dir / "deployment-report.md"
        temporary_md = markdown.with_suffix(".md.tmp")
        temporary_md.write_text(self.clean("\n".join(lines)) + "\n", encoding="utf-8")
        temporary_md.replace(markdown)

    def load_upgrade_report(self, path):
        path = path.expanduser().resolve()
        upgrade = json.loads(path.read_text(encoding="utf-8"))
        if upgrade.get("status") != "PASS":
            raise RuntimeError(f"Cache upgrade did not pass. Inspect {path}")
        target = upgrade.get("new", {}).get("resolved_path")
        if not isinstance(target, str) or Path(target).resolve() != self.root:
            raise RuntimeError("Upgrade report does not belong to this project directory.")
        if upgrade.get("old", {}).get("version") != "0.1.3" or upgrade.get("new", {}).get("version") != "0.1.4":
            raise RuntimeError("Upgrade report must describe 0.1.3 -> 0.1.4.")
        self.report["evidence"]["upgrade"] = upgrade
        self.report["upgrade_report_path"] = str(path)
        self.emit("升级缓存报告：" + str(path))
        self.emit("旧版本保留：" + str(upgrade.get("old", {}).get("resolved_path", "")))
        self.emit("缓存复用策略：" + str(upgrade.get("cache", {}).get("strategy", "unknown")))

    def step(self, name, action):
        record = {"name": name, "status": "RUNNING", "started_at": utc_now(),
                  "log": f"{len(self.report['steps']) + 1:02d}-{name}.log"}
        self.report["steps"].append(record)
        self.stage_log = self.log_dir / record["log"]
        self.stage_log.touch()
        started = time.monotonic()
        self.emit(f"\n[{name}] {record['started_at']}")
        self.write_report()
        try:
            action()
        except BaseException as error:
            record["status"] = "REBOOT_REQUIRED" if isinstance(error, RebootRequired) else "FAIL"
            record["error"] = self.clean(str(error) or type(error).__name__)
            raise
        else:
            record["status"] = "PASS"
        finally:
            record["seconds"] = round(time.monotonic() - started, 2)
            self.stage_log = None
            self.write_report()

    def command(self, args, *, timeout=7200, check=True, pass_token=False,
                sudo_keepalive=False, env_overrides=None):
        args = [str(arg) for arg in args]
        if args and args[0] == "sudo":
            self.sudo_validate()
            args.insert(1, "-n")
        elif sudo_keepalive:
            self.sudo_validate()
        self.emit("$ " + shlex.join(args))
        output = []
        environment = self.env.copy()
        environment.update(env_overrides or {})
        if not pass_token:
            environment.pop("HF_TOKEN", None)
        # An independent process group permits subtree cancellation while
        # retaining the controlling TTY and sudo's TTY-scoped credentials.
        group_options = {"process_group": 0} if os.name == "posix" else {}
        process = subprocess.Popen(args, cwd=self.root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   **group_options)
        lines = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line.rstrip("\r\n"))
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        refresh_at = time.monotonic() + SUDO_REFRESH_SECONDS if sudo_keepalive else None
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Command exceeded {timeout}s: {args[0]}")
                if refresh_at is not None and time.monotonic() >= refresh_at:
                    self.sudo_refresh()
                    refresh_at = time.monotonic() + SUDO_REFRESH_SECONDS
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                self.emit(line)
                # Large pip logs are on disk; only the tail is needed in memory.
                output.append(self.clean(line))
                if len(output) > 2000:
                    del output[:1000]
            code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except BaseException:
            self.terminate(process)
            raise
        finally:
            reader.join(timeout=1)
            if not reader.is_alive():
                process.stdout.close()
        if code and check:
            raise RuntimeError(f"Command exited {code}: {shlex.join(args)}. See this step's log.")
        return code, "\n".join(output)

    @staticmethod
    def terminate(process):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The parent may already have exited while grandchildren still own
            # the output pipe; kill the group regardless of the parent's state.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def request(self, path, data=None, timeout=20):
        url = self.ollama_url + path
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def wait_unloaded(self):
        for _ in range(30):
            models = self.request("/api/ps").get("models", [])
            if not models:
                return
            time.sleep(1)
        raise RuntimeError("Ollama still has resident models after unload. Stop them before speech acceptance.")

    def platform_check(self, audio):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("This one-click route requires native Ubuntu 24.04 x86_64.")
        release = platform.freedesktop_os_release()
        if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04" or "microsoft" in platform.release().lower():
            raise RuntimeError("Use native Ubuntu 24.04 x86_64; WSL and other releases need a separate setup.")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Run with Python 3.12.")
        if os.geteuid() == 0:
            raise RuntimeError("Run as a normal user. The script invokes sudo only for system operations.")
        if not audio.is_file():
            raise RuntimeError(f"Recording not found: {audio}")
        for command in ("ffmpeg", "curl", "sox", "git", "systemctl", "sudo", "zstd"):
            if not shutil.which(command):
                raise RuntimeError(f"Missing {command}. Run the root deploy_ubuntu.sh to install system dependencies.")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "0") not in ("0",):
            raise RuntimeError("Unset CUDA_VISIBLE_DEVICES for the default GPU 0 acceptance route.")
        config = tomllib.loads((self.root / "configs/default.toml").read_text(encoding="utf-8"))
        expected = {"diarize": "pyannote", "separate": "clearvoice", "match": "pyannote_embedding", "transcribe": "qwen"}
        for stage, backend in expected.items():
            if config[stage].get("device") != "cuda:0" or config[stage].get("backend") != backend:
                raise RuntimeError(f"One-click acceptance requires the default {stage} backend {backend} on cuda:0. Use the manual setup for alternative backends/devices.")
            if "token_env" in config[stage] and config[stage]["token_env"] != "HF_TOKEN":
                raise RuntimeError("One-click acceptance uses HF_TOKEN for Hugging Face access. Restore token_env=HF_TOKEN or use manual deployment.")
        self.ollama_url = config["summary"]["url"].rstrip("/")
        if self.ollama_url != "http://127.0.0.1:11434":
            raise RuntimeError("One-click acceptance requires summary.url=http://127.0.0.1:11434.")
        self.model = config["summary"]["model"]
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", self.model):
            raise RuntimeError("Invalid Ollama model name.")
        if ":" not in self.model.rsplit("/", 1)[-1]:
            self.model += ":latest"
        self.env["OLLAMA_HOST"] = self.ollama_url
        version = tomllib.loads((self.root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        self.report["evidence"].update(os=release.get("PRETTY_NAME"), python=sys.version.split()[0],
                                       project_version=version, input_audio=str(audio),
                                       free_disk_gib=round(shutil.disk_usage(self.root).free / 2**30, 1))
        self.emit("首次安装与模型下载需要数十 GB 空间和网络；本次环境、缓存和报告均保留。")

    def driver_check(self, install_driver=False):
        marker = self.root / ".cache/deployment-driver-reboot.json"
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("boot_id") == boot_id:
            raise RebootRequired("Driver installation is recorded in this boot. Reboot manually, then rerun; drivers will not be installed again before reboot.")
        driver_error = None
        try:
            if not shutil.which("nvidia-smi"):
                raise RuntimeError("nvidia-smi is missing")
            _, output = self.command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"], timeout=30)
            rows = list(csv.reader(output.splitlines()))
            if len(rows) != 1:
                raise ValueError("One-click acceptance supports a single GPU machine. Use the manual deployment to select devices on multi-GPU machines.")
            gpu = next(row for row in rows if row[0].strip() == "0")
            if "RTX 5080" not in gpu[1]:
                raise ValueError(f"GPU 0 is {gpu[1].strip()}, expected RTX 5080")
            version = tuple(int(value) for value in gpu[2].strip().split("."))
            if version < (570, 124, 6):
                raise RuntimeError("The driver must be >=570.124.06 for the cu128 baseline")
            kernel = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8")
            if "Open Kernel Module" not in kernel:
                raise RuntimeError("Blackwell requires NVIDIA's open kernel module driver")
            self.report["evidence"]["gpu"] = {"index": 0, "name": gpu[1].strip(), "driver": gpu[2].strip(),
                                                     "memory_mib": int(gpu[3].strip()), "kernel": kernel.strip()}
            marker.unlink(missing_ok=True)
            return
        except ValueError:
            raise
        except (OSError, StopIteration, RuntimeError) as error:
            driver_error = str(error)
        if install_driver:
            self.emit("驱动检测失败；检查 Ubuntu 推荐的 NVIDIA open 驱动。安装后需重启并重新执行本命令。")
            _, devices = self.command(["ubuntu-drivers", "devices"], timeout=120)
            packages = re.findall(r"\b(nvidia-driver-\d+(?:-server)?-open)\b[^\n]*\brecommended\b", devices)
            if len(set(packages)) != 1:
                raise RuntimeError("No unambiguous recommended NVIDIA open driver was found. Use Ubuntu's driver manager to install a supported open driver, reboot and rerun.")
            package = packages[0]
            branch = package.removeprefix("nvidia-driver-")
            if int(branch.split("-")[0]) < 570:
                raise RuntimeError("Ubuntu's recommended open driver is too old for this Blackwell baseline.")
            self.command(["sudo", "ubuntu-drivers", "install", "nvidia:" + branch], timeout=1800)
            # Some ubuntu-drivers versions do not propagate the internal apt
            # command's failure. Verify actual dpkg state before claiming reboot.
            _, package_state = self.command(["dpkg-query", "-W", "-f=${db:Status-Status}", package], timeout=30)
            if package_state.strip() != "installed":
                raise RuntimeError(f"Driver package {package} is not fully installed. Inspect the driver log and repair apt/dpkg before retrying; no reboot marker was written.")
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary_marker = marker.with_suffix(".json.tmp")
            temporary_marker.write_text(json.dumps({"boot_id": boot_id, "package": package, "installed_at": utc_now()}), encoding="utf-8")
            temporary_marker.replace(marker)
            raise RebootRequired("Driver installation completed. Reboot manually, then rerun the same command; open kernel module will be checked again.")
        raise RuntimeError(driver_error + ". Install a current Ubuntu NVIDIA open driver, reboot, then rerun; --install-driver can request Ubuntu's recommended driver installation.")

    def token_input(self):
        self.emit("请先在 https://huggingface.co/pyannote/speaker-diarization-community-1 接受访问条件。")
        # Never fall back to echoing stdin (getpass otherwise warns and does so).
        try:
            with open("/dev/tty", "r+"):
                pass
        except OSError as error:
            raise RuntimeError("HF_TOKEN requires an interactive terminal. Run this script directly in a terminal.") from error
        try:
            with warnings.catch_warnings():
                # getpass otherwise falls back to echoing stdin when termios
                # cannot disable echo, even if opening /dev/tty succeeded.
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("HF_TOKEN（隐藏输入；留空复用当前环境变量）: ").strip()
        except getpass.GetPassWarning as error:
            raise RuntimeError("Cannot disable terminal echo for HF_TOKEN. Use a regular interactive terminal; no token was read.") from error
        token = token or self.env.get("HF_TOKEN", "").strip()
        if not token or any(char.isspace() for char in token):
            raise RuntimeError("HF_TOKEN is empty or contains whitespace.")
        self.secrets.append(token)
        self.env["HF_TOKEN"] = token
        self.emit("HF_TOKEN 已读取，仅传入当前进程及子进程，不写入配置、命令或报告。")

    def hf_access(self):
        code = ("import os; from huggingface_hub import hf_hub_download; "
                "hf_hub_download(repo_id='pyannote/speaker-diarization-community-1', filename='config.yaml', token=os.environ['HF_TOKEN']); "
                "print('HF gated repository access: PASS')")
        self.command([self.root / ".venvs/diarize/bin/python", "-c", code], timeout=600, pass_token=True)

    def install_environments(self):
        self.command(["bash", self.root / "scripts/install_ubuntu.sh", "--python", sys.executable], timeout=14400)
        snapshots = self.log_dir / "installed-versions"
        snapshots.mkdir()
        for name in ("app", "diarize", "separate", "asr"):
            shutil.copyfile(self.root / f".venvs/{name}/installed-versions.txt", snapshots / f"{name}.txt")
        self.report["evidence"]["installed_versions"] = str(snapshots)

    def ollama_setup(self):
        state_code, state = self.command(
            ["systemctl", "show", "ollama.service", "-p", "LoadState", "--value"],
            timeout=30, check=False)
        state = state.strip()
        if state not in {"loaded", "not-found"} or state == "loaded" and state_code:
            raise RuntimeError(f"Ollama service LoadState is {state or 'unavailable'} (exit {state_code}). "
                               "Resolve masked, invalid, or inaccessible service configuration before retrying; the installer was not run.")
        # A partially completed first install can leave the executable in PATH
        # before it creates the service. Repair that case on the next attempt.
        if not shutil.which("ollama") or state == "not-found":
            installer = self.log_dir / "ollama-install.sh"
            self.command(["curl", "--fail", "--show-error", "--location", "--proto", "=https", "--tlsv1.2",
                          "https://ollama.com/install.sh", "--output", installer], timeout=300)
            # The official installer calls sudo repeatedly, including after a
            # large download. Its child runs in a background process group, so
            # it must never try to prompt on the controlling terminal. This
            # temporary shim contains no credential and is removed on exit.
            with tempfile.TemporaryDirectory(prefix="speaker-lab-sudo-") as temporary:
                shim = Path(temporary) / "sudo"
                shim.write_text('#!/bin/sh\nexec /usr/bin/sudo -n "$@"\n', encoding="utf-8")
                shim.chmod(0o755)
                self.command(["sh", installer], timeout=1800, sudo_keepalive=True,
                             env_overrides={"PATH": temporary + os.pathsep + self.env.get("PATH", os.defpath)})
        # The official installer sets up the service; existing installations reuse it.
        self.command(["sudo", "systemctl", "enable", "--now", "ollama"], timeout=120)
        for attempt in range(30):
            try:
                tags = self.request("/api/tags")
                break
            except (OSError, ValueError, urllib.error.URLError):
                if attempt == 29:
                    raise RuntimeError("Ollama did not become ready. Check: sudo journalctl -u ollama -n 100")
                time.sleep(1)
        self.command(["ollama", "--version"], timeout=30)
        resident = self.request("/api/ps").get("models", [])
        if resident:
            raise RuntimeError("Ollama has resident models. Stop them yourself using ollama ps / ollama stop MODEL, then rerun.")
        names = {row.get("name") for row in tags.get("models", [])}
        if self.model not in names:
            self.command(["ollama", "pull", self.model], timeout=7200)
        else:
            self.emit(f"Ollama 模型 {self.model} 已存在，复用已有文件。")
        tags = self.request("/api/tags")
        self.report["evidence"]["ollama_model_file"] = [row for row in tags.get("models", []) if row.get("name") == self.model]
        details = self.request("/api/show", {"model": self.model})
        if details.get("remote_host") or details.get("remote_model"):
            raise RuntimeError("Acceptance requires a local Ollama model, not a cloud model.")

    def ollama_gpu(self):
        self.ollama_test_loaded = True
        result = self.request("/api/chat", {"model": self.model, "messages": [{"role": "user", "content": "只回复：正常"}],
                              "stream": False, "think": False, "keep_alive": "5m",
                              "options": {"num_ctx": 8192, "num_predict": 32}}, timeout=300)
        if result.get("done") is not True or not result.get("message", {}).get("content", "").strip():
            raise RuntimeError("Ollama generation returned no completed answer.")
        resident = self.request("/api/ps")
        _, table = self.command(["ollama", "ps"], timeout=30)
        target = [line for line in table.splitlines() if line.split() and line.split()[0] == self.model]
        rows = [row for row in resident.get("models", []) if row.get("name", row.get("model")) == self.model]
        self.report["evidence"]["ollama"] = {"model": self.model, "generation_done": True,
                                                   "generation_text": result["message"]["content"],
                                                   "ps_table": table, "resident": resident}
        if len(target) != 1 or not re.search(r"\b100%\s+GPU\b", target[0]) or not rows or not all(
            type(row.get("size")) is int and type(row.get("size_vram")) is int
            and row["size"] > 0 and row["size_vram"] == row["size"] for row in rows
        ):
            raise RuntimeError("Ollama model must show 100% GPU in ollama ps and size_vram == size > 0 in /api/ps.")
        self.command(["ollama", "stop", self.model], timeout=60)
        self.wait_unloaded()
        self.ollama_test_loaded = False
        self.emit("Ollama 实际生成与 100% GPU 驻留：PASS；已卸载模型以释放语音阶段显存。")

    def prepare_audio(self, source):
        destination = self.log_dir / "input.wav"
        self.command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", source,
                      "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", destination], timeout=300)
        with wave.open(str(destination), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not 10 <= duration <= 120:
            raise RuntimeError(f"Acceptance recording is {duration:.1f}s. Supply a 10–120s recording (recommended 30–60s) with solo speech from both people and several seconds of overlap.")
        self.report["evidence"]["normalized_audio"] = {"path": str(destination), "duration_seconds": duration}

    def accept(self):
        child_logs = self.log_dir / "recording-acceptance"
        self.command([self.root / ".venvs/app/bin/python", "-u", self.root / "scripts/acceptance.py",
                      self.log_dir / "input.wav", "--config", self.root / "configs/default.toml", "--log-dir", child_logs], timeout=10800, pass_token=True)
        evidence = json.loads((child_logs / "acceptance-report.json").read_text(encoding="utf-8"))
        if evidence.get("automatic_checks") != "PASS":
            raise RuntimeError("Recording acceptance did not return PASS.")
        self.report["evidence"]["recording_acceptance"] = evidence
        self.wait_unloaded()
        self.report["evidence"]["ollama_after_summary"] = "No resident models"

    def execute(self, audio, install_driver=False):
        self.step("platform", lambda: self.platform_check(audio))
        # Keep sudo's password prompt directly on the terminal, before capturing
        # command output. Later sudo calls can refresh this credential as needed.
        self.step("sudo", self.sudo_validate)
        self.step("driver", lambda: self.driver_check(install_driver))
        self.step("audio", lambda: self.prepare_audio(audio))
        self.step("hf-token", self.token_input)
        self.step("python-environments", self.install_environments)
        self.step("hf-access", self.hf_access)
        self.step("ollama-setup", self.ollama_setup)
        self.step("ollama-gpu", self.ollama_gpu)
        self.step("doctor", lambda: self.command([self.root / ".venvs/app/bin/python", self.root / "scripts/doctor.py"], timeout=1200, pass_token=True))
        self.step("clearvoice-cuda", lambda: self.command([self.root / ".venvs/separate/bin/python", self.root / "scripts/smoke_models.py", "--device", "cuda:0"], timeout=300))
        self.step("recording-acceptance", self.accept)
        self.report["ui_command"] = "cd " + shlex.quote(str(self.root)) + " && bash scripts/run.sh"

    def sudo_validate(self):
        environment = self.env.copy()
        environment.pop("HF_TOKEN", None)
        self.emit("sudo -v：如需密码，请在终端输入。")
        result = subprocess.run(["sudo", "-v"], env=environment, check=False)
        if result.returncode:
            raise RuntimeError("sudo authentication failed.")

    def sudo_refresh(self):
        """Refresh the terminal's cached authorization without ever prompting."""
        environment = self.env.copy()
        environment.pop("HF_TOKEN", None)
        result = subprocess.run(["/usr/bin/sudo", "-n", "-v"], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=10, check=False)
        if result.stdout:
            self.emit(result.stdout.rstrip())
        if result.returncode:
            raise RuntimeError("sudo cached authorization could not be refreshed noninteractively. "
                               "The installer was stopped; rerun from an interactive terminal to authorize sudo again.")
        self.emit("sudo authorization refreshed noninteractively.")

    def finish(self, status, error=None):
        self.report["status"] = status
        self.report["finished_at"] = utc_now()
        if error:
            self.report["error"] = self.clean(error)
        failed_logs = []
        for step in self.report["steps"]:
            if step.get("status") in {"FAIL", "REBOOT_REQUIRED", "RUNNING"}:
                failed_logs.append(str((self.log_dir / step["log"]).resolve()))
        if status != "PASS":
            failed_logs.append(str(self.log_dir / "deployment.log"))
            if (self.log_dir / "bootstrap.log").is_file():
                failed_logs.append(str(self.log_dir / "bootstrap.log"))
        self.report["failed_logs"] = list(dict.fromkeys(failed_logs))
        recording_report = self.log_dir / "recording-acceptance/acceptance-report.json"
        if recording_report.is_file():
            self.report["recording_acceptance_report_path"] = str(recording_report.resolve())
        self.write_report()
        self.emit("\n===== 本次部署与验收报告 =====\n" + (self.log_dir / "deployment-report.md").read_text(encoding="utf-8"))
        self.emit(f"完整报告：{self.log_dir / 'deployment-report.md'}")
        self.emit(f"JSON 报告：{self.log_dir / 'deployment-report.json'}")
        if recording_report.is_file():
            self.emit("录音验收报告：" + str(recording_report.resolve()))
        else:
            self.emit("录音验收报告尚未生成：本次未执行或未完成录音验收。")
        for path in self.report["failed_logs"]:
            self.emit("失败日志：" + path)
        if status == "PASS":
            self.emit("自动执行与产物验收通过。请回听音轨并校对转写、纪要；详见报告。")
            if self.report.get("ui_command"):
                self.emit("启动 Gradio：" + self.report["ui_command"])


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message + "; use --help for usage")


def early_failure(error, secrets=()):
    """Preserve prior reports and use a fresh fallback if initialization fails."""
    parent = ROOT.parent / "deployment_logs"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="initialization-failure-", dir=parent))
    except OSError:
        directory = Path(tempfile.mkdtemp(prefix="speaker-lab-failure-"))
    failed = Deployment(ROOT, directory, secrets=secrets)
    failed.report["steps"].append({"name": "initialization", "status": "FAIL", "log": "deployment.log"})
    failed.finish("FAIL", f"{type(error).__name__}: {error}")


def main(argv=None):
    parser = Arguments(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--upgrade-report", type=Path, help="Verified cache upgrade report produced by deploy_ubuntu.sh --upgrade")
    parser.add_argument("--install-driver", action="store_true", help="On a driver check failure, install Ubuntu's recommended driver, then stop for a manual reboot")
    try:
        args = parser.parse_args(argv)
    except ValueError as exception:
        early_failure(exception)
        return 2
    log_dir = args.log_dir or ROOT / "deployment_logs" / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    try:
        deployment = Deployment(ROOT, log_dir.expanduser())
    except Exception as exception:
        early_failure(exception)
        return 1
    status, error, code = "PASS", None, 0
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def terminate_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate_signal)
    try:
        if args.upgrade_report:
            deployment.step("upgrade-report", lambda: deployment.load_upgrade_report(args.upgrade_report))
        deployment.execute(args.audio.expanduser().resolve(), args.install_driver)
    except RebootRequired as exception:
        status, error, code = "REBOOT_REQUIRED", str(exception), 20
    except KeyboardInterrupt:
        status, error, code = "FAIL", "Interrupted by user; remaining steps were not executed.", 130
    except Exception as exception:
        status, error, code = "FAIL", f"{type(exception).__name__}: {exception}", 1
    finally:
        # A second Ctrl+C/SIGTERM must not skip final report persistence.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                if deployment.ollama_test_loaded:
                    deployment.command(["ollama", "stop", deployment.model], timeout=30)
            except BaseException as cleanup_error:
                deployment.report["cleanup_warning"] = deployment.clean(str(cleanup_error) or type(cleanup_error).__name__)
            finally:
                deployment.finish(status, error)
        except Exception as report_error:
            early_failure(report_error, deployment.secrets)
            code = 1
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
