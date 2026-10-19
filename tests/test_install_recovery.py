"""Run the installer venv bootstrap in Bash using offline Python/pip stand-ins."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = (ROOT / "scripts/install_ubuntu.sh").read_text(encoding="utf-8")
MAKE_ENV = "make_env() {\n" + INSTALLER.split("make_env() {\n", 1)[1].split("\nproject_install() {", 1)[0]


@pytest.fixture
def installer_sandbox(tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for the offline installer recovery checks")
    worker = tmp_path / "worker-python"
    worker.write_text("""#!/usr/bin/env bash
set -eu
environment="$(dirname -- "$(dirname -- "$0")")"
printf 'worker:%s\\n' "$*" >> "$TRACE"
if [[ "$1" == -c ]]; then
  [[ "$(cat "$environment/version")" == 3.12 ]] || { printf 'Existing venv must use Python 3.12\\n' >&2; exit 1; }
elif [[ "$1 $2 $3" == '-m pip --version' ]]; then
  [[ -f "$environment/pip-ready" ]]
elif [[ "$1 $2" == '-m ensurepip' ]]; then
  touch "$environment/pip-ready"
elif [[ "$1 $2 $3" == '-m pip install' ]]; then
  [[ -f "$environment/pip-ready" ]]
else
  printf 'Unexpected Python invocation\\n' >&2
  exit 2
fi
""", encoding="utf-8", newline="\n")
    worker.chmod(0o755)
    base = tmp_path / "base-python"
    base.write_text("""#!/usr/bin/env bash
set -eu
printf 'base:%s\\n' "$*" >> "$TRACE"
[[ "$1 $2" == '-m venv' ]]
mkdir -p "$3/bin"
printf 'version = 3.12\\n' > "$3/pyvenv.cfg"
printf '3.12\\n' > "$3/version"
cp "$FAKE_WORKER" "$3/bin/python"
chmod +x "$3/bin/python"
touch "$3/pip-ready"
""", encoding="utf-8", newline="\n")
    base.chmod(0o755)
    script = tmp_path / "make-env.sh"
    script.write_text("set -Eeuo pipefail\n" + MAKE_ENV + "\nmake_env app\n", encoding="utf-8", newline="\n")
    trace = tmp_path / "trace.log"

    def run():
        environment = os.environ.copy()
        environment.update(PYTHON_BIN=str(base), FAKE_WORKER=str(worker), TRACE=str(trace))
        return subprocess.run([bash, str(script)], cwd=tmp_path, env=environment,
                              text=True, capture_output=True, timeout=30)

    return tmp_path, worker, trace, run


def test_first_install_then_repeat_reuses_interpreter_and_files(installer_sandbox):
    root, _, trace, run = installer_sandbox
    first = run()
    assert first.returncode == 0, first.stderr
    environment = root / ".venvs/app"
    cached = environment / "existing-cache.bin"
    cached.write_bytes(b"keep the cache")
    second = run()
    assert second.returncode == 0, second.stderr
    commands = trace.read_text(encoding="utf-8").splitlines()
    assert commands.count("base:-m venv .venvs/app") == 1
    assert len([line for line in commands if line.startswith("worker:-m pip install")]) == 2
    assert not any("ensurepip" in line for line in commands)
    assert cached.read_bytes() == b"keep the cache"


def test_cfg_only_interrupted_creation_completes_without_deleting_files(installer_sandbox):
    root, _, trace, run = installer_sandbox
    environment = root / ".venvs/app"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("version = 3.12\n", encoding="utf-8")
    preserved = environment / "do-not-delete.txt"
    preserved.write_text("preserved", encoding="utf-8")
    result = run()
    assert result.returncode == 0, result.stderr
    assert (environment / "bin/python").is_file()
    assert "base:-m venv .venvs/app" in trace.read_text(encoding="utf-8")
    assert preserved.read_text(encoding="utf-8") == "preserved"


def make_existing_environment(root, worker, version="3.12", ready=False):
    environment = root / ".venvs/app"
    (environment / "bin").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("version = " + version + "\n", encoding="utf-8")
    (environment / "version").write_text(version + "\n", encoding="utf-8")
    shutil.copyfile(worker, environment / "bin/python")
    (environment / "bin/python").chmod(0o755)
    if ready:
        (environment / "pip-ready").touch()
    return environment


def test_interpreter_without_pip_runs_ensurepip_then_installs(installer_sandbox):
    root, worker, trace, run = installer_sandbox
    environment = make_existing_environment(root, worker)
    result = run()
    assert result.returncode == 0, result.stderr
    commands = trace.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("base:") for line in commands)
    assert commands.index("worker:-m ensurepip --upgrade") < next(
        index for index, line in enumerate(commands) if line.startswith("worker:-m pip install")
    )
    assert (environment / "pip-ready").is_file()


@pytest.mark.parametrize("has_config", [False, True])
def test_other_python_version_stops_before_any_repair(installer_sandbox, has_config):
    root, worker, trace, run = installer_sandbox
    environment = make_existing_environment(root, worker, version="3.11")
    if not has_config:
        (environment / "pyvenv.cfg").unlink()
    before = {path.relative_to(environment): path.read_bytes() for path in environment.rglob("*") if path.is_file()}
    result = run()
    assert result.returncode != 0
    assert "Python 3.12" in result.stderr
    commands = trace.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 1 and commands[0].startswith("worker:-c ")
    assert {path.relative_to(environment): path.read_bytes() for path in environment.rglob("*") if path.is_file()} == before
