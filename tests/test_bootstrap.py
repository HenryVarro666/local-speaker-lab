"""Exercise real bootstrap extraction without apt, sudo, Linux, or model downloads."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import shlex
import signal
import stat
import subprocess
import sys
import time
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "deploy_ubuntu.sh"
EXTRACTOR = BOOTSTRAP.read_text(encoding="utf-8").split("<<'PYTHON_SAFE_EXTRACT'\n", 1)[1].split("\nPYTHON_SAFE_EXTRACT", 1)[0]
REQUIRED = {
    "local-speaker-lab/pyproject.toml": '[project]\nname = "local-speaker-lab"\nversion = "0.1.4"\n',
    "local-speaker-lab/scripts/deploy_and_accept.py": "print('runner')\n",
    "local-speaker-lab/deploy_ubuntu.sh": "#!/usr/bin/env bash\n",
}


def bash_executable():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is unavailable")
    return bash


def make_archive(path, extra=(), members=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, contents in (REQUIRED if members is None else members).items():
                archive.writestr(name, contents)
            for name, contents in extra:
                archive.writestr(name, contents)
    return path


def extract(archive, destination, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(sys, "argv", ["bootstrap-extractor", str(archive), str(destination)])
        exec(compile(EXTRACTOR, str(BOOTSTRAP) + ":safe-extract", "exec"), {})


def test_repeated_zip_reuses_sources_and_preserves_runtime_files(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip")
    destination = tmp_path / "project"
    extract(archive, destination, monkeypatch)
    cache = destination / ".cache" / "model.bin"
    cache.parent.mkdir()
    cache.write_bytes(b"downloaded model")
    (destination / "deploy_ubuntu.sh").unlink()
    extract(archive, destination, monkeypatch)
    assert cache.read_bytes() == b"downloaded model"
    assert (destination / "deploy_ubuntu.sh").is_file()
    extract(archive, destination, monkeypatch)
    assert cache.read_bytes() == b"downloaded model"


def test_user_source_change_stops_before_any_project_write(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip")
    destination = tmp_path / "project"
    extract(archive, destination, monkeypatch)
    metadata = destination / "pyproject.toml"
    metadata.write_text("user modifications", encoding="utf-8")
    absent = destination / "deploy_ubuntu.sh"
    absent.unlink()
    with pytest.raises(RuntimeError, match="Existing source differs"):
        extract(archive, destination, monkeypatch)
    assert metadata.read_text(encoding="utf-8") == "user modifications"
    assert not absent.exists()


def test_interrupted_extraction_into_cache_directory_resumes_without_partial_source(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip")
    destination = tmp_path / "project"
    cache = destination / ".cache/model.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"previously downloaded")
    original_link = os.link
    linked = []

    def interrupt_second_link(source, target):
        if linked:
            raise InterruptedError("simulated interruption")
        original_link(source, target)
        linked.append(Path(target).name)

    with monkeypatch.context() as context:
        context.setattr(os, "link", interrupt_second_link)
        with pytest.raises(InterruptedError, match="simulated interruption"):
            extract(archive, destination, context)
    assert linked == ["pyproject.toml"]
    assert 'version = "0.1.4"' in (destination / "pyproject.toml").read_text(encoding="utf-8")
    extract(archive, destination, monkeypatch)
    assert cache.read_bytes() == b"previously downloaded"
    assert (destination / "scripts/deploy_and_accept.py").is_file()


@pytest.mark.parametrize("member", [
    "local-speaker-lab/../escape",
    "/local-speaker-lab/escape",
    "another-root/escape",
    "local-speaker-lab/./escape",
    "local-speaker-lab//escape",
    "local-speaker-lab",
])
def test_unsafe_member_stops_before_destination_write(tmp_path, monkeypatch, member):
    archive = make_archive(tmp_path / "source.zip", [(member, "unsafe")])
    destination = tmp_path / "project"
    with pytest.raises(RuntimeError, match="Unsafe ZIP member"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("replacement", [b"foo\\bar", b"foo\x00bar"])
def test_raw_zip_backslash_and_nul_names_are_refused(tmp_path, monkeypatch, replacement):
    # Windows zipfile normalizes backslashes when writing. Alter both ZIP headers
    # directly so this test covers the same hostile bytes on Windows and Linux.
    archive = make_archive(tmp_path / "source.zip", [("local-speaker-lab/foo-bar", "unsafe")])
    archive.write_bytes(archive.read_bytes().replace(b"local-speaker-lab/foo-bar", b"local-speaker-lab/" + replacement))
    destination = tmp_path / "project"
    with pytest.raises(RuntimeError, match="Unsafe ZIP member"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_archive_links_and_special_files_are_refused(tmp_path, monkeypatch, kind):
    entry = zipfile.ZipInfo("local-speaker-lab/special")
    entry.create_system = 3
    entry.external_attr = (kind | 0o777) << 16
    archive = make_archive(tmp_path / "source.zip", [(entry, "/tmp/escape")])
    destination = tmp_path / "project"
    with pytest.raises(RuntimeError, match="Links, special files"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()


def test_duplicate_member_is_refused(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip", [("local-speaker-lab/pyproject.toml", "duplicate")])
    destination = tmp_path / "project"
    with pytest.raises(RuntimeError, match="Duplicate ZIP member"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()


def test_missing_project_files_are_refused(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip", members={"local-speaker-lab/unrelated": "data"})
    destination = tmp_path / "project"
    with pytest.raises(RuntimeError, match="missing the project"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()


def test_crc_failure_leaves_project_untouched(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip")
    archive.write_bytes(archive.read_bytes().replace(b"print('runner')", b"print('broken')", 1))
    destination = tmp_path / "project"
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        extract(archive, destination, monkeypatch)
    assert not destination.exists()


def test_existing_symbolic_link_is_refused(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "source.zip")
    destination = tmp_path / "project"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (destination / "scripts").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links is not permitted on this host")
    with pytest.raises(RuntimeError, match="symbolic link"):
        extract(archive, destination, monkeypatch)
    assert not list(outside.iterdir())
    assert not (destination / "pyproject.toml").exists()


def test_invalid_arguments_save_failure_reports_without_running_system_setup(tmp_path):
    bash = bash_executable()
    secret = "hf_TEST_PRIVATE_BOOTSTRAP_TOKEN"
    environment = dict(os.environ, HF_TOKEN=secret)
    result = subprocess.run([
        bash, str(BOOTSTRAP), "--destination", str(tmp_path / "project"), "--unknown-option",
    ], capture_output=True, text=True, env=environment, timeout=20)
    assert result.returncode == 2
    reports = list((tmp_path / "deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "FAILED" and report["stage"] == "arguments"
    assert "Unknown argument" in report["error"]
    for name in ("bootstrap.log", "bootstrap-report.json", "bootstrap-report.md"):
        assert secret not in (reports[0].parent / name).read_text(encoding="utf-8")
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("destination", ["", "/"])
def test_invalid_destination_uses_fallback_and_saves_failure_report(tmp_path, destination):
    environment = dict(os.environ, HOME=tmp_path.as_posix())
    result = subprocess.run([
        bash_executable(), str(BOOTSTRAP), "--destination", destination, "--audio", "missing.wav",
    ], capture_output=True, text=True, env=environment, timeout=20)
    assert result.returncode == 2
    reports = list((tmp_path / "speaker-lab/deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "FAILED" and report["stage"] == "arguments"
    assert "destination" in report["error"]
    assert "fallback" in (reports[0].parent / "bootstrap.log").read_text(encoding="utf-8")


def test_unwritable_log_parent_falls_back_without_losing_argument_failure(tmp_path):
    blocked_parent = tmp_path / "ordinary-file"
    blocked_parent.write_text("preserve", encoding="utf-8")
    environment = dict(os.environ, HOME=tmp_path.as_posix())
    result = subprocess.run([
        bash_executable(), str(BOOTSTRAP), "--destination", str(blocked_parent / "project"), "--unknown",
    ], capture_output=True, text=True, env=environment, timeout=20)
    assert result.returncode == 2
    reports = list((tmp_path / "speaker-lab/deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text(encoding="utf-8"))["error"].startswith("Unknown argument")
    assert blocked_parent.read_text(encoding="utf-8") == "preserve"


def test_bootstrap_redacts_token_in_argument_and_destination(tmp_path):
    secret = "hf_TEST_PRIVATE_BOOTSTRAP_TOKEN"
    environment = dict(os.environ, HF_TOKEN=secret)
    destination = tmp_path / secret / "project"
    result = subprocess.run([
        bash_executable(), str(BOOTSTRAP), "--destination", str(destination), "--" + secret,
    ], capture_output=True, text=True, env=environment, timeout=20)
    assert result.returncode == 2
    assert secret not in result.stdout + result.stderr
    reports = list((destination.parent / "deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert "[REDACTED]" in report["project_directory"]
    assert secret not in "".join(path.read_text(encoding="utf-8") for path in reports[0].parent.iterdir())


def test_json_report_encoder_preserves_unicode_and_escapes_all_control_characters(tmp_path):
    source = BOOTSTRAP.read_text(encoding="utf-8")
    helpers = source.split("redact_into() {", 1)[1].split("# Invalid destinations", 1)[0]
    script = tmp_path / "report-encoding.sh"
    script.write_text('set -Eeuo pipefail\nredact_into() {' + helpers + '\njson_string "$PAYLOAD"\n', encoding="utf-8")
    value = '中文 path\\quote"\n\t\r\x01\x08\x0c\x1b\n'
    result = subprocess.run([
        bash_executable(), str(script),
    ], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PAYLOAD=value, HF_TOKEN=""), timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == value


def bootstrap_harness(tmp_path):
    """Run the complete shell flow, substituting only host metadata and commands.

    No apt/service/GPU operation runs on the test host. ZIP checks, extraction,
    logging, argument handling, shell traps and child exit handling stay real.
    """
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    source = BOOTSTRAP.read_text(encoding="utf-8")
    # Git Bash accepts C:/ paths for cd, but a drive colon is a PATH separator.
    # Resolve the directory inside Bash so PATH receives /c/... on Windows.
    source = source.replace("set -Eeuo pipefail", 'set -Eeuo pipefail\nPATH="$(cd -- "$FAKE_BIN" && pwd):$PATH"', 1)
    source = source.replace('[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]', "true")
    source = source.replace("[[ -r /etc/os-release ]]", "true")
    source = source.replace(". /etc/os-release", "ID=ubuntu; VERSION_ID=24.04")
    source = source.replace("((EUID != 0))", "true")
    source = source.replace('if { exec 4<>/dev/tty; } 2>/dev/null; then', 'if false; then')
    script = tmp_path / "bootstrap-under-test.sh"
    script.write_text(source, encoding="utf-8", newline="\n")
    # The deployment target is Python 3.12 even when pytest itself runs 3.11.
    # Simulate only the executable's version probe in its own child process;
    # archive extraction and all other Python work use the actual test host.
    host_python = shlex.quote(Path(sys.executable).as_posix())
    python_probe = (
        "import os, sys; "
        "sys.version_info = (3, int(os.environ.get('FAKE_PYTHON_MINOR', '12')), 0, 'final', 0); "
        "exec(sys.argv[1])"
    )
    python_stub = (
        '#!/usr/bin/env bash\n'
        'if [[ "${1:-}" == -c ]]; then\n'
        f'  exec {host_python} -c {shlex.quote(python_probe)} "$2"\n'
        'fi\n'
        f'exec {host_python} "$@"\n'
    )
    commands = {
        "python3.12": python_stub,
        "sudo": '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$FAKE_COMMANDS"\nprintf \'%s\\n\' "${HF_TOKEN:-}"\n',
        "dpkg-query": '#!/usr/bin/env bash\nif [[ "${FAKE_PACKAGES_MISSING:-0}" == 1 ]]; then printf not-installed; else printf installed; fi\n',
    }
    for name, contents in commands.items():
        path = fake_bin / name
        path.write_text(contents, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    runner = """import argparse, os, signal, time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--audio')
parser.add_argument('--log-dir')
parser.add_argument('--install-driver', action='store_true')
parser.add_argument('--upgrade-report')
args=parser.parse_args()
def stop(signum, frame):
    Path(args.log_dir, 'child-terminated.txt').write_text('stopped', encoding='utf-8')
    raise SystemExit(130)
signal.signal(signal.SIGTERM, stop)
Path(args.log_dir, 'child-ran.txt').write_text('called', encoding='utf-8')
if args.upgrade_report:
    assert Path(args.upgrade_report).is_file()
    Path(args.log_dir, 'child-upgrade-report.txt').write_text(args.upgrade_report, encoding='utf-8')
print('runner diagnostic token: ' + os.environ.get('HF_TOKEN', ''))
if os.environ.get('FAKE_RUNNER_SLEEP'):
    time.sleep(15)
    Path(args.log_dir, 'child-continued.txt').write_text('unexpected', encoding='utf-8')
raise SystemExit(int(os.environ.get('FAKE_RUNNER_EXIT', '0')))
"""
    cache_helper = '''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--old')
parser.add_argument('--new')
parser.add_argument('--report')
args = parser.parse_args()
report = dict(status='PASS', old=args.old, new=args.new)
if os.environ.get('FAKE_UPGRADE_EXIT'):
    report.update(status='FAILED', error='cache conflict detected')
Path(args.report).write_text(json.dumps(report), encoding='utf-8')
raise SystemExit(int(os.environ.get('FAKE_UPGRADE_EXIT', '0')))
'''
    archive = make_archive(tmp_path / "source.zip", members={
        **REQUIRED, "local-speaker-lab/scripts/deploy_and_accept.py": runner,
        "local-speaker-lab/scripts/upgrade_cache.py": cache_helper,
    })
    Path(str(archive) + ".sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  source.zip\n", encoding="ascii")
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"test recording; decoder is a runner concern")
    destination = tmp_path / "project"
    environment = dict(os.environ, HOME=tmp_path.as_posix(), FAKE_BIN=fake_bin.as_posix(),
                       FAKE_COMMANDS=str(tmp_path / "commands.txt"))

    arguments = [bash_executable(), str(script), "--zip", str(archive), "--destination", str(destination), "--audio", str(audio)]

    def run(bootstrap_args=(), input_text=None, **variables):
        return subprocess.run([*arguments, *bootstrap_args], capture_output=True, text=True, encoding="utf-8", input=input_text,
                              env={**environment, **variables}, timeout=120)

    def start(**variables):
        return subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", env={**environment, **variables})

    run.start = start

    return run, destination, Path(environment["FAKE_COMMANDS"])


def make_old_project(directory, version="0.1.3"):
    (directory / "scripts").mkdir(parents=True)
    (directory / "scripts/deploy_and_accept.py").write_text("# old source retained\n", encoding="utf-8")
    (directory / "pyproject.toml").write_text(
        f'[project]\nname = "local-speaker-lab"\nversion = "{version}"\n', encoding="utf-8")
    (directory / "user-config.toml").write_text("private custom config", encoding="utf-8")
    return directory


@pytest.mark.parametrize("accept_defaults", [False, True])
def test_upgrade_prompts_read_separate_terminal_and_ignore_standard_input(tmp_path, accept_defaults):
    old = make_old_project(tmp_path / "old project")
    new = tmp_path / "new project"
    selections = tmp_path / "terminal-selections.txt"
    selections.write_text("\n\n" if accept_defaults else f"{old.as_posix()}\n{new.as_posix()}\n", encoding="utf-8", newline="\n")
    source = BOOTSTRAP.read_text(encoding="utf-8")
    selector = "select_upgrade_directories() {" + source.split("select_upgrade_directories() {", 1)[1].split("\nif ((UPGRADE));", 1)[0]
    # A file substitutes only the controlling terminal, keeping the real read,
    # default handling, path safety checks and independent stdin untouched.
    selector = selector.replace('exec 4<>/dev/tty', 'exec 4< "$SELECTIONS"').replace(">&4", ">&2")
    script = tmp_path / "directory-selector.sh"
    script.write_text(
        'set -Eeuo pipefail\nOLD_DIRECTORY="$DEFAULT_OLD"\nDESTINATION="$DEFAULT_NEW"\n'
        'OLD_DIRECTORY_SET=0\nDESTINATION_SET=0\n'
        'die() { printf "%s\\n" "$1" >&2; exit "${2:-1}"; }\n'
        'redact() { printf "%s" "$1"; }\n' + selector + '\nselect_upgrade_directories\n', encoding="utf-8", newline="\n")
    result = subprocess.run([bash_executable(), str(script)], capture_output=True, text=True,
                            input="stdin must not answer directory prompts\n", encoding="utf-8",
                            env=dict(os.environ, SELECTIONS=selections.as_posix(),
                                     DEFAULT_OLD=old.as_posix() if accept_defaults else "wrong old default",
                                     DEFAULT_NEW=new.as_posix() if accept_defaults else "wrong new default"), timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "old project" in result.stdout and "new project" in result.stdout
    assert "stdin must not answer" not in result.stdout
    assert "Old 0.1.3 directory" in result.stderr and "New 0.1.4 directory" in result.stderr
    assert not new.exists()


def test_upgrade_first_and_repeat_run_keep_old_sources_and_pass_report_to_runner(tmp_path):
    old = make_old_project(tmp_path / "sources" / "old project")
    old_contents = {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    run, destination, _ = bootstrap_harness(tmp_path)
    options = ["--upgrade", "--old-directory", str(old)]
    for _ in range(2):
        result = run(bootstrap_args=options)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Upgrade versions verified" in result.stdout
        assert "Upgrade report:" in result.stdout
    assert {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()} == old_contents
    assert (destination / "scripts/upgrade_cache.py").is_file()
    reports = list((tmp_path / "speaker-lab/deployment_logs").glob("*/upgrade-report.json"))
    assert len(reports) == 2
    for report in reports:
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASS"
        assert (report.parent / "child-upgrade-report.txt").is_file()


@pytest.mark.parametrize("relationship", ["same", "new-inside-old", "old-inside-new"])
def test_upgrade_refuses_same_and_nested_directories_before_any_old_write(tmp_path, relationship):
    run, destination, commands = bootstrap_harness(tmp_path)
    old = destination if relationship == "same" else (
        destination.parent if relationship == "new-inside-old" else destination / "old")
    # The parent itself cannot be our temp root: create it for this selected old
    # project, keeping the harness scripts separate from old source snapshots.
    if relationship == "new-inside-old":
        old = make_old_project(tmp_path / "old")
        selected_new = old / "new"
    else:
        old = make_old_project(old)
        selected_new = destination
    before = {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()}
    result = run(bootstrap_args=["--upgrade", "--old-directory", str(old), "--destination", str(selected_new)])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "must not contain one another" in result.stdout
    assert not commands.exists()
    assert {path.relative_to(old): path.read_bytes() for path in old.rglob("*") if path.is_file()} == before


def test_upgrade_rejects_wrong_old_version_before_source_extraction(tmp_path):
    run, destination, _ = bootstrap_harness(tmp_path)
    old = make_old_project(tmp_path / "old", version="0.1.2")
    result = run(bootstrap_args=["--upgrade", "--old-directory", str(old)])
    assert result.returncode != 0
    assert "Expected local-speaker-lab 0.1.3" in result.stdout
    assert not destination.exists()


def test_upgrade_refuses_new_directory_containing_old_source_version(tmp_path):
    run, destination, _ = bootstrap_harness(tmp_path)
    old = make_old_project(tmp_path / "old")
    make_old_project(destination)
    original = (destination / "pyproject.toml").read_bytes()
    result = run(bootstrap_args=["--upgrade", "--old-directory", str(old)])
    assert result.returncode != 0
    assert "Expected local-speaker-lab 0.1.4" in result.stdout
    assert (destination / "pyproject.toml").read_bytes() == original
    assert not (destination / "scripts/upgrade_cache.py").exists()


def test_upgrade_wrong_zip_version_never_writes_project_files(tmp_path, monkeypatch):
    members = {**REQUIRED, "local-speaker-lab/scripts/upgrade_cache.py": "print('helper')\n"}
    members["local-speaker-lab/pyproject.toml"] = members["local-speaker-lab/pyproject.toml"].replace("0.1.4", "0.1.3")
    archive = make_archive(tmp_path / "wrong-version.zip", members=members)
    destination = tmp_path / "project"
    monkeypatch.setattr(sys, "argv", ["extractor", str(archive), str(destination), "1"])
    with pytest.raises(RuntimeError, match="0.1.4 ZIP"):
        exec(compile(EXTRACTOR, str(BOOTSTRAP) + ":safe-extract", "exec"), {})
    assert not destination.exists()


def test_upgrade_missing_terminal_or_explicit_old_path_gets_failure_report(tmp_path):
    run, destination, commands = bootstrap_harness(tmp_path)
    result = run(bootstrap_args=["--upgrade"])
    assert result.returncode == 2
    assert "directory selection requires a terminal" in result.stdout
    assert not commands.exists() and not destination.exists()
    report, = (tmp_path / "speaker-lab/deployment_logs").glob("*/bootstrap-report.json")
    assert json.loads(report.read_text(encoding="utf-8"))["stage"] == "upgrade-directories"


def test_upgrade_cache_failure_shows_error_and_report_paths_without_running_runner(tmp_path):
    run, _, commands = bootstrap_harness(tmp_path)
    old = make_old_project(tmp_path / "old")
    result = run(bootstrap_args=["--upgrade", "--old-directory", str(old)], FAKE_UPGRADE_EXIT="9")
    assert result.returncode == 9, result.stdout + result.stderr
    assert "cache conflict detected" in result.stdout
    assert "Acceptance has not started or did not complete" in result.stdout
    assert "Bootstrap log:" in result.stdout and "Upgrade report:" in result.stdout
    report, = (tmp_path / "speaker-lab/deployment_logs").glob("*/bootstrap-report.json")
    assert json.loads(report.read_text(encoding="utf-8"))["stage"] == "upgrade-cache"
    assert not (report.parent / "child-ran.txt").exists()
    assert not commands.exists()


def test_first_install_and_repeat_run_execute_full_bootstrap_without_overwriting_cache(tmp_path):
    run, destination, commands = bootstrap_harness(tmp_path)
    secret = "hf_TEST_PRIVATE_BOOTSTRAP_TOKEN"
    first = run(FAKE_PACKAGES_MISSING="1", HF_TOKEN=secret)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "apt-get update" in commands.read_text(encoding="utf-8")
    assert "apt-get install -y" in commands.read_text(encoding="utf-8")
    cached = destination / ".cache/model.bin"
    cached.parent.mkdir()
    cached.write_bytes(b"keep downloaded weights")
    command_count = len(commands.read_text(encoding="utf-8").splitlines())
    repeated = run(FAKE_PACKAGES_MISSING="0", HF_TOKEN=secret)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert "apt-get" not in "\n".join(commands.read_text(encoding="utf-8").splitlines()[command_count:])
    assert cached.read_bytes() == b"keep downloaded weights"
    reports = list((tmp_path / "deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text(encoding="utf-8"))["status"] == "COMPLETED" for path in reports)
    assert secret not in first.stdout + first.stderr + repeated.stdout + repeated.stderr
    for report in reports:
        assert secret not in (report.parent / "bootstrap.log").read_text(encoding="utf-8")


def test_bootstrap_still_rejects_simulated_python311_before_extraction(tmp_path):
    run, destination, _ = bootstrap_harness(tmp_path)
    result = run(FAKE_PYTHON_MINOR="11")
    assert result.returncode != 0
    assert "Python 3.12 is required" in result.stdout + result.stderr
    assert not destination.exists()
    report, = (tmp_path / "deployment_logs").glob("*/bootstrap-report.json")
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "FAILED"
    assert not (report.parent / "child-ran.txt").exists()


def test_driver_reboot_exit_is_preserved_and_rerun_reuses_extracted_sources(tmp_path):
    run, destination, _ = bootstrap_harness(tmp_path)
    reboot = run(FAKE_RUNNER_EXIT="20")
    assert reboot.returncode == 20, reboot.stdout + reboot.stderr
    reports = list((tmp_path / "deployment_logs").glob("*/bootstrap-report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status"] == "REBOOT_REQUIRED" and report["stage"] == "deployment-and-acceptance"
    restored = run(FAKE_RUNNER_EXIT="0")
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert "Reused matching project sources" in restored.stdout
    assert (destination / "scripts/deploy_and_accept.py").is_file()
    reports = list((tmp_path / "deployment_logs").glob("*/bootstrap-report.json"))
    assert sorted(json.loads(path.read_text(encoding="utf-8"))["status"] for path in reports) == ["COMPLETED", "REBOOT_REQUIRED"]


@pytest.mark.skipif(os.name != "posix", reason="Requires native POSIX signal semantics")
def test_sigterm_to_bootstrap_stops_its_runner_before_writing_failure_report(tmp_path):
    run, _, _ = bootstrap_harness(tmp_path)
    process = run.start(FAKE_RUNNER_SLEEP="1")
    try:
        deadline = time.monotonic() + 30
        while not list((tmp_path / "deployment_logs").glob("*/child-ran.txt")):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 143, stdout + stderr
        reports = list((tmp_path / "deployment_logs").glob("*/bootstrap-report.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        assert report["status"] == "FAILED" and "Terminated" in report["error"]
        assert (reports[0].parent / "child-terminated.txt").is_file()
        assert not (reports[0].parent / "child-continued.txt").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=20)
