"""Offline upgrades must preserve old installations and existing model data."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upgrade_cache", ROOT / "scripts/upgrade_cache.py")
upgrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upgrade)


def project(root, version):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "local-speaker-lab"\nversion = "{version}"\n', encoding="utf-8")
    return root


@pytest.fixture
def installations(tmp_path):
    return project(tmp_path / "old", "0.1.3"), project(tmp_path / "new", "0.1.4")


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314 and target.is_dir():
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, capture_output=True, text=True)
            return
        pytest.skip(f"Directory symlinks unavailable: {exc}")


def test_links_partial_cache_and_preserves_every_old_file_and_credentials(installations, monkeypatch):
    original_link = Path.symlink_to

    def directory_link(path, target, **kwargs):
        try:
            return original_link(path, target, **kwargs)
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
                raise
            # A directory junction exercises real resolve/read/idempotency
            # semantics when this Windows test host cannot create symlinks.
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(path), str(target)],
                           check=True, capture_output=True, text=True)

    monkeypatch.setattr(Path, "symlink_to", directory_link)
    old, new = installations
    cache = old / ".cache/huggingface"
    cache.mkdir(parents=True)
    (cache / "model.incomplete").write_bytes(b"partial download")
    (cache / "token").write_text("hf_abcdefghijk12345", encoding="utf-8")
    (old / "settings.toml").write_text("custom old configuration", encoding="utf-8")
    (old / ".venvs").mkdir()
    (old / ".venvs/old-python").write_bytes(b"not portable")
    before = snapshot(old)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "PASS", result
    assert result["cache"]["strategy"] == "link_old_cache"
    assert result["cache"]["reused_old"] is True
    assert (new / ".cache/huggingface").resolve() == cache.resolve()
    assert snapshot(old) == before
    assert not (new / ".venvs").exists()
    assert not (new / "settings.toml").exists()
    assert not (new / "token").exists()
    again = upgrade.run_upgrade(old, new)
    assert again["status"] == "PASS"
    assert again["cache"]["strategy"] == "reuse_old_existing_link"
    assert snapshot(old) == before


def test_absent_old_cache_is_not_an_error_or_old_write(installations):
    old, new = installations
    before = snapshot(old)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "PASS"
    assert result["cache"]["strategy"] == "download_missing_cache"
    assert snapshot(old) == before
    assert not (old / ".cache").exists()


@pytest.mark.parametrize("entry", ["directory", "symlink"])
def test_existing_new_cache_is_kept_without_merging(installations, tmp_path, entry):
    old, new = installations
    old_cache = old / ".cache/huggingface"
    old_cache.mkdir(parents=True)
    (old_cache / "weights").write_bytes(b"old")
    (new / ".cache").mkdir()
    target = new / ".cache/huggingface" if entry == "directory" else tmp_path / "separate-cache"
    target.mkdir()
    (target / "weights").write_bytes(b"new")
    if entry == "symlink":
        symlink(new / ".cache/huggingface", target)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "PASS"
    assert result["cache"]["strategy"] == "keep_existing_new_cache"
    assert not result["cache"]["reused_old"]
    assert "not selected" in " ".join(result["warnings"])
    assert (target / "weights").read_bytes() == b"new"
    assert (old_cache / "weights").read_bytes() == b"old"


@pytest.mark.parametrize("changed,version", [("old", "0.1.2"), ("new", "0.1.3")])
def test_version_mismatch_stops_before_cache_creation(installations, changed, version):
    old, new = installations
    project(old if changed == "old" else new, version)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert "Expected version" in result["error"]
    assert not (new / ".cache").exists()


def test_unrelated_project_with_matching_version_is_rejected(installations):
    old, new = installations
    (old / "pyproject.toml").write_text('[project]\nname = "unrelated"\nversion = "0.1.3"\n', encoding="utf-8")
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert "Expected local-speaker-lab project" in result["error"]
    assert not (new / ".cache").exists()


@pytest.mark.parametrize("shape", ["same", "new_in_old", "old_in_new", "symlink_alias"])
def test_same_or_nested_paths_are_rejected_before_writes(tmp_path, shape):
    old = project(tmp_path / "old", "0.1.3")
    if shape == "same":
        new = old
    elif shape == "new_in_old":
        new = project(old / "new", "0.1.4")
    elif shape == "old_in_new":
        new = project(tmp_path / "container", "0.1.4")
        old = project(new / "old", "0.1.3")
    else:
        new = tmp_path / "alias"
        symlink(new, old)
    before = snapshot(old)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert "cannot be nested" in result["error"]
    assert snapshot(old) == before


@pytest.mark.parametrize("location", ["old_cache", "old_parent", "new_cache", "new_parent"])
def test_broken_cache_symlinks_are_preserved_and_reported(installations, tmp_path, location):
    old, new = installations
    root = old if location.startswith("old") else new
    if location.endswith("parent"):
        entry = root / ".cache"
    else:
        (root / ".cache").mkdir()
        entry = root / ".cache/huggingface"
    target = tmp_path / "missing"
    symlink(entry, target)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert entry.is_symlink()
    assert entry.readlink() == target


def test_new_cache_parent_cannot_write_into_old_project(installations):
    old, new = installations
    symlink(new / ".cache", old)
    before = snapshot(old)
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert "outside the new project" in result["error"]
    assert snapshot(old) == before
    assert not (old / "huggingface").exists()


@pytest.mark.parametrize("location", ["old_cache", "old_parent", "new_cache", "new_parent"])
def test_dangling_link_detection_without_platform_symlink_privileges(installations, monkeypatch, location):
    old, new = installations
    root = old if location.startswith("old") else new
    if location.endswith("parent"):
        entry = root / ".cache"
    else:
        (root / ".cache").mkdir()
        entry = root / ".cache/huggingface"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == entry or original(path))
    result = upgrade.run_upgrade(old, new)
    assert result["status"] == "FAIL"
    assert "Broken" in result["error"]
    assert not entry.exists()


def test_failed_cli_writes_redacted_report_and_nonzero_exit(tmp_path, monkeypatch, capsys):
    token = "custom-token-secret-without-prefix"
    monkeypatch.setenv("HF_TOKEN", token)
    old = project(tmp_path / token, "wrong")
    new = project(tmp_path / "new", "0.1.4")
    destination = tmp_path / "logs/report.json"
    code = upgrade.main(["--old", str(old), "--new", str(new), "--report", str(destination)])
    assert code == 1
    raw = destination.read_text(encoding="utf-8")
    assert token not in raw
    assert "[REDACTED]" in raw
    assert json.loads(raw)["status"] == "FAIL"
    assert token not in capsys.readouterr().out
    assert list(destination.parent.iterdir()) == [destination]


def test_hf_shaped_secret_redacted_without_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token = "hf_AbcDef0123456789"
    old = project(tmp_path / token, "wrong")
    new = project(tmp_path / "new", "0.1.4")
    destination = tmp_path / "report.json"
    assert upgrade.main(["--old", str(old), "--new", str(new), "--report", str(destination)]) == 1
    assert token not in destination.read_text(encoding="utf-8")
    assert token not in capsys.readouterr().out


def test_report_destination_in_old_project_is_rejected(installations, capsys):
    old, new = installations
    before = snapshot(old)
    assert upgrade.main(["--old", str(old), "--new", str(new), "--report", str(old / "report.json")]) == 1
    assert snapshot(old) == before
    assert "outside the retained old project" in capsys.readouterr().err


def test_report_guard_uses_real_old_path_before_token_redaction(tmp_path, monkeypatch, capsys):
    token = "hf_SecretTokenAsDirectory123"
    monkeypatch.setenv("HF_TOKEN", token)
    old = project(tmp_path / token, "0.1.3")
    new = project(tmp_path / "new", "0.1.4")
    destination = old / "report.json"
    assert upgrade.main(["--old", str(old), "--new", str(new), "--report", str(destination)]) == 1
    assert not destination.exists()
    assert token not in capsys.readouterr().err


def test_reports_are_atomically_replaced(installations, tmp_path):
    old, new = installations
    destination = tmp_path / "report.json"
    destination.write_text("obsolete", encoding="utf-8")
    assert upgrade.main(["--old", str(old), "--new", str(new), "--report", str(destination)]) == 0
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert not list(tmp_path.glob(".upgrade-cache-*.tmp"))
