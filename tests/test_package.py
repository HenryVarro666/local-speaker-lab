"""The transfer archive must contain the documented source and no runtime data."""
import importlib.util
from pathlib import Path
import os
import posixpath
import re
from urllib.parse import unquote, urlsplit
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_source_package", ROOT / "scripts/package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    (project / "README.md").write_text("Source release\n", encoding="utf-8")
    (project / "docs/guide.md").write_text("Reviewed documentation\n", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module, "INCLUDE", ["README.md", "docs"])
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    return module, project


def test_source_archive_has_examples_and_excludes_runtime_data(tmp_path):
    spec = importlib.util.spec_from_file_location("source_package", ROOT / "scripts/package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive_path = module.build_archive(tmp_path / "source.zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        for name in ("README.md", "DEPLOYMENT.md", "DEPLOYMENT_REVIEW.md", "deploy_ubuntu.sh", "scripts/deploy_and_accept.py",
                     "CONTRIBUTING.md", "CHANGELOG.md", ".gitattributes", ".editorconfig", ".github/workflows/tests.yml",
                     "docs/architecture.md", "docs/deployment.md", "docs/development.md", "docs/model-backends.md",
                     "scripts/upgrade_cache.py", "tests/test_upgrade_cache.py",
                     "requirements/locks/diarize-linux-py312.txt",
                     "scripts/smoke_models.py", "experiments/glossary/README.md",
                     "experiments/glossary/personal.example.json"):
            assert "local-speaker-lab/" + name in names
        protocol = "local-speaker-lab/experiments/glossary/protocols/ascend-val200-train-vocab30-v2/"
        assert {name.removeprefix(protocol) for name in names if name.startswith(protocol)} == {
            "README.md", "protocol.json", "analysis_plan.json", "selected_ids.json", "correct.json", "unrelated.json",
        }
        for name in names:
            assert not any(part in name for part in (".venv", ".audit-downloads/", ".cache/", "__pycache__", "/runs/", "deployment_logs/"))
        broken_links = []
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            for link in re.findall(r"\]\(([^)]+)\)", archive.read(name).decode("utf-8")):
                target = urlsplit(link.strip("<>"))
                if target.scheme or target.netloc or not target.path:
                    continue
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(target.path)))
                if resolved not in names and not any(entry.startswith(resolved.rstrip("/") + "/") for entry in names):
                    broken_links.append(f"{name} -> {link}")
        assert not broken_links, "Links missing from source archive:\n" + "\n".join(broken_links)


def test_archive_is_reproducible_across_local_timestamps_and_permissions(source_tree, tmp_path):
    module, project = source_tree
    first = module.build_archive(tmp_path / "one.zip").read_bytes()
    os.utime(project / "README.md", (1700000000, 1700000000))
    (project / "README.md").chmod(0o600)
    assert module.build_archive(tmp_path / "two.zip").read_bytes() == first


def test_source_release_excludes_local_credentials_runtime_and_unreviewed_reports(source_tree, tmp_path):
    module, project = source_tree
    for name in (".env", ".env.production", "credentials.json", "secrets.json", "token.json",
                 "recording.wav", "worker.log", "model.bin", "module.pyc"):
        (project / "docs" / name).write_bytes(b"local only")
    cache = project / "docs/.cache"
    cache.mkdir()
    (cache / "private.json").write_text("private")
    reports = project / "experiments/glossary"
    reports.mkdir(parents=True)
    (reports / "UNREVIEWED_RESULTS.md").write_text("local notes")
    with zipfile.ZipFile(module.build_archive(tmp_path / "source.zip")) as archive:
        assert set(archive.namelist()) == {"local-speaker-lab/README.md", "local-speaker-lab/docs/guide.md"}


def test_missing_required_source_stops_without_replacing_previous_archive(source_tree, tmp_path):
    module, project = source_tree
    destination = module.build_archive(tmp_path / "source.zip")
    previous = destination.read_bytes()
    (project / "README.md").unlink()
    with pytest.raises(FileNotFoundError, match="Required source entry"):
        module.build_archive(destination)
    assert destination.read_bytes() == previous


def test_failed_read_does_not_replace_previous_archive_or_leave_temporary_files(source_tree, tmp_path, monkeypatch):
    module, project = source_tree
    destination = module.build_archive(tmp_path / "source.zip")
    previous = destination.read_bytes()
    original = Path.read_bytes

    def read(path):
        if path == project / "docs/guide.md":
            raise OSError("simulated source read failure")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(OSError, match="simulated source"):
        module.build_archive(destination)
    assert destination.read_bytes() == previous
    assert not list(destination.parent.glob("*.zip.tmp"))


def test_source_link_is_rejected_before_external_file_is_read(source_tree, tmp_path, monkeypatch):
    module, project = source_tree
    source = project / "docs/guide.md"
    outside = tmp_path / "private.md"
    outside.write_text("private external content")
    try:
        source.unlink()
        source.symlink_to(outside)
    except OSError:
        source.write_text("simulated link")
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == source or original(path))
    with pytest.raises(ValueError, match="Source links are not allowed"):
        module.build_archive(tmp_path / "source.zip")
    assert not (tmp_path / "source.zip").exists()


def test_windows_directory_junction_is_rejected_before_recursion(source_tree, tmp_path, monkeypatch):
    module, project = source_tree
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == project / "docs":
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="Source links are not allowed"):
        module.build_archive(tmp_path / "source.zip")
    assert not (tmp_path / "source.zip").exists()


@pytest.mark.parametrize("epoch", ["invalid", "0", "4354819199"])
def test_invalid_reproducible_timestamp_preserves_previous_archive(source_tree, tmp_path, monkeypatch, epoch):
    module, _ = source_tree
    destination = module.build_archive(tmp_path / "source.zip")
    previous = destination.read_bytes()
    monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
    with pytest.raises(ValueError):
        module.build_archive(destination)
    assert destination.read_bytes() == previous
