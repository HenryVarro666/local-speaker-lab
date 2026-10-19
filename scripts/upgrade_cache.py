#!/usr/bin/env python3
"""Share model caches when preserving a 0.1.3 installation beside 0.1.4.

This bootstrap helper uses the standard library only. It never copies credentials,
configuration, or virtual environments and never changes an existing cache entry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib


def _redact(value):
    if isinstance(value, str):
        token = os.environ.get("HF_TOKEN", "")
        if token:
            value = value.replace(token, "[REDACTED]")
        return re.sub(r"hf_[A-Za-z0-9]{8,}", "[REDACTED]", value)
    if isinstance(value, dict):
        return {_redact(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_redact(item) for item in value]
    return value


def _resolved(path):
    return Path(path).expanduser().absolute().resolve()


def _version(root, expected):
    if not root.is_dir():
        raise RuntimeError(f"Project directory does not exist: {root}")
    metadata = root / "pyproject.toml"
    if not metadata.is_file():
        raise RuntimeError(f"Project metadata is missing: {metadata}")
    with metadata.open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    if project.get("name") != "local-speaker-lab":
        raise RuntimeError(f"Expected local-speaker-lab project metadata at {root}")
    version = project.get("version")
    if version != expected:
        raise RuntimeError(f"Expected version {expected} at {root}, found {version!r}")
    return version


def _cache_directory(path, label):
    """Distinguish missing caches from broken links, including a linked parent."""
    for component in reversed((path, *path.parents)):
        if component.is_symlink() and not component.exists():
            raise RuntimeError(f"Broken {label} symlink; preserve and repair it before retrying: {component}")
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"{label} must be a directory: {path}")
    return path.resolve() if path.is_dir() else None


def _new_cache_parent(new):
    cache_parent = new / ".cache"
    resolved_parent = cache_parent.resolve()
    if not resolved_parent.is_relative_to(new):
        raise RuntimeError(f"New .cache parent resolves outside the new project: {cache_parent}")
    if cache_parent.exists() and not cache_parent.is_dir():
        raise RuntimeError(f"New .cache parent is not a directory: {cache_parent}")
    for component in (cache_parent,):
        if component.is_symlink() and not component.exists():
            raise RuntimeError(f"Broken new cache parent symlink: {component}")
    return cache_parent


def _initial_report(old, new):
    return {
        "status": "FAIL",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "old": {"path": str(Path(old).expanduser().absolute())},
        "new": {"path": str(Path(new).expanduser().absolute())},
        "old_retained": True,
        "copied_configuration": False,
        "copied_virtual_environments": False,
        "credentials_written": False,
        "warnings": [],
        "cache": {},
        "shared_caches": {
            "clearvoice": {
                "action": "reuse_default_user_cache",
                "path": str(Path.home() / ".cache/speaker-lab/clearvoice"),
            },
            "ollama": {"action": "reuse_existing_system_service_model_store"},
        },
        "retention_dependency": None,
    }


def run_upgrade(old_path, new_path):
    """Return a report, including a safe error on failure, without writing it."""
    report = _initial_report(old_path, new_path)
    try:
        old, new = _resolved(old_path), _resolved(new_path)
        report["old"]["resolved_path"] = str(old)
        report["new"]["resolved_path"] = str(new)
        if old.is_relative_to(new) or new.is_relative_to(old):
            raise RuntimeError("Old and new project directories must be distinct and cannot be nested, including symlink aliases")
        report["old"]["version"] = _version(old, "0.1.3")
        report["new"]["version"] = _version(new, "0.1.4")
        cache_parent = _new_cache_parent(new)
        old_cache, new_cache = old / ".cache/huggingface", cache_parent / "huggingface"
        old_target = _cache_directory(old_cache, "old HF cache")
        new_target = _cache_directory(new_cache, "new HF cache")
        report["cache"].update({
            "old_entry": str(old_cache),
            "link_path": str(new_cache),
            "reused_old": False,
        })
        if new_target is not None:
            same = old_target is not None and old_target == new_target
            report["cache"].update({
                "strategy": "reuse_old_existing_link" if same else "keep_existing_new_cache",
                "resolved_target": str(new_target),
                "reused_old": same,
            })
            if not same:
                report["warnings"].append("New HF cache already exists and is preserved; the old HF cache is not selected. Missing model files may download into the existing new cache.")
        elif old_target is None:
            report["cache"].update({
                "strategy": "download_missing_cache",
                "resolved_target": str(new_cache.resolve()),
            })
            report["warnings"].append("No old HF cache exists; missing models will download into the new project's cache.")
        else:
            # Only a previously absent directory entry may be created. Existing
            # cache directories, files, and even broken links remain untouched.
            cache_parent.mkdir(parents=True, exist_ok=True)
            _new_cache_parent(new)
            new_cache.symlink_to(old_target, target_is_directory=True)
            report["cache"].update({
                "strategy": "link_old_cache",
                "resolved_target": str(old_target),
                "reused_old": True,
            })
        if report["cache"]["reused_old"]:
            report["retention_dependency"] = {
                "target": report["cache"]["resolved_target"],
                "message": "Keep this cache target available. The new version shares its model downloads; removing or moving it breaks the new cache link.",
            }
            report["warnings"].append("The complete HF_HOME is shared through a symlink. Existing authentication files remain at their original location; no token is copied or saved by this helper.")
        report["warnings"].append("Old configuration and virtual environments are preserved and are not migrated. New environments are installed separately.")
        report["status"] = "PASS"
    except (OSError, RuntimeError, ValueError, tomllib.TOMLDecodeError) as exc:
        report["error"] = str(exc)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return _redact(report)


def write_report(report, destination, *, old_path=None):
    """Atomically save the report, without permitting a write into the old tree."""
    destination = Path(destination).expanduser().absolute()
    # Resolve the actual input before redaction: a secret may itself occur in
    # a directory name, and the diagnostic replacement is not a real path.
    old = _resolved(old_path if old_path is not None else report["old"]["path"])
    if destination.resolve().is_relative_to(old):
        raise RuntimeError("Upgrade report must be outside the retained old project")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".upgrade-cache-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(_redact(report), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Preserved 0.1.3 project directory")
    parser.add_argument("--new", required=True, help="Extracted 0.1.4 project directory")
    parser.add_argument("--report", required=True, help="JSON report path outside the old project")
    args = parser.parse_args(argv)
    report = run_upgrade(args.old, args.new)
    try:
        write_report(report, args.report, old_path=args.old)
    except (OSError, RuntimeError, ValueError) as exc:
        print(_redact(f"Cache upgrade FAIL: could not save report: {exc}"), file=sys.stderr)
        return 1
    print(_redact(f"Cache upgrade {report['status']}: {report.get('error', report['cache'].get('strategy', ''))}"))
    print(_redact(f"Upgrade report: {Path(args.report).expanduser().absolute()}"))
    if report["status"] == "PASS":
        print(_redact(f"HF cache: {report['cache']['resolved_target']}"))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
