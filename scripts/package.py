#!/usr/bin/env python3
"""Package source only; exclude virtualenvs, credentials, models and user recordings."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "README.zh-CN.md", "PUBLICATION.md", "docs/figures/overview.svg", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "DEPLOYMENT.md", "DEPLOYMENT_REVIEW.md", "deploy_ubuntu.sh", "pyproject.toml", ".gitignore", ".gitattributes", ".editorconfig", ".github", "docs", "configs", "requirements", "scripts", "src/speaker_lab", "tests"]
# Include the checked-in examples referenced by README, not user recordings,
# exported datasets, or subsequently edited personal glossary files.
INCLUDE += ["experiments/glossary/README.md", "experiments/glossary/DESIGN_NOTES.md",
            "experiments/glossary/ASCEND_PILOT_RESULTS.md", "experiments/glossary/ASCEND_TRAIN_GLOSSARY_RESULTS.md",
            "experiments/glossary/ASCEND_200_RESULTS.md",
            "experiments/glossary/ASCEND_SIX_EVENT_REVIEW.md",
            "experiments/glossary/ASCEND_REMAINING_EXECUTION.md",
            "experiments/glossary/personal.example.json",
            "experiments/glossary/unrelated.example.json", "experiments/glossary/manifest.example.jsonl"]
# Frozen, shareable protocol inputs referenced by the repository documentation.
# Keep this list explicit: the surrounding experiment directory also holds audio
# and local data which must never enter a source release.
INCLUDE += ["experiments/glossary/protocols/ascend-val200-train-vocab30-v2/" + name
            for name in ("README.md", "protocol.json", "analysis_plan.json",
                         "selected_ids.json", "correct.json", "unrelated.json")]
INCLUDE += ["experiments/glossary/protocols/ascend-val-remaining725-train-vocab30-v1/" + name
            for name in ("README.md", "protocol.json", "analysis_plan.json", "selected_ids.json",
                         "correct.json", "unrelated.json", "config.toml", "environment.json",
                         "selection_audit.json", "FREEZE.json")]
INCLUDE += ["experiments/glossary/audits/ascend-remaining-v1/" + name
            for name in ("six_event_audit.json", "preservation_before.json", "execution_log.jsonl")]
# Reviewed DuplexLab documentation and aggregate evidence only. Keep local
# replay pages, recordings, predictions and future unreviewed reports excluded.
INCLUDE += ["experiments/duplexlab/README.md", "experiments/duplexlab/README.zh-CN.md"]
INCLUDE += ["experiments/scheduling/README.md", "experiments/scheduling/README.zh-CN.md"]
INCLUDE += ["experiments/scheduling/" + name for name in ("RESULTS.md", "SUMMARY.json", "ENVIRONMENT.json")]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-mac-cpu/" + name
            for name in ("RESULTS.md", "VALIDATION.md", "SUMMARY.json", "INPUTS.json")]
INCLUDE += ["experiments/duplexlab/CONTINUATION_PROTOCOL.md"]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-continuation/" + name
            for name in ("RESULTS.md", "VALIDATION.md", "LANGUAGE_COMPARISON.json", "SUMMARY.json",
                         "test-errors.svg", "test-errors.png", "training-loss.svg", "training-loss.png")]
INCLUDE += ["experiments/duplexlab/MULTISEED_PROTOCOL.md", "experiments/duplexlab/MULTISEED_SENSITIVITY.md"]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-multiseed/" + name
            for name in ("RESULTS.md", "VALIDATION.md", "REPRODUCE.md", "REGRESSION_ANALYSIS.md",
                         "SUMMARY.json", "REGRESSIONS.json", "REGRESSION_STABILITY.json",
                         "seed-effects.svg", "seed-effects.png", "speaker-effects.svg", "speaker-effects.png")]
INCLUDE += ["experiments/duplexlab/COVERAGE_PROTOCOL.md"]
INCLUDE += ["experiments/duplexlab/DISFLUENCY_PROTOCOL.md", "experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md"]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-coverage/" + name
            for name in ("DESIGN.md", "RESULTS.md", "VALIDATION.md", "REPRODUCE.md",
                         "SUMMARY.json", "COVERAGE_AUDIT.json", "TRAIN_DEV_FEATURES.json",
                         "test-seeds.svg", "test-seeds.png", "coverage-design.svg", "coverage-design.png")]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-disfluency/" + name
            for name in ("DESIGN.md", "RESULTS.md", "VALIDATION.md", "REPRODUCE.md", "REGRESSION_ANALYSIS.md",
                         "SUMMARY.json", "COVERAGE_AUDIT.json", "TRAIN_DEV_FEATURES.json", "PRIOR_ANALYSIS.json",
                         "DISFLUENCY_GROUPS.json", "AUXDEV_RESULTS.json",
                         "test-seeds.svg", "test-seeds.png", "coverage-design.svg", "coverage-design.png")]
INCLUDE += ["experiments/duplexlab/DELETION_PROTOCOL.md", "experiments/duplexlab/DELETION_EVAL_ADDENDUM.md"]
INCLUDE += ["experiments/duplexlab/reports/2026-10-20-deletion/" + name
            for name in ("DESIGN.md", "RESULTS.md", "VALIDATION.md", "REPRODUCE.md", "REGRESSION_ANALYSIS.md",
                         "SUMMARY.json", "PRIOR_ANALYSIS.json", "COVERAGE_AUDIT.json", "DELETION_GROUPS.json",
                         "test-seeds.svg", "test-seeds.png", "deletion-contrasts.svg", "deletion-contrasts.png")]
INCLUDE += ["experiments/scheduling/native-injection/" + name
            for name in ("PLAN.md", "RESULTS.md", "SUMMARY.json", "BACKEND.json", "server-native.patch",
                         "transcribe_probe.py", "BASELINE_REPORT.md", "BASELINE_SUMMARY.json", "BASELINE_ENVIRONMENT.json",
                         "LATENCY_PLAN.md", "LATENCY_RESULTS.md", "LATENCY_SUMMARY.json",
                         "PREFILL_PLAN.md", "PREFILL_RESULTS.md", "PREFILL_SUMMARY.json", "PREFILL_ENVIRONMENT.json", "PREFILL_BACKEND.json", "server-prefill.patch",
                         "CALENDAR_BACKEND.json", "server-calendar.patch",
                         "PRE_CACHE_BACKEND.json", "PRE_CACHE_SUMMARY.json", "PRE_CACHE_ENVIRONMENT.json",
                         "server-native-uncached.patch")]
INCLUDE += ["experiments/scheduling/date-clarity-2026-10-21/" + name
            for name in ("PLAN.md", "RESULTS.md", "SUMMARY.json", "ENVIRONMENT.json")]
INCLUDE += ["experiments/duplexlab/NORMALIZATION_PROTOCOL.md"]
INCLUDE += ["experiments/duplexlab/reports/2026-10-21-normalization/" + name
            for name in ("DESIGN.md", "RESULTS.md", "VALIDATION.md", "REPRODUCE.md", "REGRESSION_ANALYSIS.md",
                         "SUMMARY.json", "NORMALIZATION_DIAGNOSIS.json", "COVERAGE_AUDIT.json", "DELETION_GROUPS.json",
                         "test-seeds.svg", "test-seeds.png", "deletion-contrasts.svg", "deletion-contrasts.png")]
SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".txt", ".md", ".json", ".jsonl", ".yml", ".yaml", ".in", ".html"}
EXCLUDED_PARTS = {"__pycache__", ".cache", ".pytest_cache", ".venvs", "runs", "deployment_logs"}
PRIVATE_NAMES = {".netrc", "credentials.json", "secrets.json", "token", "token.json"}


def _source_link(path):
    # Windows junctions are reparse points but are not reported by is_symlink.
    # lstat works on both supported Python versions, including Windows 3.11.
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def source_files():
    """Return reviewed source paths, without following links into local data.

    Recursion is restricted to the source directories in INCLUDE and known text
    formats. Experiment inputs/reports are individually listed above; adding a
    local recording, log, credential file or new report does not publish it.
    """
    root = ROOT.resolve()
    paths = set()
    for name in INCLUDE:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source entry must be relative to the project: {name}")
        target = root / relative
        if not target.exists() and not target.is_symlink():
            raise FileNotFoundError(f"Required source entry is missing: {name}")
        candidates = target.rglob("*") if target.is_dir() and not _source_link(target) else [target]
        for path in candidates:
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.startswith(".venv") for part in relative.parts):
                continue
            if _source_link(path):
                raise ValueError(f"Source links are not allowed: {relative.as_posix()}")
            if not path.is_file():
                continue
            if any(_source_link(root / Path(*relative.parts[:index]))
                   for index in range(1, len(relative.parts))):
                raise ValueError(f"Source path crosses a link: {relative.as_posix()}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Source path leaves the project: {relative.as_posix()}")
            if path.name in PRIVATE_NAMES or path.name == ".env" or path.name.startswith(".env."):
                continue
            if path != target and path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _zip_timestamp():
    # A fixed timestamp makes the same source snapshot produce identical bytes.
    # SOURCE_DATE_EPOCH supports release tooling without relying on local mtimes.
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    if not 315532800 <= epoch <= 4354819198:
        raise ValueError("SOURCE_DATE_EPOCH must be within the ZIP timestamp range (1980-2107).")
    return time.gmtime(epoch)[:6]


def build_archive(destination: Path | None = None) -> Path:
    destination = destination or ROOT / "dist" / "local-speaker-lab.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths, timestamp = source_files(), _zip_timestamp()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip.tmp", delete=False) as output:
            temporary = Path(output.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                name = "local-speaker-lab/" + path.relative_to(ROOT.resolve()).as_posix()
                entry = zipfile.ZipInfo(name, date_time=timestamp)
                entry.create_system = 3
                mode = 0o100755 if path.suffix == ".sh" else 0o100644
                entry.external_attr = mode << 16
                archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    destination = build_archive()
    checksum = destination.with_suffix(".zip.sha256")
    checksum.write_text(hashlib.sha256(destination.read_bytes()).hexdigest() + "  " + destination.name + "\n",
                        encoding="ascii", newline="\n")
    bootstrap = destination.parent / "deploy_ubuntu.sh"
    shutil.copyfile(ROOT / "deploy_ubuntu.sh", bootstrap)
    print(destination)
    print(checksum)
    print(bootstrap)


if __name__ == "__main__":
    main()
