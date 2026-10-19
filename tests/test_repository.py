"""Portable documentation and a credential-free CI are part of the source repo."""
import importlib.util
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_documentation_links_resolve_inside_the_source_archive():
    spec = importlib.util.spec_from_file_location("repository_source_package", ROOT / "scripts/package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    included = {path.resolve() for path in package.source_files()}
    documents = sorted(path for path in included if path.suffix.lower() == ".md")
    failures = []
    for document in documents:
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = urlsplit(link.strip("<>"))
            if target.scheme or target.netloc or not target.path:
                continue
            resolved = (document.parent / unquote(target.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {link}")
            elif resolved.is_file() and resolved not in included:
                failures.append(f"{document.relative_to(ROOT)} -> {link} (not packaged)")
            elif resolved.is_dir() and not any(path.is_relative_to(resolved) for path in included):
                failures.append(f"{document.relative_to(ROOT)} -> {link} (directory has no packaged files)")
    assert not failures, "Broken/nonportable repository links:\n" + "\n".join(failures)


def test_ci_has_no_model_credentials_or_privileged_deployment_execution():
    path = ROOT / ".github/workflows/tests.yml"
    # BaseLoader preserves YAML's 'on' key rather than YAML 1.1's boolean alias.
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    core = workflow["jobs"]["tests"]
    assert core["strategy"]["matrix"] == {
        "os": ["ubuntu-latest", "windows-latest"], "python-version": ["3.11", "3.12"],
    }
    core_commands = "\n".join(step.get("run", "") for step in core["steps"])
    assert "python -m pytest" in core_commands
    assert "--junitxml=test-results/pytest.xml" in core_commands
    integration = workflow["jobs"]["integration"]
    assert integration["runs-on"] == "ubuntu-24.04"
    setup = next(step for step in integration["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "3.12"
    expected_actions = {
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9",
    }
    commands = []
    for job in workflow["jobs"].values():
        environment = {**workflow.get("env", {}), **job.get("env", {})}
        assert environment["HF_TOKEN"] == "" and environment["HF_HUB_OFFLINE"] == "1"
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        for step in job["steps"]:
            if "uses" in step:
                assert step["uses"] in expected_actions
                if step["uses"].startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] == "false"
            commands.append(step.get("run", ""))
    commands = "\n".join(commands)
    assert "python scripts/package.py" in commands
    for job in workflow["jobs"].values():
        setup = next(step for step in job["steps"]
                     if step.get("uses", "").startswith("actions/setup-python@"))
        cache_inputs = set(setup["with"]["cache-dependency-path"].split())
        assert {"pyproject.toml", "requirements/bootstrap.txt", "requirements/dev.txt"} <= cache_inputs
        # These checks enforce the build boundary rather than specific step labels:
        # editable installs must use the same explicit backend as the distribution test.
        for step in job["steps"]:
            for command in step.get("run", "").splitlines():
                if "pip install" in command and "--editable" in command:
                    assert "--no-build-isolation" in command
    uploads = [step for step in integration["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
    reports = next(step for step in uploads if step["with"]["name"] == "offline-integration-reports")
    assert reports["if"] == "always()" and reports["with"]["path"] == "test-results/"
    bundle = next(step for step in uploads if step["with"]["name"] == "local-speaker-lab-source")
    assert set(bundle["with"]["path"].split()) == {
        "dist/local-speaker-lab.zip", "dist/local-speaker-lab.zip.sha256", "dist/deploy_ubuntu.sh",
    }
    assert not re.search(r"(?m)^\s*bash\s+(?:scripts/install_ubuntu|deploy_ubuntu)\.sh", commands)
    assert not re.search(r"nvidia-driver|ollama pull|requirements/(?:diarize|separate|asr)\.txt", commands)
    # Installing a model lock indirectly would still violate the no-GPU CI
    # boundary, even if the deploy scripts themselves were never executed.
    requirement_files = set(re.findall(r"(?:-r|--requirement)\s+(\S+)", commands))
    assert requirement_files <= {"requirements/bootstrap.txt", "requirements/dev.txt",
                                 "requirements/locks/app-linux-py312.txt"}
    assert "secrets." not in path.read_text(encoding="utf-8")


def test_optional_gradio_ui_builds_without_starting_a_server(monkeypatch):
    # The portable matrix can omit this optional extra; the locked application
    # integration job installs Gradio and exercises the actual component API.
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    if importlib.util.find_spec("gradio") is None:
        pytest.skip("Optional Gradio UI dependency is not installed")
    from speaker_lab.app import build_ui

    application = build_ui(str(ROOT / "configs/default.toml"))
    try:
        assert application.get_config_file()["components"]
    finally:
        application.close(verbose=False)
