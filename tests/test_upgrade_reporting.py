"""An upgrade must report this attempt, including exact failure log paths."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("upgrade_reporting", ROOT / "scripts/deploy_and_accept.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def upgrade_file(tmp_path, new):
    path = tmp_path / "upgrade-report.json"
    document = {"status": "PASS", "old": {"resolved_path": str(tmp_path / "old"), "version": "0.1.3"},
                "new": {"resolved_path": str(new), "version": "0.1.4"},
                "cache": {"strategy": "linked_old_cache", "reused_old": True}, "old_retained": True}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_upgrade_evidence_and_complete_report_are_displayed_for_this_attempt(tmp_path, monkeypatch, capsys):
    project = tmp_path / "new"
    project.mkdir()
    monkeypatch.setattr(deploy, "ROOT", project)
    path, document = upgrade_file(tmp_path, project)
    monkeypatch.setattr(deploy.Deployment, "execute", lambda *args: None)
    logs = tmp_path / "logs"
    assert deploy.main(["--audio", "input.wav", "--log-dir", str(logs), "--upgrade-report", str(path)]) == 0
    report = json.loads((logs / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["evidence"]["upgrade"] == document
    assert report["upgrade_report_path"] == str(path.resolve())
    assert report["failed_logs"] == []
    output = capsys.readouterr().out
    markdown = (logs / "deployment-report.md").read_text(encoding="utf-8")
    assert markdown in output
    assert str(logs / "deployment-report.md") in output
    assert str(logs / "deployment-report.json") in output
    assert "PENDING_MANUAL_REVIEW" == report["quality_status"]


def test_failure_paths_are_absolute_exist_and_are_printed_after_report(tmp_path, monkeypatch, capsys):
    token = "hf_reportSecret123456789"
    monkeypatch.setenv("HF_TOKEN", token)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "bootstrap.log").write_text("bootstrap output", encoding="utf-8")
    def execute(self, audio, install_driver=False):
        def fail():
            raise RuntimeError("download failed " + token)
        self.step("model-download", fail)
    monkeypatch.setattr(deploy.Deployment, "execute", execute)
    assert deploy.main(["--audio", "input.wav", "--log-dir", str(logs)]) == 1
    report = json.loads((logs / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["failed_logs"] == [str(logs / "01-model-download.log"), str(logs / "deployment.log"), str(logs / "bootstrap.log")]
    output = capsys.readouterr().out
    assert token not in output
    assert (logs / "deployment-report.md").read_text(encoding="utf-8") in output
    for path in report["failed_logs"]:
        assert Path(path).is_absolute() and Path(path).is_file()
        assert "失败日志：" + path in output


@pytest.mark.parametrize("change", ["failed", "other-directory", "old-version", "new-version"])
def test_bad_upgrade_report_stops_before_installing_or_inference(tmp_path, monkeypatch, change):
    project = tmp_path / "new"
    project.mkdir()
    monkeypatch.setattr(deploy, "ROOT", project)
    path, document = upgrade_file(tmp_path, project)
    if change == "failed":
        document["status"] = "FAIL"
    elif change == "other-directory":
        document["new"]["resolved_path"] = str(tmp_path / "other")
    elif change == "old-version":
        document["old"]["version"] = "0.1.2"
    else:
        document["new"]["version"] = "0.1.3"
    path.write_text(json.dumps(document), encoding="utf-8")
    executed = []
    monkeypatch.setattr(deploy.Deployment, "execute", lambda *args: executed.append(True))
    logs = tmp_path / "logs"
    assert deploy.main(["--audio", "input.wav", "--log-dir", str(logs), "--upgrade-report", str(path)]) == 1
    assert executed == []
    report = json.loads((logs / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["steps"][0]["name"] == "upgrade-report" and report["steps"][0]["status"] == "FAIL"


def test_old_shell_paths_cannot_override_new_sources_or_reported_cache(tmp_path, monkeypatch):
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "PYTHONPATH"):
        monkeypatch.setenv(name, "/old/project/" + name)
    monkeypatch.setenv("HF_HOME", "/old/custom-hf-home")
    run = deploy.Deployment(tmp_path, tmp_path / "logs")
    assert run.env["PYTHONPATH"] == str(tmp_path / "src")
    assert run.env["HF_HOME"] == str(tmp_path / ".cache/huggingface")
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        assert name not in run.env


def test_actual_upgrade_helper_report_is_consumed_and_recording_report_path_printed(tmp_path, monkeypatch, capsys):
    helper_spec = importlib.util.spec_from_file_location("cache_helper_integration", ROOT / "scripts/upgrade_cache.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    old, new = tmp_path / "old", tmp_path / "new"
    for directory, version in ((old, "0.1.3"), (new, "0.1.4")):
        directory.mkdir()
        (directory / "pyproject.toml").write_text(f'[project]\nname="local-speaker-lab"\nversion="{version}"\n', encoding="utf-8")
    monkeypatch.setattr(deploy, "ROOT", new)
    logs = tmp_path / "logs"
    logs.mkdir()
    upgrade_report = logs / "upgrade-report.json"
    assert helper.main(["--old", str(old), "--new", str(new), "--report", str(upgrade_report)]) == 0
    recording_report = logs / "recording-acceptance/acceptance-report.json"
    def execute(self, audio, install_driver=False):
        recording_report.parent.mkdir()
        recording_report.write_text('{"automatic_checks":"PASS"}', encoding="utf-8")
    monkeypatch.setattr(deploy.Deployment, "execute", execute)
    assert deploy.main(["--audio", "input.wav", "--log-dir", str(logs), "--upgrade-report", str(upgrade_report)]) == 0
    report = json.loads((logs / "deployment-report.json").read_text(encoding="utf-8"))
    assert report["evidence"]["upgrade"]["cache"]["strategy"] == "download_missing_cache"
    assert report["recording_acceptance_report_path"] == str(recording_report.resolve())
    assert "录音验收报告：" + str(recording_report.resolve()) in capsys.readouterr().out
