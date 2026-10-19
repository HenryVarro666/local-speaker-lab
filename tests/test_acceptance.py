import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acceptance_script", ROOT / "scripts/acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "run"
    for stage in acceptance.STAGES:
        save(run / "work" / stage / "request.json", {"options": {"device": "cuda:0"}})
        result = {"jobs": [{"id": "overlap_0", "sources": ["a.wav", "b.wav"]}]} if stage == "separate" else {}
        save(run / "work" / stage / "response.json", {"ok": True, "result": result})
    save(run / "manifest.json", {"status": "completed", "models": {stage: stage for stage in acceptance.STAGES}, "warnings": []})
    save(run / "speaker_matching.json", {"jobs": [{"id": "overlap_0", "speakers": ["SPEAKER_01", "SPEAKER_00"]}],
                                        "uncertain": [{"job_id": "overlap_0"}]})
    samples = np.linspace(-0.1, 0.1, 16000, dtype="float32")
    sf.write(run / "work" / "input.wav", samples, 16000, subtype="FLOAT")
    for speaker in acceptance.SPEAKERS:
        sf.write(run / f"{speaker}.wav", samples, 16000, subtype="FLOAT")
    rows = [{"id": f"seg_{index + 1:05d}", "start": index * 0.3, "end": index * 0.3 + 0.2,
             "speaker": speaker, "text_raw": "真实字词", "words": [{"text": "字词", "start": index * 0.3, "end": index * 0.3 + 0.2}]}
            for index, speaker in enumerate(sorted(acceptance.SPEAKERS))]
    save(run / "segments.json", {"segments": rows})
    (run / "summary.md").write_text("事实引用 [seg_00001 00:00:00]。", encoding="utf-8")
    return run


def test_valid_run_passes_but_keeps_quality_review_explicit(completed_run):
    report = acceptance.verify_run(completed_run)
    assert report["automatic_checks"] == "PASS"
    assert report["separation_jobs"] == 1
    assert report["uncertain_matches"]
    assert report["manual_review_required"]


def test_completed_status_does_not_hide_skipped_separation(completed_run):
    (completed_run / "work" / "separate" / "response.json").unlink()
    with pytest.raises(RuntimeError, match="did not run"):
        acceptance.verify_run(completed_run)


def test_completed_status_does_not_hide_summary_failure(completed_run):
    (completed_run / "summary.md").unlink()
    with pytest.raises(RuntimeError, match="Summary is missing"):
        acceptance.verify_run(completed_run)


def test_cpu_request_cannot_pass_gpu_acceptance(completed_run):
    save(completed_run / "work" / "match" / "request.json", {"options": {"device": "cpu"}})
    with pytest.raises(RuntimeError, match="not requested on CUDA"):
        acceptance.verify_run(completed_run)


def test_track_alignment_is_checked(completed_run):
    sf.write(completed_run / "SPEAKER_00.wav", np.ones(8000, dtype="float32"), 16000, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="aligned"):
        acceptance.verify_run(completed_run)


@pytest.mark.parametrize("corruption", ["no_words", "nan_word", "wrong_speaker", "bad_citation"])
def test_transcript_and_summary_contracts(completed_run, corruption):
    document = json.loads((completed_run / "segments.json").read_text(encoding="utf-8"))
    if corruption == "no_words":
        document["segments"][0]["words"] = []
    elif corruption == "nan_word":
        document["segments"][0]["words"][0]["start"] = float("nan")
    elif corruption == "wrong_speaker":
        document["segments"][1]["speaker"] = "SPEAKER_00"
    else:
        (completed_run / "summary.md").write_text("[seg_99999]", encoding="utf-8")
    save(completed_run / "segments.json", document)
    with pytest.raises(RuntimeError):
        acceptance.verify_run(completed_run)


def test_existing_run_mode_does_not_start_models(completed_run):
    assert acceptance.main(["--run-dir", str(completed_run)]) == 0
    assert (completed_run / "acceptance-report.json").is_file()


@pytest.mark.parametrize("failure", [False, True])
def test_recording_mode_records_exact_result_and_stops_monitor(completed_run, tmp_path, monkeypatch, failure):
    import speaker_lab.pipeline as pipeline
    monitors = []
    class Monitor:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            monitors.append(self)
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, timeout):
            return self.returncode
    def run(*args, **kwargs):
        if failure:
            raise RuntimeError("injected worker failure")
        return {"run_dir": str(completed_run), "segments": []}
    monkeypatch.setattr(acceptance.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(acceptance.subprocess, "Popen", Monitor)
    monkeypatch.setattr(pipeline, "run_pipeline", run)
    log_dir = tmp_path / "exact-acceptance-log"
    assert acceptance.main(["test.wav", "--log-dir", str(log_dir)]) == int(failure)
    assert len(monitors) == 1 and monitors[0].returncode == 0
    if not failure:
        result = json.loads((log_dir / "result.json").read_text(encoding="utf-8"))
        assert result["run_dir"] == str(completed_run)
        assert (log_dir / "acceptance-report.json").is_file()
