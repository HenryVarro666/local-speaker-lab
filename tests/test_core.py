import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import numpy as np
import pytest
import soundfile as sf

from speaker_lab import pipeline
from speaker_lab.audio import normalize_audio, read_source
from speaker_lab.config import load_config
from speaker_lab.timeline import activity_intervals, overlap_jobs

ROOT = Path(__file__).resolve().parents[1]


def test_activity_sweep_counts_duplicate_turns_and_keeps_overlap():
    turns = [{"start": 0, "end": 2, "speaker": "z"}, {"start": 1, "end": 3, "speaker": "z"},
             {"start": 2, "end": 4, "speaker": "a"}]
    intervals, mapping = activity_intervals(turns, 4 * 16000)
    assert mapping == {"z": "SPEAKER_00", "a": "SPEAKER_01"}
    assert [(row["start"] / 16000, row["end"] / 16000, row["speakers"]) for row in intervals] == [
        (0, 2, ("SPEAKER_00",)), (2, 3, ("SPEAKER_00", "SPEAKER_01")), (3, 4, ("SPEAKER_01",))]


def test_jobs_cover_overlap_without_gaps_and_include_context():
    rows = [{"start": 16000, "end": 50 * 16000, "speakers": ("SPEAKER_00", "SPEAKER_01")}]
    jobs = overlap_jobs(rows, 52 * 16000, 12 * 16000, 8000)
    assert len(jobs) == 5
    assert jobs[0]["start"] == 16000
    assert jobs[-1]["end"] == 50 * 16000
    assert all(a["end"] == b["start"] for a, b in zip(jobs, jobs[1:]))
    assert all(job["input_end"] - job["input_start"] <= 13 * 16000 for job in jobs)


@pytest.mark.parametrize("turns", [[], [{"start": 0, "end": 1, "speaker": "a"}],
                                    [{"start": float("nan"), "end": 1, "speaker": "a"}]])
def test_invalid_diarization_rejected(turns):
    with pytest.raises(ValueError):
        activity_intervals(turns, 16000)


@pytest.mark.parametrize("replacement", ['"12"', "-1", "nan", "true", "0.0000001"])
def test_invalid_numeric_config_rejected(tmp_path, replacement):
    text = (ROOT / "configs/default.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(text.replace("separation_chunk_seconds = 12.0", "separation_chunk_seconds = " + replacement), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_asr_rejects_nontext_instead_of_fabricating_string():
    jobs = [{"id": "a", "speaker": "SPEAKER_00", "start": 0, "end": 1}]
    result = {"segments": [{"job_id": "a", "speaker": "SPEAKER_00", "start": 0,
                             "end": 1, "text": None}]}
    with pytest.raises(ValueError, match="text must be a string"):
        pipeline._validate_segments(result, jobs, 1, [])


def test_blending_aligns_swapped_outputs_and_smooths_context(tmp_path):
    count = 50 * 16000
    intervals = [{"start": 0, "end": count, "speakers": ("SPEAKER_00", "SPEAKER_01")}]
    jobs = overlap_jobs(intervals, count, 12 * 16000, 8000)
    separated, matched = [], []
    for job in jobs:
        length = job["input_end"] - job["input_start"]
        paths = []
        for index, value in enumerate([0.2, 0.1]):
            path = tmp_path / f"{job['id']}_{index}.wav"
            sf.write(path, np.full(length, value, dtype="float32"), 16000, subtype="FLOAT")
            paths.append(str(path))
        separated.append({"id": job["id"], "sources": paths})
        matched.append({"id": job["id"], "speakers": ["SPEAKER_01", "SPEAKER_00"]})
    audio = np.zeros(count, dtype="float32")
    tracks = {speaker: audio.copy() for speaker in pipeline.SPEAKERS}
    pipeline._blend_jobs(audio, tracks, jobs, {"jobs": separated}, {"jobs": matched})
    np.testing.assert_allclose(tracks["SPEAKER_00"], 0.1, atol=1e-7)
    np.testing.assert_allclose(tracks["SPEAKER_01"], 0.2, atol=1e-7)


def test_missing_clean_reference_is_actionable(tmp_path):
    audio = np.ones(32000, dtype="float32")
    config = load_config(ROOT / "configs/default.toml")
    with pytest.raises(ValueError, match="Upload a clean reference"):
        pipeline._references(audio, [], {}, config, tmp_path)


def test_invalid_backend_audio_rejected(tmp_path):
    path = tmp_path / "wrong_sr.wav"
    sf.write(path, np.ones(8000), 8000)
    with pytest.raises(ValueError, match="16 kHz"):
        read_source(path, 16000)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    text = (ROOT / "configs/default.toml").read_text(encoding="utf-8")
    text = text.replace('root = ".."', f'root = "{ROOT.as_posix()}"')
    for stage in ("diarize", "separate", "match", "transcribe"):
        text = text.replace(f'{stage}_python = "' + {
            "diarize": ".venvs/diarize/bin/python", "separate": ".venvs/separate/bin/python",
            "match": ".venvs/diarize/bin/python", "transcribe": ".venvs/asr/bin/python"}[stage] + '"',
            f'{stage}_python = "{Path(sys.executable).as_posix()}"')
    for name, cls in (("pyannote", "Diarizer"), ("clearvoice", "Separator"),
                      ("pyannote_embedding", "Matcher"), ("qwen", "Transcriber")):
        text = text.replace(f'backend = "{name}"', f'backend = "fake_backend_plugin:{cls}"')
    text = text.replace('output_dir = "runs"', f'output_dir = "{(tmp_path / "runs").as_posix()}"')
    config_path = tmp_path / "test.toml"
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    time = np.arange(6 * 16000) / 16000
    first = (0.1 * np.sin(2 * np.pi * 330 * time) * (time < 4)).astype("float32")
    second = (0.1 * np.sin(2 * np.pi * 660 * time) * (time >= 2)).astype("float32")
    source = tmp_path / "mixture.wav"
    sf.write(source, first + second, 16000, subtype="FLOAT")
    return source, config_path, first, second


def _in_memory_decode(source, destination, max_seconds):
    """Unit-test decoder; subprocess integration and actual FFmpeg are tested separately."""
    data, sr = sf.read(source, dtype="float32")
    assert sr == 16000
    sf.write(destination, data, sr, subtype="FLOAT")
    return data


def test_full_pipeline_real_worker_processes_without_model_downloads(synthetic_run, monkeypatch):
    source, config, first, second = synthetic_run
    monkeypatch.setattr(pipeline, "normalize_audio", _in_memory_decode)
    result = pipeline.run_pipeline(str(source), str(config), summarize=False)
    for speaker, expected in zip(pipeline.SPEAKERS, (first, second)):
        actual, sr = sf.read(result["tracks"][speaker], dtype="float32")
        assert sr == 16000 and len(actual) == len(first)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    document = json.loads(Path(result["segments_path"]).read_text(encoding="utf-8"))
    assert [segment["speaker"] for segment in document["segments"]] == list(pipeline.SPEAKERS)
    assert [(segment["start"], segment["end"]) for segment in document["segments"]] == [(0, 4), (2, 6)]
    assert Path(result["transcript_path"]).is_file()
    with zipfile.ZipFile(result["archive_path"]) as archive:
        assert "segments.json" in archive.namelist()
        assert "work/input.wav" not in archive.namelist()
    assert all((Path(result["run_dir"]) / "work" / stage / "response.json").is_file()
               for stage in ("diarize", "separate", "match", "transcribe"))


def test_summary_failure_keeps_audio_and_transcript(synthetic_run, monkeypatch):
    import speaker_lab.summarize as summary_module
    monkeypatch.setattr(pipeline, "normalize_audio", _in_memory_decode)
    def fail(*args):
        raise RuntimeError("test server unavailable")
    monkeypatch.setattr(summary_module, "summarize_segments", fail)
    source, config, *_ = synthetic_run
    result = pipeline.run_pipeline(str(source), str(config))
    assert result["summary_path"] is None
    assert Path(result["tracks"]["SPEAKER_00"]).is_file()
    assert any("test server unavailable" in warning for warning in result["warnings"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg executable not installed")
def test_real_ffmpeg_decode_keeps_duration(synthetic_run, tmp_path):
    source, _, first, second = synthetic_run
    actual = normalize_audio(source, tmp_path / "decoded.wav", 10)
    np.testing.assert_allclose(actual, first + second, atol=1e-7)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg executable not installed")
def test_real_decode_workers_and_local_http_summary(synthetic_run):
    """Real file/process/HTTP boundaries, synthetic model plugin explicitly only in tests."""
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            response = json.dumps({"done": True, "message": {"content": "要点：两人发言。[seg_00001 00:00:00] [seg_00002 00:00:02]"}}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *args):
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source, config, first, _ = synthetic_run
    config.write_text(config.read_text(encoding="utf-8").replace("http://127.0.0.1:11434", f"http://127.0.0.1:{server.server_port}"), encoding="utf-8")
    try:
        result = pipeline.run_pipeline(str(source), str(config))
        assert len(requests) == 1
        assert requests[0]["keep_alive"] == 0
        assert requests[0]["stream"] is False
        assert requests[0]["think"] is False
        assert "seg_00001" in Path(result["summary_path"]).read_text(encoding="utf-8")
        actual, _ = sf.read(result["tracks"]["SPEAKER_00"], dtype="float32")
        np.testing.assert_allclose(actual, first, atol=1e-7)
        with zipfile.ZipFile(result["archive_path"]) as archive:
            assert "summary.md" in archive.namelist()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_worker_failure_retains_error_directory(synthetic_run, monkeypatch):
    source, config, *_ = synthetic_run
    monkeypatch.setattr(pipeline, "normalize_audio", _in_memory_decode)
    text = config.read_text(encoding="utf-8").replace("fake_backend_plugin:Diarizer", "nonexistent_test_plugin:Diarizer")
    config.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="运行目录及日志"):
        pipeline.run_pipeline(str(source), str(config), summarize=False)
    errors = list((source.parent / "runs").glob("*/error.json"))
    assert len(errors) == 1
    assert "nonexistent_test_plugin" in errors[0].read_text(encoding="utf-8")
