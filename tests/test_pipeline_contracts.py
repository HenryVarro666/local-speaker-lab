"""Boundary contracts between real adapters and the model-independent pipeline."""

from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from speaker_lab.backends.pyannote import assign_speakers
from speaker_lab.backends.qwen import aligned_words
from speaker_lab import pipeline
from speaker_lab.config import load_config
from speaker_lab.pipeline import _blend_jobs, _validate_segments
from speaker_lab.timeline import activity_intervals, SPEAKERS


ROOT = Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.job = {"id": "asr_0", "speaker": "SPEAKER_00", "start": 10.0, "end": 12.0}
        self.segment = {
            "job_id": "asr_0", "speaker": "SPEAKER_00", "start": 10.2, "end": 11.8,
            "text": "测试", "words": [{"text": "测试", "start": 10.2, "end": 11.8}],
        }
        self.overlap = {
            "id": "overlap_0", "start": 0, "end": 16000,
            "input_start": 0, "input_end": 16000, "region_start": 0, "region_end": 16000,
        }

    def test_real_qwen_timestamp_schema_is_accepted(self):
        result = SimpleNamespace(time_stamps=SimpleNamespace(items=[
            SimpleNamespace(text="测试", start_time=0.2, end_time=1.8)
        ]))
        segment = {**self.segment, "words": aligned_words(result, 10, 2)}
        validated = _validate_segments({"segments": [segment]}, [self.job], 20, [])
        self.assertEqual(validated[0]["words"], segment["words"])
        self.assertEqual(validated[0]["id"], "seg_00001")
        self.assertEqual(validated[0]["source_audio"], "SPEAKER_00.wav")

    def test_real_matcher_order_maps_source_paths_to_canonical_speakers(self):
        matched = assign_speakers([[0.1, 0.95], [0.95, 0.1]], list(SPEAKERS))
        tracks = {speaker: np.zeros(16000, dtype=np.float32) for speaker in SPEAKERS}
        separated = {"jobs": [{"id": "overlap_0", "sources": ["channel0.wav", "channel1.wav"]}]}
        with patch("speaker_lab.pipeline.read_source", side_effect=lambda path, count: np.full(count, 0.2 if "0" in path else 0.8)):
            _blend_jobs(np.zeros(16000), tracks, [self.overlap], separated,
                        {"jobs": [{"id": "overlap_0", **matched}]})
        np.testing.assert_allclose(tracks["SPEAKER_00"], 0.8)
        np.testing.assert_allclose(tracks["SPEAKER_01"], 0.2)

    def test_chronological_canonicalization_preserves_actual_overlap(self):
        intervals, mapping = activity_intervals([
            {"start": 0, "end": 3, "speaker": "Z"},
            {"start": 1, "end": 4, "speaker": "A"},
        ], sample_count=64000)
        self.assertEqual(mapping, {"Z": "SPEAKER_00", "A": "SPEAKER_01"})
        self.assertEqual(intervals[1], {"start": 16000, "end": 48000, "speakers": SPEAKERS})

    def test_rejects_nonfinite_and_out_of_range_word_times(self):
        for start, end in [(float("nan"), 11.8), (10.2, float("inf")), (9.0, 11.8), (10.2, 13.0), (11.5, 10.3)]:
            with self.subTest(start=start, end=end):
                segment = {**self.segment, "words": [{"text": "测试", "start": start, "end": end}]}
                self.assertRaises(ValueError, _validate_segments, {"segments": [segment]}, [self.job], 20, [])

    def test_rejects_invalid_word_container(self):
        for value in (None, "word", {}, ["word"]):
            with self.subTest(value=value):
                segment = {**self.segment, "words": value}
                self.assertRaises(ValueError, _validate_segments, {"segments": [segment]}, [self.job], 20, [])

    def test_rejects_duplicate_separation_result_ids(self):
        duplicate = {"id": "overlap_0", "sources": ["a.wav", "b.wav"]}
        tracks = {speaker: np.zeros(16000, dtype=np.float32) for speaker in SPEAKERS}
        with patch("speaker_lab.pipeline.read_source", return_value=np.zeros(16000)):
            self.assertRaises(
                ValueError, _blend_jobs, np.zeros(16000), tracks, [self.overlap],
                {"jobs": [duplicate, duplicate]},
                {"jobs": [{"id": "overlap_0", "speakers": list(SPEAKERS), "margin": 1, "uncertain": False}]},
            )

    def test_rejects_duplicate_matching_result_ids(self):
        duplicate = {"id": "overlap_0", "speakers": list(SPEAKERS), "margin": 1, "uncertain": False}
        tracks = {speaker: np.zeros(16000, dtype=np.float32) for speaker in SPEAKERS}
        with patch("speaker_lab.pipeline.read_source", return_value=np.zeros(16000)):
            self.assertRaises(
                ValueError, _blend_jobs, np.zeros(16000), tracks, [self.overlap],
                {"jobs": [{"id": "overlap_0", "sources": ["a.wav", "b.wav"]}]},
                {"jobs": [duplicate, duplicate]},
            )

    def test_rejects_unknown_separation_id(self):
        tracks = {speaker: np.zeros(16000, dtype=np.float32) for speaker in SPEAKERS}
        with patch("speaker_lab.pipeline.read_source", return_value=np.zeros(16000)):
            self.assertRaises(
                ValueError, _blend_jobs, np.zeros(16000), tracks, [self.overlap],
                {"jobs": [{"id": "unexpected", "sources": ["a.wav", "b.wav"]}]},
                {"jobs": [{"id": "overlap_0", "speakers": list(SPEAKERS), "margin": 1, "uncertain": False}]},
            )


@pytest.mark.parametrize("result", [
    {},
    {"jobs": None},
    {"jobs": {}},
    {"jobs": ["invalid job"]},
    {"jobs": [{"id": 1, "sources": ["a.wav", "b.wav"]}]},
    {"jobs": []},
    {"jobs": [{"id": "other", "sources": ["a.wav", "b.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": ["a.wav", "b.wav"]}] * 2},
    {"jobs": [{"id": "overlap_00000", "sources": "two paths"}]},
    {"jobs": [{"id": "overlap_00000", "sources": ["a.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": [None, "b.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": ["a.wav", None]}]},
    {"jobs": [{"id": "overlap_00000", "sources": [42, "b.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": [{}, "b.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": ["", "b.wav"]}]},
    {"jobs": [{"id": "overlap_00000", "sources": ["a.wav", " \t"]}]},
])
def test_invalid_separation_stops_pipeline_before_matcher_starts(tmp_path, monkeypatch, result):
    """Exercise the real orchestration, rather than only its validation helper."""
    config = load_config(ROOT / "configs/default.toml")
    config["pipeline"]["output_dir"] = str(tmp_path / "runs")
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "normalize_audio", lambda *_: np.full(6 * 16000, 0.1, dtype=np.float32))
    stages = []

    def run(_runner, stage, _payload):
        stages.append(stage)
        if stage == "diarize":
            return {"turns": [{"start": 0, "end": 4, "speaker": "first"},
                              {"start": 2, "end": 6, "speaker": "second"}]}
        if stage == "separate":
            return result
        pytest.fail(f"Invalid separation must stop before {stage} starts")

    monkeypatch.setattr(pipeline.WorkerRunner, "run", run)
    with pytest.raises(RuntimeError, match="Separation/matching|Every overlap job"):
        pipeline.run_pipeline(str(tmp_path / "input.wav"), summarize=False)
    assert stages == ["diarize", "separate"]
    assert len(list((tmp_path / "runs").glob("*/error.json"))) == 1


@pytest.mark.parametrize("result", [
    {"jobs": []},
    {"jobs": [{"id": "other", "speakers": list(SPEAKERS)}]},
    {"jobs": [{"id": "overlap_00000", "speakers": list(SPEAKERS)}] * 2},
    {"jobs": [{"id": "overlap_00000", "speakers": None}]},
    {"jobs": [{"id": "overlap_00000", "speakers": ["SPEAKER_00"]}]},
    {"jobs": [{"id": "overlap_00000", "speakers": ["SPEAKER_00", "SPEAKER_00"]}]},
    {"jobs": [{"id": "overlap_00000", "speakers": ["SPEAKER_00", "unknown"]}]},
    {"jobs": [{"id": "overlap_00000", "speakers": ["SPEAKER_00", {}]}]},
    {"jobs": [{"id": "overlap_00000", "speakers": list(SPEAKERS), "uncertain": "false"}]},
    {"jobs": [{"id": "overlap_00000", "speakers": list(SPEAKERS), "margin": -1}]},
    {"jobs": [{"id": "overlap_00000", "speakers": list(SPEAKERS), "margin": "0.1"}]},
    {"jobs": [{"id": "overlap_00000", "speakers": list(SPEAKERS), "margin": True}]},
])
def test_invalid_matching_stops_before_audio_blending_or_transcription(tmp_path, monkeypatch, result):
    config = load_config(ROOT / "configs/default.toml")
    config["pipeline"]["output_dir"] = str(tmp_path / "runs")
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "normalize_audio", lambda *_: np.full(6 * 16000, 0.1, dtype=np.float32))
    stages = []

    def run(_runner, stage, payload):
        stages.append(stage)
        if stage == "diarize":
            return {"turns": [{"start": 0, "end": 4, "speaker": "first"},
                              {"start": 2, "end": 6, "speaker": "second"}]}
        if stage == "separate":
            return {"jobs": [{"id": job["id"], "sources": [
                pipeline.write_wav(tmp_path / f"{job['id']}-{index}.wav",
                                   np.full(job["input_end"] - job["input_start"], 0.1, dtype=np.float32))
                for index in range(2)
            ]} for job in payload["jobs"]]}
        if stage == "match":
            return result
        pytest.fail(f"Invalid matching must stop before {stage} starts")

    monkeypatch.setattr(pipeline.WorkerRunner, "run", run)
    with patch("speaker_lab.pipeline._blend_jobs") as blend:
        with pytest.raises(RuntimeError, match="Separation/matching|Every overlap job|Matching"):
            pipeline.run_pipeline(str(tmp_path / "input.wav"), summarize=False)
        blend.assert_not_called()
    assert stages == ["diarize", "separate", "match"]
    assert len(list((tmp_path / "runs").glob("*/error.json"))) == 1


@pytest.mark.parametrize("kind", ["relative", "missing", "same", "wrong_rate", "stereo", "wrong_duration", "nonfinite"])
def test_invalid_separated_audio_stops_before_matcher(tmp_path, monkeypatch, kind):
    """A plugin's WAVs are checked before a second model consumes them."""
    import soundfile as sf

    config = load_config(ROOT / "configs/default.toml")
    config["pipeline"]["output_dir"] = str(tmp_path / "runs")
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "normalize_audio", lambda *_: np.full(6 * 16000, 0.1, dtype=np.float32))
    stages = []

    def run(_runner, stage, payload):
        stages.append(stage)
        if stage == "diarize":
            return {"turns": [{"start": 0, "end": 4, "speaker": "first"},
                              {"start": 2, "end": 6, "speaker": "second"}]}
        if stage == "separate":
            job = payload["jobs"][0]
            count = job["input_end"] - job["input_start"]
            first, second = tmp_path / "a.wav", tmp_path / "b.wav"
            sf.write(first, np.full(count, 0.1), 16000, subtype="FLOAT")
            audio = np.full(count, 0.2)
            rate = 16000
            if kind == "wrong_rate":
                rate = 8000
            elif kind == "stereo":
                audio = np.stack((audio, audio), axis=1)
            elif kind == "wrong_duration":
                audio = audio[:count // 2]
            elif kind == "nonfinite":
                audio[0] = np.nan
            sf.write(second, audio, rate, subtype="FLOAT")
            if kind == "relative":
                second = Path("relative.wav")
            elif kind == "missing":
                second = tmp_path / "missing.wav"
            elif kind == "same":
                second = first
            return {"jobs": [{"id": job["id"], "sources": [str(first), str(second)]}]}
        pytest.fail(f"Invalid separated audio must stop before {stage}")

    monkeypatch.setattr(pipeline.WorkerRunner, "run", run)
    with pytest.raises(RuntimeError, match="source|Backend"):
        pipeline.run_pipeline(str(tmp_path / "input.wav"), summarize=False)
    assert stages == ["diarize", "separate"]
    assert len(list((tmp_path / "runs").glob("*/error.json"))) == 1


@pytest.mark.parametrize("start,end", [(-0.09, -0.05), (1.05, 1.09)])
def test_asr_tolerance_does_not_export_inverted_segments(start, end):
    jobs = [{"id": "a", "speaker": "SPEAKER_00", "start": 0, "end": 1}]
    segment = {"job_id": "a", "speaker": "SPEAKER_00", "start": start, "end": end, "text": "x"}
    with pytest.raises(ValueError, match="outside"):
        _validate_segments({"segments": [segment]}, jobs, 1, [])


def test_asr_rounding_is_clipped_consistently_for_segments_and_words():
    jobs = [{"id": "a", "speaker": "SPEAKER_00", "start": 1.0, "end": 2.0}]
    segment = {"job_id": "a", "speaker": "SPEAKER_00", "start": 0.95, "end": 2.05, "text": "x",
               "words": [{"text": "x", "start": 0.95, "end": 2.05}]}
    result = _validate_segments({"segments": [segment]}, jobs, 3, [])[0]
    assert (result["start"], result["end"]) == (1.0, 2.0)
    assert result["words"] == [{"text": "x", "start": 1.0, "end": 2.0}]


def test_asr_rejects_word_entirely_outside_normalized_segment():
    jobs = [{"id": "a", "speaker": "SPEAKER_00", "start": 0, "end": 1}]
    segment = {"job_id": "a", "speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "x",
               "words": [{"text": "x", "start": -0.09, "end": -0.05}]}
    with pytest.raises(ValueError, match="outside"):
        _validate_segments({"segments": [segment]}, jobs, 1, [])


if __name__ == "__main__":
    unittest.main()
