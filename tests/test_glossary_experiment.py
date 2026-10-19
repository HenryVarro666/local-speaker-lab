"""Evaluation integrity checks, requiring no model or GPU."""

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from speaker_lab.glossary_experiment import load_manifest, load_terms, make_contexts, run, summarize


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def row(clip="one", reference="检查 cache"):
    return {"id": clip, "audio": "one.wav", "reference": reference, "group": "mixed", "split": "test"}


def test_manifest_resolves_audio_and_rejects_duplicate_ids(tmp_path):
    sf.write(tmp_path / "one.wav", np.zeros(16000), 16000)
    data = load_manifest(write_manifest(tmp_path, [row()]), check_audio=True)
    assert data[0]["duration_seconds"] == 1
    assert Path(data[0]["audio_path"]).is_absolute()
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(write_manifest(tmp_path, [row(), row()]))


def test_wrong_sample_rate_is_rejected_before_inference(tmp_path):
    sf.write(tmp_path / "one.wav", np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="16kHz"):
        load_manifest(write_manifest(tmp_path, [row()]), check_audio=True)


def test_glossary_input_rejects_normalized_duplicates(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text('["API", "ａｐｉ"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_terms(path)


def test_contexts_are_fixed_plain_lists_without_reference_text():
    assert make_contexts(["cache", "API"], ["cairn", "ATP"]) == {
        "none": "", "personal": "cache\nAPI", "unrelated": "cairn\nATP"}
    with pytest.raises(ValueError, match="equal"):
        make_contexts(["API"], ["ATP", "cache"])
    with pytest.raises(ValueError, match="overlap"):
        make_contexts(["API"], ["api"])


def test_missing_or_duplicate_paired_prediction_fails():
    predictions = [{"id": "one", "condition": "none", "text": "检查 cache"}]
    with pytest.raises(ValueError, match="Incomplete"):
        summarize([row()], predictions, ["cache"], ["cairn"])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row()], predictions * 2, ["cache"], ["cairn"])


def test_primary_recall_uses_personal_terms_and_union_tracks_decoy_invention():
    predictions = [{"id": "one", "condition": arm, "text": text, "seconds": .5,
                    "duration_seconds": 1} for arm, text in (
        ("none", "检查 cash"), ("personal", "检查 cache"), ("unrelated", "检查 cairn"))]
    result = summarize([row()], predictions, ["cache"], ["cairn"])["slices"]
    assert result["all"]["personal"]["term_recall"] == 1
    assert result["all"]["none"]["term_recall"] == 0
    assert result["all"]["unrelated"]["term_hyp_mentions"] == 0
    assert result["all"]["unrelated"]["all_candidates"]["term_false_mentions"] == 1
    assert result["all"]["personal"]["rtf"] == .5
    assert set(result) == {"all", "group:mixed", "split:test"}


def test_runner_loads_once_keeps_references_out_of_context_and_records_complete_pairs(tmp_path, monkeypatch):
    sf.write(tmp_path / "one.wav", np.zeros(16000), 16000)
    manifest = write_manifest(tmp_path, [row(reference="绝不能进入模型的参考答案 cache")])
    personal, unrelated = tmp_path / "personal.json", tmp_path / "unrelated.json"
    personal.write_text('["cache"]', encoding="utf-8")
    unrelated.write_text('["cairn"]', encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(f'[asr]\nmodel = "{model_dir.as_posix()}"\ndevice = "cpu"\ndtype = "float32"\nlanguage = ""\n', encoding="utf-8")
    calls = {"load": [], "infer": []}

    class FakeQwen:
        def __init__(self):
            generation = SimpleNamespace(to_dict=lambda: {"do_sample": False, "num_beams": 1})
            self.model = SimpleNamespace(eval=lambda: None, generation_config=generation)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["load"].append((args, kwargs))
            return cls()

        def transcribe(self, **kwargs):
            calls["infer"].append(kwargs)
            return [SimpleNamespace(text="检查 cache", language="Chinese")]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, device=lambda value: SimpleNamespace(type="cpu"),
        float32="fp32", inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=FakeQwen))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: None))
    monkeypatch.setattr("speaker_lab.glossary_experiment.importlib.metadata.version", lambda name: "test")
    output = tmp_path / "output"
    run(SimpleNamespace(config=str(config), manifest=str(manifest), personal=str(personal),
                        unrelated=str(unrelated), output_dir=str(output), seed=1, offline=True))
    assert len(calls["load"]) == 1
    assert "forced_aligner" not in calls["load"][0][1]
    assert len(calls["infer"]) == 4  # one warm-up, then all three arms
    assert sorted(call["context"] for call in calls["infer"][1:]) == ["", "cache", "cairn"]
    assert all(call["language"] is None and call["return_time_stamps"] is False for call in calls["infer"])
    assert all("reference" not in call for call in calls["infer"])
    provenance = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "complete"
    assert len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert (output / "scores.json").is_file()
