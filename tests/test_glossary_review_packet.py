"""Listening worksheets must never silently turn a prediction into auditory evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_glossary_review_packet.py"
SPEC = importlib.util.spec_from_file_location("make_glossary_review_packet", SCRIPT)
packet_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet_tool)


def inputs(tmp_path):
    rows, predictions = [], []
    for clip_id in packet_tool.EVENT_IDS:
        audio = tmp_path / f"{clip_id}.wav"
        audio.write_bytes(b"local fixture audio bytes")
        rows.append({"id": clip_id, "audio": audio.name, "reference": "reference " + clip_id})
        predictions.extend({"id": clip_id, "condition": arm, "text": "hypothesis " + arm}
                           for arm in packet_tool.CONDITIONS)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in predictions), encoding="utf-8")
    provenance = {"status": "complete", "manifest_sha256": packet_tool.file_hash(manifest),
                  "config": {"model": "frozen-local-model"}, "seed": 20261018}
    (run_dir / "run.json").write_text(json.dumps(provenance), encoding="utf-8")
    return manifest, run_dir


def rewrite_lines(path, transform):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = transform(rows)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_packet_keeps_audio_in_place_and_review_pending(tmp_path):
    manifest, run_dir = inputs(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "worksheet"
    page = packet_tool.build_packet(manifest, run_dir, output)
    template = json.loads((output / "review_template.json").read_text(encoding="utf-8"))
    assert template["status"] == "pending_human_listening"
    assert template["reviewer"] is None
    assert {item["id"] for item in template["events"]} == set(packet_tool.EVENT_IDS)
    assert all(not item["listened"] and item["heard_text"] is None
               and item["confidence"] == "not_reviewed" for item in template["events"])
    assert "CC BY-SA 4.0" in template["license"]
    assert template["source"]["manifest_sha256"] == packet_tool.file_hash(manifest)
    html = page.read_text(encoding="utf-8")
    assert html.count("<audio controls") == 6
    assert "<details>" in html and "<details open" not in html
    assert 'link.download = "review.json"' in html
    assert "automated agent cannot listen" in html
    assert all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha
               for p, sha in before.items())
    assert sorted(p.name for p in output.iterdir()) == ["index.html", "review_template.json"]
    with pytest.raises(FileExistsError, match="overwrite"):
        packet_tool.build_packet(manifest, run_dir, output)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "invalid"])
def test_invalid_paired_predictions_fail_before_creating_output(tmp_path, mode):
    manifest, run_dir = inputs(tmp_path)
    path = run_dir / "predictions.jsonl"
    def transform(rows):
        if mode == "missing":
            return rows[1:]
        if mode == "duplicate":
            return rows + [rows[0]]
        rows[0]["condition"] = "unknown"
        return rows
    rewrite_lines(path, transform)
    output = tmp_path / "worksheet"
    with pytest.raises(ValueError, match="diagnostic prediction"):
        packet_tool.build_packet(manifest, run_dir, output)
    assert not output.exists()


def test_missing_local_audio_fails_without_copy_or_inference(tmp_path):
    manifest, run_dir = inputs(tmp_path)
    (tmp_path / f"{packet_tool.EVENT_IDS[0]}.wav").unlink()
    with pytest.raises(ValueError, match="existing local file"):
        packet_tool.build_packet(manifest, run_dir, tmp_path / "worksheet")
    assert not (tmp_path / "worksheet").exists()


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_duplicate_manifest_event_is_rejected(tmp_path, mode):
    manifest, run_dir = inputs(tmp_path)
    rewrite_lines(manifest, lambda rows: rows[1:] if mode == "missing" else rows + [rows[0]])
    run_file = run_dir / "run.json"
    provenance = json.loads(run_file.read_text(encoding="utf-8"))
    provenance["manifest_sha256"] = packet_tool.file_hash(manifest)
    run_file.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing diagnostic|Duplicate manifest"):
        packet_tool.build_packet(manifest, run_dir, tmp_path / "worksheet")
    assert not (tmp_path / "worksheet").exists()


def test_invalid_source_audio_hash_shape_has_clear_failure(tmp_path):
    manifest, run_dir = inputs(tmp_path)
    run_file = run_dir / "run.json"
    provenance = json.loads(run_file.read_text(encoding="utf-8"))
    provenance["audio_sha256"] = []
    run_file.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="audio_sha256 must be a JSON object"):
        packet_tool.build_packet(manifest, run_dir, tmp_path / "worksheet")


@pytest.mark.parametrize("change", ["incomplete", "manifest", "audio"])
def test_source_provenance_mismatch_is_rejected(tmp_path, change):
    manifest, run_dir = inputs(tmp_path)
    run_file = run_dir / "run.json"
    provenance = json.loads(run_file.read_text(encoding="utf-8"))
    if change == "incomplete":
        provenance["status"] = "started"
    elif change == "manifest":
        provenance["manifest_sha256"] = "0" * 64
    else:
        provenance["audio_sha256"] = {packet_tool.EVENT_IDS[0]: "0" * 64}
    run_file.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="status complete|SHA-256"):
        packet_tool.build_packet(manifest, run_dir, tmp_path / "worksheet")
    assert not (tmp_path / "worksheet").exists()


def test_hostile_reference_prediction_and_audio_filename_are_escaped(tmp_path):
    manifest, run_dir = inputs(tmp_path)
    hostile = '</script><script>alert("x")</script>&\u2028'
    rewrite_lines(manifest, lambda rows: [dict(row, reference=hostile) for row in rows])
    rewrite_lines(run_dir / "predictions.jsonl",
                  lambda rows: [dict(row, text=hostile) for row in rows])
    provenance_path = run_dir / "run.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["manifest_sha256"] = packet_tool.file_hash(manifest)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    page = packet_tool.build_packet(manifest, run_dir, tmp_path / "worksheet")
    html = page.read_text(encoding="utf-8")
    assert hostile not in html
    assert "&lt;/script&gt;" in html
    assert "\\u003c/script\\u003e" in html
    data = html.split('<script type="application/json" id="packet-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(data)["events"][0]["reference"] == hostile
    assert html.count("<script>") == 1


def test_cli_reports_failure_and_returns_nonzero(tmp_path, capsys):
    manifest, run_dir = inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    assert packet_tool.main(["--manifest", str(manifest), "--run-dir", str(run_dir),
                             "--output-dir", str(output)]) == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
