import json
import shutil

import pytest

from duplex_helpers import make_clip, write_manifest


def test_freeze_is_portable_and_detects_audio_tampering(tmp_path):
    from speaker_lab.duplex.protocol import freeze_protocol, verify_protocol
    root = tmp_path / "data"
    row = make_clip(root)
    manifest = write_manifest(tmp_path / "input.jsonl", [row])
    config = tmp_path / "config.toml"
    config.write_text('[asr]\nlanguage = ""\naligner = ""\n', encoding="utf-8")
    first = tmp_path / "protocol"
    freeze_protocol(manifest, root, config, first)
    moved = tmp_path / "elsewhere"
    shutil.copytree(first, moved)
    assert verify_protocol(moved, root)[0]["reference"] == "明天 at four thirty"
    (root / row["audio_relpath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        verify_protocol(moved, root)


@pytest.mark.parametrize("change", [
    {"audio_relpath": "../outside.wav"}, {"split": "future"},
    {"origin": "humanish"}, {"license_id": ""}, {"duration_seconds": float("nan")},
])
def test_manifest_rejects_unsafe_or_invalid_records(tmp_path, change):
    from speaker_lab.duplex.protocol import load_manifest
    root = tmp_path / "data"
    row = {**make_clip(root), **change}
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path / "input.jsonl", [row]), root)


def test_manifest_rejects_symlink_escape_and_duplicate_ids(tmp_path):
    from speaker_lab.duplex.protocol import load_manifest
    root = tmp_path / "data"
    row = make_clip(root)
    outside = tmp_path / "outside.wav"
    shutil.copy(root / row["audio_relpath"], outside)
    (root / "escape.wav").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        load_manifest(write_manifest(tmp_path / "a.jsonl", [{**row, "audio_relpath": "escape.wav"}]), root)
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(write_manifest(tmp_path / "b.jsonl", [row, row]), root)


def test_manifest_rejects_source_family_crossing_splits(tmp_path):
    from speaker_lab.duplex.protocol import load_manifest
    root = tmp_path / "data"
    a = make_clip(root, "first", split="train")
    b = make_clip(root, "second", split="test", seconds=1.3)
    b["source_group_id"] = a["source_group_id"]
    with pytest.raises(ValueError, match="split"):
        load_manifest(write_manifest(tmp_path / "input.jsonl", [a, b]), root)


def test_synthetic_generation_condition_is_not_human_verification(tmp_path):
    from speaker_lab.duplex.protocol import load_manifest
    root = tmp_path / "data"
    row = make_clip(root, origin="synthetic")
    row.update(script_id="s1", voice_id="v1", style_requested="hurried",
               generator_model="test-generator", generator_revision="v1")
    loaded = load_manifest(write_manifest(tmp_path / "input.jsonl", [row]), root)
    assert loaded[0]["style_verified"] is False
    assert loaded[0]["origin"] == "synthetic"


def test_freeze_refuses_overwrite_and_manifest_modification(tmp_path):
    from speaker_lab.duplex.protocol import freeze_protocol, verify_protocol
    root = tmp_path / "data"
    manifest = write_manifest(tmp_path / "input.jsonl", [make_clip(root)])
    config = tmp_path / "config.toml"
    config.write_text('[asr]\nlanguage=""\n', encoding="utf-8")
    out = tmp_path / "protocol"
    freeze_protocol(manifest, root, config, out)
    with pytest.raises(FileExistsError):
        freeze_protocol(manifest, root, config, out)
    (out / "manifest.jsonl").write_text(json.dumps({"fake": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_protocol(out, root)
