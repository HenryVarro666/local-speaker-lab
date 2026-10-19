"""Synthetic metadata-only freezing checks; no ASCEND data or models needed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import wave

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_ascend_remaining.py"
SPEC = importlib.util.spec_from_file_location("freeze_ascend_remaining", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def write_audio(path, frames, value=1, rate=16000, channels=1, width=2):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(value.to_bytes(width, "little") * frames * channels)


def inputs(tmp_path, specs=None, prior=("old",)):
    source = tmp_path / "source"
    source.mkdir()
    specs = specs or [("old", "mixed", 16000), ("new-a", "mixed", 32000),
                      ("new-b", "mandarin", 192000), ("new-c", "english", 16000),
                      ("short", "english", 15999), ("long", "mandarin", 192001)]
    rows = []
    for index, (clip_id, group, frames) in enumerate(specs):
        filename = f"{index}.wav"
        write_audio(source / filename, frames, value=index + 1)
        rows.append({"id": clip_id, "group": group, "split": "validation",
                     "audio": filename, "reference": f"synthetic reference {index}"})
    manifest = source / "manifest.jsonl"
    write_manifest(manifest, rows)
    protocol = tmp_path / "prior"
    protocol.mkdir()
    (protocol / "selected_ids.json").write_text(json.dumps(list(prior)), encoding="utf-8")
    return manifest, protocol, rows


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_excludes_old_and_uses_inclusive_audio_header_bounds(tmp_path):
    manifest, protocol, rows = inputs(tmp_path)
    # Stored duration metadata is irrelevant; actual WAV headers decide.
    for row in rows:
        row["duration_seconds"] = 100
    write_manifest(manifest, rows)
    output = tmp_path / "frozen"
    audit = selector.freeze(manifest, protocol, output)
    assert audit["source_count"] == 6
    assert audit["eligible_count"] == 4
    assert audit["excluded_duration_count"] == 2
    assert audit["prior_count"] == 1
    assert audit["count"] == 3
    assert audit["groups"] == {"mixed": 1, "mandarin": 1, "english": 1}
    assert audit["total_audio_seconds"] == 15
    assert audit["audio_hash_audit"]["checked"] == 4
    assert audit["reference_not_used_for_selection"] is True
    assert audit["runtime_manifest_portable"] is False
    assert json.loads((output / "selected_ids.json").read_text()) == ["new-a", "new-b", "new-c"]
    new_rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert all(Path(row["audio"]).is_absolute() and Path(row["audio"]).exists() for row in new_rows)
    assert new_rows[0]["reference"] == "synthetic reference 1"
    for filename, key in (("manifest.jsonl", "manifest_sha256"),
                          ("selected_ids.json", "selected_ids_sha256")):
        data = (output / filename).read_bytes()
        assert b"\r\n" not in data
        assert hashlib.sha256(data).hexdigest() == audit[key]


def test_order_and_membership_do_not_depend_on_reference_or_source_order(tmp_path):
    specs = [("old", "mixed", 16000)] + [(f"new-{i}", "mixed", 16000) for i in range(6)]
    manifest, protocol, rows = inputs(tmp_path, specs)
    first = tmp_path / "first"
    selector.freeze(manifest, protocol, first)
    for index, row in enumerate(rows):
        row["reference"] = "arbitrary unrelated changed text " * index
    write_manifest(manifest, list(reversed(rows)))
    second = tmp_path / "second"
    selector.freeze(manifest, protocol, second)
    assert (first / "selected_ids.json").read_bytes() == (second / "selected_ids.json").read_bytes()
    expected = sorted([f"new-{i}" for i in range(6)],
                      key=lambda clip_id: (hashlib.sha256(f"20261018:{clip_id}".encode()).hexdigest(), clip_id))
    assert json.loads((first / "selected_ids.json").read_text()) == expected


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(dict(rows[0])), "Duplicate clip ID"),
    (lambda rows: rows[0].update(split="test"), "Only validation"),
    (lambda rows: rows[0].update(group="unknown"), "Unknown language group"),
    (lambda rows: rows[0].update(audio="../outside.wav"), "Unsafe audio path"),
    (lambda rows: rows[0].update(audio="C:\\outside.wav"), "Unsafe audio path"),
    (lambda rows: rows[0].update(audio="\\\\server\\share\\outside.wav"), "Unsafe audio path"),
    (lambda rows: rows[0].update(audio="missing.wav"), "missing.wav"),
])
def test_invalid_source_is_rejected_before_creating_output(tmp_path, mutation, match):
    manifest, protocol, rows = inputs(tmp_path)
    mutation(rows)
    write_manifest(manifest, rows)
    output = tmp_path / "not-created"
    with pytest.raises((ValueError, OSError), match=match):
        selector.freeze(manifest, protocol, output)
    assert not output.exists()


@pytest.mark.parametrize("prior,match", [
    (["old", "old"], "Duplicate IDs"),
    (["not-found"], "missing or duration-ineligible"),
    (["short"], "missing or duration-ineligible"),
    (["long"], "missing or duration-ineligible"),
    ([1], "nonempty strings"),
    ({"old": True}, "list of nonempty strings"),
])
def test_invalid_prior_is_rejected_without_output(tmp_path, prior, match):
    manifest, protocol, _ = inputs(tmp_path)
    (protocol / "selected_ids.json").write_text(json.dumps(prior), encoding="utf-8")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=match):
        selector.freeze(manifest, protocol, output)
    assert not output.exists()


@pytest.mark.parametrize("target_index", [1, 2])
def test_duplicate_audio_rejected_within_and_across_groups_including_prior(tmp_path, target_index):
    manifest, protocol, rows = inputs(tmp_path)
    shutil.copyfile(manifest.parent / rows[0]["audio"], manifest.parent / rows[target_index]["audio"])
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Duplicate eligible audio"):
        selector.freeze(manifest, protocol, output)
    assert not output.exists()


def test_duplicate_duration_ineligible_audio_does_not_filter_other_clips(tmp_path):
    manifest, protocol, rows = inputs(tmp_path)
    shutil.copyfile(manifest.parent / rows[-1]["audio"], manifest.parent / rows[-2]["audio"])
    audit = selector.freeze(manifest, protocol, tmp_path / "frozen")
    assert audit["count"] == 3
    assert audit["audio_hash_audit"]["checked"] == 4


@pytest.mark.parametrize("audio_options", [
    {"rate": 8000}, {"channels": 2}, {"width": 1},
])
def test_audio_format_rejected_before_writing(tmp_path, audio_options):
    manifest, protocol, rows = inputs(tmp_path)
    write_audio(manifest.parent / rows[1]["audio"], 32000, **audio_options)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="16kHz mono PCM_16"):
        selector.freeze(manifest, protocol, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    manifest, protocol, _ = inputs(tmp_path)
    output = tmp_path / "frozen"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        selector.freeze(manifest, protocol, output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_symlink_audio_is_rejected(tmp_path):
    manifest, protocol, rows = inputs(tmp_path)
    link = manifest.parent / "linked.wav"
    try:
        link.symlink_to(manifest.parent / rows[0]["audio"])
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this host")
    rows[1]["audio"] = link.name
    write_manifest(manifest, rows)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="symlink or reparse"):
        selector.freeze(manifest, protocol, output)
    assert not output.exists()


def test_cli_reports_metadata_only_and_stops_on_failure(tmp_path, capsys):
    manifest, protocol, _ = inputs(tmp_path)
    output = tmp_path / "frozen"
    argv = ["--source-manifest", str(manifest), "--prior-protocol", str(protocol),
            "--output-dir", str(output)]
    assert selector.main(argv) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed)["count"] == 3
    assert "synthetic reference" not in printed
    with pytest.raises(SystemExit) as failure:
        selector.main(argv)
    assert failure.value.code == 1
    assert "already exists" in capsys.readouterr().err
