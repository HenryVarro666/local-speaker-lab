import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_ascend_glossary.py"
SPEC = importlib.util.spec_from_file_location("prepare_ascend_glossary", SCRIPT)
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)
COMMIT = "a" * 40


def make_row(clip_id="00000", *, language="mixed", reference="检查这个 API", sample_rate=16000, channels=1):
    audio = io.BytesIO()
    samples = np.zeros((sample_rate // 10, channels), dtype=np.float32)
    samples[:, 0] = np.sin(np.arange(len(samples)) * 0.1) * 0.1
    sf.write(audio, samples, sample_rate, format="WAV", subtype="PCM_16")
    return {"id": clip_id, "transcription": reference, "language": language,
            "audio": {"bytes": audio.getvalue(), "path": "original.wav"}}


def export(rows, output_dir, **kwargs):
    return importer.export_rows(
        rows, output_dir, split="test", commit=COMMIT, requested_revision="main",
        source_filename=importer.SPLIT_FILES["test"], official_count=len(rows), **kwargs,
    )


def test_download_resolves_revision_and_requests_only_selected_official_file(tmp_path):
    requests = []
    info = SimpleNamespace(sha=COMMIT, siblings=[
        SimpleNamespace(rfilename=name) for name in importer.SPLIT_FILES.values()
    ])
    api = SimpleNamespace(dataset_info=lambda **kw: requests.append(("info", kw)) or info)

    def download(**kwargs):
        requests.append(("download", kwargs))
        return str(tmp_path / "split.parquet")

    path, commit, filename = importer.resolve_split("validation", "main", api=api, downloader=download)
    assert commit == COMMIT
    assert path == tmp_path / "split.parquet"
    assert requests == [
        ("info", {"repo_id": "CAiRE/ASCEND", "revision": "main"}),
        ("download", {"repo_id": "CAiRE/ASCEND", "repo_type": "dataset",
                      "revision": COMMIT, "filename": "main/validation-00000-of-00001.parquet"}),
    ]
    assert filename == "main/validation-00000-of-00001.parquet"


def test_missing_split_is_rejected_before_download():
    api = SimpleNamespace(dataset_info=lambda **kw: SimpleNamespace(sha=COMMIT, siblings=[]))

    def unexpected_download(**kwargs):
        pytest.fail("Missing official split must not start any download")

    with pytest.raises(ValueError, match="split file missing"):
        importer.resolve_split("test", "main", api=api, downloader=unexpected_download)


def test_export_writes_portable_manifest_mono_16k_audio_and_source_metadata(tmp_path):
    output = tmp_path / "test-data"
    rows = [make_row(sample_rate=8000, channels=2), make_row("00001", language="en", reference="cache")]
    metadata = export(rows, output)
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest == [
        {"id": "00000", "audio": "audio/00000.wav", "reference": "检查这个 API", "split": "test", "group": "mixed"},
        {"id": "00001", "audio": "audio/00001.wav", "reference": "cache", "split": "test", "group": "english"},
    ]
    audio = sf.info(output / manifest[0]["audio"])
    assert (audio.samplerate, audio.channels, audio.frames, audio.subtype) == (16000, 1, 1600, "PCM_16")
    assert metadata == json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert metadata["count"] == metadata["official_split_count"] == 2
    assert metadata["subset"] is False
    assert metadata["source"]["revision"] == COMMIT
    assert metadata["source"]["license"] == "CC-BY-SA-4.0"


def test_limit_labels_subset_and_group_comes_only_from_metadata(tmp_path):
    rows = [make_row(language="zh", reference="这个 API"), make_row("00001")]
    metadata = export(rows, tmp_path / "subset", limit=1)
    assert metadata["subset"] is True
    assert metadata["count"] == metadata["limit"] == 1
    assert metadata["official_split_count"] == 2
    assert metadata["selection"] == "first_n_in_official_order"
    assert metadata["groups"] == {"mandarin": 1}
    manifest = json.loads((tmp_path / "subset" / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["group"] == "mandarin"


@pytest.mark.parametrize("bad_row", [
    {"id": "incomplete"},
    {**make_row(), "id": "../../outside"},
    {**make_row(), "transcription": " "},
    {**make_row(), "language": None},
    {**make_row(), "audio": {"path": "untrusted_external.wav"}},
])
def test_invalid_row_leaves_no_partial_output_or_files(tmp_path, bad_row):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        export([make_row("valid"), bad_row], output)
    assert list(tmp_path.iterdir()) == []


def test_duplicate_ids_do_not_overwrite_audio(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        export([make_row(), make_row()], tmp_path / "duplicate")
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export([make_row()], output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_help_runs_without_site_packages_or_optional_dependencies():
    result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--revision" in result.stdout and "--limit" in result.stdout
