"""Preparing the local extension must preserve the installed upstream source."""

import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize("variant", ["cached", "prefill", "calendar"])
def test_backend_patch_is_verified_idempotent_and_preserves_existing_source(tmp_path, variant):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_personaplex_native.py"
    spec = importlib.util.spec_from_file_location("native_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = "VALUE = 1\n"
    after = "VALUE = 2\n"
    source = tmp_path / "server.py"
    source.write_text(before)
    output = tmp_path / "server_native.py"
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / {"cached":"server-native.patch","prefill":"server-prefill.patch","calendar":"server-calendar.patch"}[variant]).write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="server.py",
                tofile="server_native.py",
            )
        )
    )
    (folder / {"cached":"BACKEND.json","prefill":"PREFILL_BACKEND.json","calendar":"CALENDAR_BACKEND.json"}[variant]).write_text(
        json.dumps(
            {
                "original_server_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "candidate_server_sha256": hashlib.sha256(after.encode()).hexdigest(),
            }
        )
    )
    module.prepare(source, output, folder, variant=variant)
    assert source.read_text() == before and output.read_text() == after
    module.prepare(source, output, folder, variant=variant)
    output.write_text("unrelated existing candidate\n")
    with pytest.raises(FileExistsError):
        module.prepare(source, output, folder, variant=variant)
    assert output.read_text() == "unrelated existing candidate\n"
    source.write_text("different upstream revision\n")
    with pytest.raises(ValueError, match="source hash"):
        module.prepare(source, tmp_path / "new.py", folder, variant=variant)
