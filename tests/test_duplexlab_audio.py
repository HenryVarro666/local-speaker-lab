import hashlib

import numpy as np
import pytest
import soundfile as sf

from duplex_helpers import make_clip


def test_audit_preserves_source_and_produces_finite_mono_pcm(tmp_path):
    from speaker_lab.duplex.audio import audit_clip
    root = tmp_path / "data"
    row = make_clip(root)
    result = audit_clip(row, root, tmp_path / "derived", {})
    assert result["decision"] == "pass"
    samples, rate = sf.read(result["normalized_path"])
    assert rate == 16000 and samples.ndim == 1 and len(samples) == 19200
    assert hashlib.sha256((root / row["audio_relpath"]).read_bytes()).hexdigest() == row["audio_sha256"]


@pytest.mark.parametrize("samples,reason", [
    (np.zeros(16000), "silent"), (np.full(16000, np.nan), "nonfinite"),
    (np.ones(8000) * .1, "duration"),
])
def test_audit_records_rejection_instead_of_losing_invalid_clip(tmp_path, samples, reason):
    from speaker_lab.duplex.audio import audit_clip
    root = tmp_path / "data"
    row = make_clip(root)
    sf.write(root / row["audio_relpath"], samples, 16000, subtype="FLOAT")
    row["audio_sha256"] = hashlib.sha256((root / row["audio_relpath"]).read_bytes()).hexdigest()
    result = audit_clip(row, root, tmp_path / "derived", {})
    assert result["decision"] == "reject" and reason in result["reasons"]
    assert result["clip_id"] == row["clip_id"]


def test_clipping_and_quiet_speech_need_review_without_gain_normalization(tmp_path):
    from speaker_lab.duplex.audio import audit_clip
    root = tmp_path / "data"
    row = make_clip(root)
    sf.write(root / row["audio_relpath"], np.ones(19200), 16000, subtype="FLOAT")
    row["audio_sha256"] = hashlib.sha256((root / row["audio_relpath"]).read_bytes()).hexdigest()
    result = audit_clip(row, root, tmp_path / "derived", {})
    assert result["decision"] == "needs_review" and "clipping" in result["reasons"]
    assert np.max(sf.read(result["normalized_path"])[0]) == 1.0


def test_pcm_dedup_handles_distinct_lossless_containers_and_rejects_cross_split(tmp_path):
    from speaker_lab.duplex.audio import audit_clip, find_exact_duplicates
    root = tmp_path / "data"
    a = make_clip(root, "a")
    b = make_clip(root, "b")
    samples, rate = sf.read(root / a["audio_relpath"], dtype="float32")
    sf.write(root / b["audio_relpath"], samples, rate, format="FLAC", subtype="PCM_16")
    b["audio_sha256"] = hashlib.sha256((root / b["audio_relpath"]).read_bytes()).hexdigest()
    rows = [audit_clip(r, root, tmp_path / "derived", {}) for r in (a, b)]
    assert rows[0]["pcm_sha256"] == rows[1]["pcm_sha256"]
    assert find_exact_duplicates(rows)[0]["duplicate_of"] == "a"
    rows[1]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        find_exact_duplicates(rows)
