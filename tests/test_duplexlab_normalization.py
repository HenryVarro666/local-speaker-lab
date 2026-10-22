import copy

import pytest


def test_uniform_control_preserves_audio_targets_and_original_policy():
    from speaker_lab.duplex.normalization import uniform_sidecar, validate_loss_contrast

    weighted = {
        "schema_version": 1,
        "entries": [
            {
                "audio": "a.wav",
                "target": "language None<asr_text>嗯嗯",
                "spans": [{"start": 23, "end": 24}],
            },
            {
                "audio": "b.wav",
                "target": "language English<asr_text>hello",
                "spans": [],
            },
        ],
    }
    original = copy.deepcopy(weighted)
    uniform = uniform_sidecar(weighted)
    assert weighted == original
    assert all(not r["spans"] for r in uniform["entries"])
    validate_loss_contrast(weighted, uniform)
    uniform["entries"][0]["target"] += " changed"
    with pytest.raises(ValueError, match="targets"):
        validate_loss_contrast(weighted, uniform)


@pytest.mark.parametrize(
    "change", ["duplicate", "no_boost", "still_weighted", "different_audio"]
)
def test_loss_contrast_rejects_nonisolated_controls(change):
    from speaker_lab.duplex.normalization import uniform_sidecar, validate_loss_contrast

    weighted = {
        "schema_version": 1,
        "entries": [
            {"audio": "a.wav", "target": "x", "spans": [{"start": 0, "end": 1}]}
        ],
    }
    uniform = uniform_sidecar(weighted)
    if change == "duplicate":
        weighted["entries"] *= 2
        uniform["entries"] *= 2
    elif change == "no_boost":
        weighted["entries"][0]["spans"] = []
    elif change == "still_weighted":
        uniform["entries"][0]["spans"] = weighted["entries"][0]["spans"]
    else:
        uniform["entries"][0]["audio"] = "other.wav"
    with pytest.raises(ValueError):
        validate_loss_contrast(weighted, uniform)
