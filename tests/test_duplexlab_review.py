import json

import pytest

from test_duplexlab_metrics import result_case


def review_payload(protocol, row):
    from speaker_lab.duplex.protocol import read_json
    return {"schema_version": 1, "protocol_hash": read_json(protocol / "FREEZE.json")["protocol_hash"],
            "clip_id": row["clip_id"], "audio_sha256": row["audio_sha256"], "reviewer_id": "person",
            "listened": True, "decision": "approve", "corrected_text": "你好", "answer_exposed": True,
            "supersedes": None, "reviewed_at": "2026-10-20T15:00:00+00:00"}


def test_review_import_is_idempotent_and_does_not_edit_source(tmp_path):
    from speaker_lab.duplex.review import import_reviews
    from speaker_lab.duplex.store import RunStore
    _, protocol, run, rows = result_case(tmp_path)
    original = (protocol / "manifest.jsonl").read_bytes()
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps([review_payload(protocol, rows[0])]))
    assert import_reviews(run, path)["inserted"] == 1
    assert import_reviews(run, path)["inserted"] == 0
    assert len(RunStore(run / "state.sqlite").iter_reviews()) == 1
    assert (protocol / "manifest.jsonl").read_bytes() == original


@pytest.mark.parametrize("change", [{"listened": False}, {"audio_sha256": "changed"}, {"reviewer_id": ""}])
def test_invalid_review_cannot_become_human_approval(tmp_path, change):
    from speaker_lab.duplex.review import import_reviews
    _, protocol, run, rows = result_case(tmp_path)
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps([{**review_payload(protocol, rows[0]), **change}]))
    with pytest.raises(ValueError):
        import_reviews(run, path)
