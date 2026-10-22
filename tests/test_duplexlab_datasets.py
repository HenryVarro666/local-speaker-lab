import copy
import hashlib
import json

import pytest

from duplex_helpers import make_clip


def records(count=12):
    rows = []
    for i in range(count):
        rows.append({"clip_id": f"c{i}", "split": "train", "origin": "real",
                     "language_group": "en", "source_group_id": "speaker-10",
                     "original_speaker_id": 10, "audio_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                     "pcm_sha256": hashlib.sha256(f"pcm{i}".encode()).hexdigest(),
                     "duration_seconds": 2.0 + i * .01, "reference": "hello there",
                     "label_status": "source_reference", "audit_decision": "pass", "silence_fraction": .1})
    return rows


def test_speaker_split_and_hash_sampling_keep_unseen_people_out_of_train():
    from speaker_lab.duplex.datasets import sample_source_records, split_by_speaker
    rows = records(8)
    for i, row in enumerate(rows):
        row["original_speaker_id"] = [1, 2, 5, 6, 13, 14, 18, 24][i]
    policy = {"excluded": [1, 2], "dev": [5, 6], "language_confirmation": [13, 14], "test": [18, 24]}
    parts = split_by_speaker(rows, policy)
    assert not parts["train"]
    assert {r["original_speaker_id"] for r in parts["test"]} == {18, 24}
    a = sample_source_records(parts["language_confirmation"], {"en": 2}, 42, 1)
    assert a == sample_source_records(list(reversed(parts["language_confirmation"])), {"en": 2}, 42, 1)
    with pytest.raises(ValueError, match="speaker"):
        split_by_speaker(rows, {**policy, "test": [13, 24]})


def test_selection_has_same_quotas_deterministic_seed_and_duration_budget():
    from speaker_lab.duplex.datasets import select_training_arms, stratum_quotas
    pool = records()
    policy = {"seed": 42, "quotas": stratum_quotas(pool, {"en": 4}), "max_duration_difference": .01}
    predictions = [{"clip_id": r["clip_id"], "hypothesis": "hello there" if i < 6 else "bad words"}
                   for i, r in enumerate(pool)]
    arms = select_training_arms(pool, predictions, [], policy)
    assert arms == select_training_arms(list(reversed(pool)), predictions, [], policy)
    assert len(arms["random"]) == len(arms["filtered"]) == 4
    a, b = [sum(r["duration_seconds"] for r in arms[k]) for k in ["random", "filtered"]]
    assert abs(a - b) / a <= .01
    assert all(r["clip_id"] in {f"c{i}" for i in range(6)} for r in arms["filtered"])


def test_selection_excludes_unresolved_rejects_and_synthetic_sources():
    from speaker_lab.duplex.datasets import select_training_arms
    pool = records(5)
    pool[0]["audit_decision"] = "needs_review"
    pool[1]["audit_decision"] = "reject"
    pool[2]["origin"] = "synthetic"
    arms = select_training_arms(pool, [{"clip_id": r["clip_id"], "hypothesis": "hello there"} for r in pool], [],
                                {"seed": 42, "quotas": {"en|1-3": 2}, "max_duration_difference": .01})
    assert {r["clip_id"] for r in arms["filtered"]} == {"c3", "c4"}
    with pytest.raises(ValueError, match="budget|quota"):
        select_training_arms(pool, [], [], {"seed": 42, "quotas": {"en|1-3": 5}})


def test_duration_budget_swaps_have_an_audit_record():
    from speaker_lab.duplex.datasets import select_training_arms
    pool = records()
    for i, row in enumerate(pool):
        row["duration_seconds"] = 1.1 + i * .08
    audit = {}
    arms = select_training_arms(pool, [{"clip_id": r["clip_id"], "hypothesis": "hello there"} for r in pool], [],
                                {"seed": 42, "quotas": {"en|1-3": 4}}, audit=audit)
    assert audit["initial_filtered_ids"] == ["c0", "c1", "c10", "c11"]
    assert audit["duration_swaps"]
    assert audit["duration_difference_fraction"] <= .01
    assert {r["clip_id"] for r in arms["filtered"]} != set(audit["initial_filtered_ids"])


@pytest.mark.parametrize("key", ["audio_sha256", "pcm_sha256", "original_speaker_id", "source_group_id"])
def test_cross_split_leakage_is_rejected(key):
    from speaker_lab.duplex.datasets import validate_partitions
    a, b = records(2)
    b.update(split="dev", source_group_id="speaker-20", original_speaker_id=20)
    b[key] = a[key]
    with pytest.raises(ValueError, match="split|partition"):
        validate_partitions({"train": [a], "dev": [b]})


def test_export_never_copies_test_and_binds_references_and_files(tmp_path):
    from speaker_lab.duplex.datasets import export_training, verify_export
    data = tmp_path / "audio"
    train = make_clip(data, "train", split="train")
    dev = make_clip(data, "dev", split="dev", seconds=1.5)
    train.update(original_speaker_id=10, source_group_id="speaker-10", audit_decision="pass")
    dev.update(original_speaker_id=20, source_group_id="speaker-20")
    out = tmp_path / "export"
    export_training({"random": [train], "filtered": [train]}, data, out, [dev])
    verified = verify_export(out)
    assert verified["roles"]["dev"] == "dev"
    row = json.loads((out / "random.jsonl").read_text())
    assert row["text"].startswith("language None<asr_text>")
    assert not (out / "test.jsonl").exists()
    with pytest.raises(FileExistsError):
        export_training({"random": [train]}, data, out, [dev])
    (out / "random.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="hash"):
        verify_export(out)
    wrong = copy.deepcopy(train)
    wrong["split"] = "test"
    with pytest.raises(ValueError, match="train"):
        export_training({"random": [wrong]}, data, tmp_path / "wrong", [dev])


def test_listened_hash_bound_correction_can_resolve_review_and_preserves_source_text(tmp_path):
    from speaker_lab.duplex.datasets import export_training, select_training_arms
    data = tmp_path / "audio"
    train = make_clip(data, "train", split="train")
    train.update(audit_decision="needs_review", original_speaker_id=10, source_group_id="speaker-10")
    dev = make_clip(data, "dev", split="dev", seconds=1.5)
    dev.update(original_speaker_id=20, source_group_id="speaker-20")
    review = {"clip_id": "train", "audio_sha256": train["audio_sha256"], "decision": "approve", "listened": True, "corrected_text": "corrected"}
    arms = select_training_arms([train], [{"clip_id": "train", "hypothesis": "corrected"}], [review],
                                {"seed": 42, "quotas": {"mixed|1-3": 1}})
    export_training(arms, data, tmp_path / "export", [dev])
    assert arms["filtered"][0]["source_reference"] == train["reference"]
    assert arms["filtered"][0]["reference"] == "corrected"
    assert arms["filtered"][0]["audit_decision"] == "needs_review"
