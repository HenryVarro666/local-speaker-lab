import pytest


def test_optimal_paths_distinguish_repeat_position_ambiguity_and_forced_loss():
    from speaker_lab.duplex.deletion import optimal_operations

    assert optimal_operations("我很很清楚", "我很清楚")[1:3] == [{"M", "D"}, {"M", "D"}]
    assert optimal_operations("我真的清楚", "我清楚")[1:3] == [{"D"}, {"D"}]
    assert optimal_operations("了", "吧") == [{"S"}]


def test_retention_spans_exclude_unk_and_representation_but_keep_repetition():
    from speaker_lab.duplex.deletion import retention_spans

    spans = retention_spans("[UNK] 我很很清楚 API 2", "我很清楚")
    assert all(s["token"] not in {"unk", "2"} for s in spans)
    assert [(s["token"], s["weight"]) for s in spans if s["token"] == "很"] == [
        ("很", 2.0),
        ("很", 2.0),
    ]
    assert not any(s["token"] == "api" for s in spans)
    assert all(
        "[UNK] 我很很清楚 API 2"[s["start"] : s["end"]].lower() == s["token"]
        for s in spans
    )


def test_offset_weights_mask_prompt_and_keep_header_eos_unweighted():
    from speaker_lab.duplex.deletion import offset_weights

    offsets = [(0, 3), (3, 8), (8, 9), (9, 10), (0, 0)]
    assert offset_weights(
        offsets, [-100, 5, 6, 7, 8], [{"start": 8, "end": 9, "weight": 2.0}]
    ) == [0.0, 1.0, 2.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        offset_weights(offsets, [1], [])


def test_new_selector_keeps_original_marginals_and_deletion_bins():
    from speaker_lab.duplex.deletion import feature_weights
    from speaker_lab.duplex.disfluency import feature_weights as old

    row = dict(
        reference="我很很清楚",
        language_group="zh",
        duration_seconds=3.0,
        original_speaker_id=8,
        selection_mer=0.2,
        base_deletion_count=1,
        base_tail_deletion_count=0,
    )
    weights = feature_weights(row)
    assert all(weights[k] == v for k, v in old(row).items())
    assert weights["base_deletions|zh|1"] == 1
    assert weights["tail_deletion|zh|0"] == 1


def test_weighted_training_command_binds_objective_and_sidecar(tmp_path):
    import json
    from speaker_lab.duplex import retention_sft
    from speaker_lab.duplex.protocol import file_hash
    from speaker_lab.duplex.training import build_training_command
    from test_duplexlab_training import setup_export

    export, config = setup_export(tmp_path)
    sidecar = tmp_path / "weights.json"
    sidecar.write_text(json.dumps({"schema_version": 1, "entries": []}))
    config.update(
        seed=42,
        seed_adapter_sha256=file_hash(
            __import__("speaker_lab.duplex.seeded_sft", fromlist=["x"]).__file__
        ),
        objective="retention-v1",
        objective_adapter_sha256=file_hash(retention_sft.__file__),
        loss_weights=str(sidecar),
        loss_weights_sha256=file_hash(sidecar),
    )
    command = build_training_command(config, export)
    assert "speaker_lab.duplex.retention_sft" in command
    assert command[command.index("--loss-weights") + 1] == str(sidecar)
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="weights"):
        build_training_command(config, export)


def test_weighted_causal_ce_shifts_masks_and_has_finite_gradient():
    torch = pytest.importorskip("torch")
    from speaker_lab.duplex.retention_sft import weighted_causal_loss

    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[-100, 0, 1, -100]])
    weights = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
    got = weighted_causal_loss(logits, labels, weights)
    expected = (
        torch.nn.functional.cross_entropy(logits[0, 0:1], labels[0, 1:2])
        + 2 * torch.nn.functional.cross_entropy(logits[0, 1:2], labels[0, 2:3])
    ) / 3
    assert torch.allclose(got, expected)
    got.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0, 2:], torch.zeros_like(logits.grad[0, 2:]))
    plain = weighted_causal_loss(logits, labels, torch.ones_like(weights))
    assert torch.allclose(
        plain,
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1), ignore_index=-100
        ),
    )
    with pytest.raises(ValueError):
        weighted_causal_loss(logits, labels, torch.zeros_like(weights))


def test_selector_uses_new_objective_without_reclassifying_difficulty():
    from speaker_lab.duplex.coverage import coverage_select
    from speaker_lab.duplex.deletion import feature_weights

    pool = []
    for i, (mer, cost) in enumerate([(0.15, 0.1), (0.05, 0.2), (0.1, 0.05)]):
        pool.append(
            dict(
                clip_id=f"train-{i}",
                reference="我清楚",
                language_group="zh",
                duration_seconds=2.0,
                original_speaker_id=8,
                selection_mer=mer,
                selection_cost=cost,
                base_deletion_count=1,
                base_tail_deletion_count=0,
                split="train",
                origin="real",
                audit_decision="pass",
                pcm_sha256=str(i),
            )
        )
    rows, audit = coverage_select(
        pool,
        [pool[0]],
        feature_weights=feature_weights,
        objective=lambda r: r["selection_cost"],
    )
    assert rows == [pool[2]]
    assert audit["marginals"]["difficulty|zh|medium"] == 1
    assert audit["coverage_objective"] == 0.05


def test_study_eval_batch_adapter_is_explicit_and_hash_bound(tmp_path):
    from speaker_lab.duplex import study_sft, seeded_sft
    from speaker_lab.duplex.protocol import file_hash
    from speaker_lab.duplex.training import build_training_command
    from test_duplexlab_training import setup_export

    export, config = setup_export(tmp_path)
    config.update(
        seed=42,
        seed_adapter_sha256=file_hash(seeded_sft.__file__),
        study_eval_batch_size=1,
        study_adapter_sha256=file_hash(study_sft.__file__),
    )
    command = build_training_command(config, export)
    assert "speaker_lab.duplex.study_sft" in command
    config["study_eval_batch_size"] = 8
    with pytest.raises(ValueError, match="eval"):
        build_training_command(config, export)
