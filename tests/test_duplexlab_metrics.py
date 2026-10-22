import json

import pytest

from duplex_helpers import make_clip, write_manifest


def result_case(tmp_path, references=None, hypotheses=None):
    from speaker_lab.duplex.protocol import freeze_protocol, read_json, write_json
    from speaker_lab.duplex.store import RunStore
    refs = references or ["你好", "one two three four", "fixture text"]
    hyps = hypotheses or ["你", "one two three four", "completely wrong"]
    root = tmp_path / "data"
    records = [make_clip(root, f"c{i}", origin="fixture" if i == 2 else "real", seconds=1.1 + i * .1)
               for i in range(len(refs))]
    for row, reference in zip(records, refs):
        row["reference"] = reference
        row["language_group"] = "zh" if row["clip_id"] == "c0" else "en"
    manifest = write_manifest(tmp_path / "manifest.jsonl", records)
    config = tmp_path / "config.toml"
    config.write_text('[asr]\ncondition="base"\n')
    protocol = tmp_path / "protocol"
    freeze_protocol(manifest, root, config, protocol)
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "run.json", {"status": "complete", "expected": len(refs), "protocol_dir": str(protocol),
                                  "data_root": str(root), "protocol_hash": read_json(protocol / "FREEZE.json")["protocol_hash"],
                                  "mode": "offline", "condition": "base"})
    store = RunStore(run / "state.sqlite")
    for row, hypothesis in zip(records, hyps):
        attempt = store.start_attempt(row["clip_id"], "base")
        store.finish_attempt(row["clip_id"], "base", attempt, {"clip_id": row["clip_id"], "hypothesis": hypothesis,
                             "condition": "base", "audio_sha256": row["audio_sha256"]})
    return root, protocol, run, records


def test_micro_error_rate_and_origin_filter_are_not_mean_of_rates(tmp_path):
    from speaker_lab.duplex.metrics import score_run
    root, protocol, run, _ = result_case(tmp_path)
    result = score_run(run, protocol, root)
    overall = result["origins"]["real"]["overall"]
    assert (overall["N"], overall["D"], overall["S"], overall["I"]) == (6, 1, 0, 0)
    assert overall["mer"] == pytest.approx(1 / 6)
    assert result["excluded_fixture"] == 1


def test_incomplete_run_is_not_a_quality_result(tmp_path):
    from speaker_lab.duplex.metrics import score_run
    root, protocol, run, _ = result_case(tmp_path)
    state = json.loads((run / "run.json").read_text())
    state["status"] = "incomplete"
    (run / "run.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="complete"):
        score_run(run, protocol, root)
