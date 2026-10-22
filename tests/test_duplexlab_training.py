import json
import sys

import pytest

from duplex_helpers import make_clip


def setup_export(tmp_path):
    from speaker_lab.duplex.datasets import export_training
    from speaker_lab.duplex.protocol import file_hash
    data = tmp_path / "audio"
    train = make_clip(data, "train", split="train")
    dev = make_clip(data, "dev", split="dev", seconds=1.5)
    train.update(original_speaker_id=10, source_group_id="speaker-10", audit_decision="pass")
    dev.update(original_speaker_id=20, source_group_id="speaker-20")
    export = tmp_path / "export"
    export_training({"random": [train]}, data, export, [dev])
    script = tmp_path / "upstream.py"
    script.write_text('raise RuntimeError("dry-run must never execute this")\n')
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{}')
    return export, {"arm": "random", "python": sys.executable, "script": str(script),
                    "script_sha256": file_hash(script), "model_path": str(model),
                    "output_dir": str(tmp_path / "checkpoint"), "batch_size": 1, "grad_acc": 8,
                    "lr": 2e-5, "epochs": 1, "save_steps": 1, "save_total_limit": 2}


def test_dry_run_checks_export_and_never_executes_or_creates_checkpoint(tmp_path):
    from speaker_lab.duplex.training import run_training
    export, config = setup_export(tmp_path)
    result = run_training(config, export, dry_run=True)
    assert result["status"] == "dry_run"
    command = result["command"]
    assert command[command.index("--eval_file") + 1] == str(export / "dev.jsonl")
    assert command[command.index("--grad_acc") + 1] == "8"
    assert "--seed" not in command
    assert not (tmp_path / "checkpoint").exists()


@pytest.mark.parametrize("change", ["test_arm", "no_dev", "overwrite", "changed_export", "changed_script"])
def test_training_rejects_invalid_inputs_before_launch(tmp_path, change):
    from speaker_lab.duplex.training import build_training_command
    export, config = setup_export(tmp_path)
    if change == "test_arm":
        config["arm"] = "dev"
    elif change == "no_dev":
        metadata = json.loads((export / "EXPORT.json").read_text())
        metadata["roles"].pop("dev")
        (export / "EXPORT.json").write_text(json.dumps(metadata))
    elif change == "overwrite":
        (tmp_path / "checkpoint").mkdir()
    elif change == "changed_export":
        (export / "random.jsonl").write_text('{}\n')
    else:
        config["script_sha256"] = "0" * 64
    with pytest.raises((ValueError, FileExistsError)):
        build_training_command(config, export)


@pytest.mark.parametrize("kind", ["partial_steps", "nonfinite_gradient", "valid"])
def test_saved_checkpoint_needs_all_steps_and_finite_training_logs(tmp_path, kind):
    from speaker_lab.duplex.training import checkpoint_summary
    for name in ["config.json", "preprocessor_config.json", "tokenizer_config.json", "chat_template.json", "model.safetensors"]:
        (tmp_path / name).write_text('{}')
    history = [{"loss": .4, "grad_norm": float("inf") if kind == "nonfinite_gradient" else 1.5}]
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 1 if kind == "partial_steps" else 2,
                                                            "epoch": 1., "log_history": history}))
    if kind == "valid":
        assert checkpoint_summary(tmp_path, 2)["optimizer_steps"] == 2
    else:
        with pytest.raises(ValueError):
            checkpoint_summary(tmp_path, 2)
