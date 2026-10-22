"""Verified launch of a pinned upstream trainer; dry runs never import Torch."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import time

from .datasets import verify_export, validate_partitions
from .protocol import file_hash, read_json, write_json


def checkpoint_summary(checkpoint: Path, expected_steps: int) -> dict:
    state = read_json(checkpoint / "trainer_state.json")
    history = state["log_history"]
    losses = [r["loss"] for r in history if "loss" in r]
    values = [
        r[k] for r in history for k in ["loss", "grad_norm", "eval_loss"] if k in r
    ]
    required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    ]
    has_weights = (checkpoint / "model.safetensors").is_file() or (
        checkpoint / "model.safetensors.index.json"
    ).is_file()
    if (
        state["global_step"] != expected_steps
        or not losses
        or not all(math.isfinite(v) for v in values)
    ):
        raise ValueError(
            "Checkpoint has incomplete optimizer steps or nonfinite training logs"
        )
    if not has_weights or not all((checkpoint / n).is_file() for n in required):
        raise ValueError("Checkpoint lacks inference weights or processor files")
    return {
        "optimizer_steps": state["global_step"],
        "epoch": state.get("epoch"),
        "losses": losses,
    }


def build_training_command(config: dict, export_dir: Path) -> list[str]:
    root = Path(export_dir).resolve()
    metadata = verify_export(root)
    arm = config["arm"]
    if metadata["roles"].get(arm) != "train" or metadata["roles"].get("dev") != "dev":
        raise ValueError("Training requires a train arm and separate dev, never test")
    validate_partitions(
        {"train": metadata["records"][arm], "dev": metadata["records"]["dev"]}
    )
    if any(r["split"] != "train" for r in metadata["records"][arm]) or any(
        r["split"] != "dev" for r in metadata["records"]["dev"]
    ):
        raise ValueError("Export roles disagree with train/dev records")
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    script = Path(config["script"]).resolve()
    if file_hash(script) != config["script_sha256"]:
        raise ValueError("Upstream training script hash mismatch")
    model = Path(config["model_path"]).resolve()
    if not (model / "config.json").is_file():
        raise ValueError("Training requires a fixed local model")
    command = [
        str(config["python"]),
        str(script),
        "--model_path",
        str(model),
        "--train_file",
        str(root / f"{arm}.jsonl"),
        "--eval_file",
        str(root / "dev.jsonl"),
        "--output_dir",
        str(output),
    ]
    if "seed" in config:
        adapter = Path(__file__).with_name("seeded_sft.py")
        if file_hash(adapter) != config.get("seed_adapter_sha256"):
            raise ValueError("Seed adapter hash mismatch")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
            raise ValueError("Invalid training seed")
        command = [
            str(config["python"]),
            "-m",
            "speaker_lab.duplex.seeded_sft",
            "--upstream",
            str(script),
            "--seed",
            str(config["seed"]),
            *command[2:],
        ]
    if "study_eval_batch_size" in config:
        adapter = Path(__file__).with_name("study_sft.py")
        if config["study_eval_batch_size"] != 1 or "seed" not in config:
            raise ValueError("Study eval batch must be one with a fixed seed")
        if file_hash(adapter) != config.get("study_adapter_sha256"):
            raise ValueError("Study eval adapter hash mismatch")
        command[command.index("speaker_lab.duplex.seeded_sft")] = (
            "speaker_lab.duplex.study_sft"
        )
    if config.get("objective"):
        if config["objective"] != "retention-v1" or "seed" not in config:
            raise ValueError("Unsupported objective or missing seed")
        adapter = Path(__file__).with_name("retention_sft.py")
        weights = Path(config["loss_weights"]).resolve()
        if file_hash(adapter) != config.get("objective_adapter_sha256"):
            raise ValueError("Objective adapter hash mismatch")
        if file_hash(weights) != config.get("loss_weights_sha256"):
            raise ValueError("Loss weights hash mismatch")
        command[
            command.index(
                "speaker_lab.duplex.study_sft"
                if "study_eval_batch_size" in config
                else "speaker_lab.duplex.seeded_sft"
            )
        ] = "speaker_lab.duplex.retention_sft"
        command[command.index("--upstream") : command.index("--upstream")] = [
            "--loss-weights",
            str(weights),
        ]
    defaults = {
        "batch_size": 1,
        "grad_acc": 8,
        "lr": 2e-5,
        "epochs": 1,
        "save_steps": 50,
        "save_total_limit": 2,
        "log_steps": 1,
        "num_workers": 0,
        "persistent_workers": 0,
        "pin_memory": 1,
    }
    for key, value in defaults.items():
        command += ["--" + key, str(config.get(key, value))]
    return command


def run_training(config: dict, export_dir: Path, dry_run: bool = True) -> dict:
    command = build_training_command(config, export_dir)
    record = {
        "status": "dry_run",
        "command": command,
        "upstream_sha256": config["script_sha256"],
        "export_hash": verify_export(export_dir)["export_hash"],
        "configuration": config,
        "checkpoint_reloaded": False,
    }
    if dry_run:
        return record
    output = Path(config["output_dir"]).resolve()
    folder = Path(config.get("record_dir", output.parent / (output.name + "-record")))
    folder.mkdir(parents=True, exist_ok=False)
    record["status"] = "running"
    write_json(folder / "training.json", record)
    start = time.perf_counter()
    with (folder / "trainer.log").open("w") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **config.get("environment", {})},
        )
    record.update(
        status="failed",
        subprocess_seconds=time.perf_counter() - start,
        exit_code=process.returncode,
        timing_scope="trainer subprocess including load, train, dev evaluation and save",
    )
    if process.returncode == 0:
        checkpoints = [
            p
            for p in output.glob("checkpoint-*")
            if re.fullmatch(r"checkpoint-\d+", p.name)
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
            count = len(verify_export(export_dir)["records"][config["arm"]])
            expected_steps = math.ceil(
                count
                / (config.get("batch_size", 1) * config.get("grad_acc", 8))
                * config.get("epochs", 1)
            )
            try:
                checked = checkpoint_summary(checkpoint, expected_steps)
                record.update(
                    status="checkpoint_saved_unverified",
                    checkpoint=str(checkpoint),
                    **checked,
                    checkpoint_hashes={
                        p.name: file_hash(p)
                        for p in checkpoint.iterdir()
                        if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}
                    },
                )
            except (ValueError, OSError, KeyError) as error:
                record["error"] = str(error)
    write_json(folder / "training.json", record)
    return record
