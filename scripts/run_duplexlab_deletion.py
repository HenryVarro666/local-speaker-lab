"""Run one frozen coverage candidate, six fresh SFTs, dev then unseen-clip test."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_duplexlab_multiseed import load, model_revision, wait_for_gpu
from speaker_lab.duplex import seeded_sft, retention_sft, study_sft
from speaker_lab.duplex.coverage import has_filler
from speaker_lab.duplex.deletion import feature_marginals
from speaker_lab.duplex.disfluency import text_features, alignment_edits
from speaker_lab.duplex.datasets import export_training, validate_partitions
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds
from speaker_lab.duplex.protocol import (
    file_hash,
    freeze_protocol,
    read_json,
    write_json,
)
from speaker_lab.duplex.runner import run_offline
from speaker_lab.duplex.training import run_training


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Physical GPU1 required")
    root, model = args.root.resolve(), args.model.resolve()
    data = root / "data/duplexlab/deletion-v1"
    runs = root / "runs/duplexlab/deletion-v1"
    preflight = read_json(runs / "PREFLIGHT.json")
    if preflight["status"] != "verified":
        raise ValueError("Input and loss preflight required")
    for name, digest in preflight["source_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError("Source changed after preflight")
    policy = read_json(data / "POLICY.json")
    for role, digest in policy["manifest_sha256"].items():
        if file_hash(data / f"{role}.jsonl") != digest:
            raise ValueError("Frozen manifest changed")
    if (
        file_hash(root / "experiments/duplexlab/DELETION_PROTOCOL.md")
        != policy["protocol_sha256"]
    ):
        raise ValueError("Protocol changed after freeze")
    if file_hash(runs / "SELECTION.json") != policy["selection_sha256"]:
        raise ValueError("Selection changed after freeze")
    parts = {role: load(data / f"{role}.jsonl") for role in ["train", "dev", "test"]}
    validate_partitions(parts)
    selection = read_json(runs / "SELECTION.json")
    arms = selection["arms"]
    if feature_marginals(arms["random"]) != feature_marginals(arms["filtered"]):
        raise ValueError("Coverage constraints changed")
    if file_hash(data / "LOSS_WEIGHTS.json") != policy["loss_weights_sha256"]:
        raise ValueError("Frozen loss weights changed")
    if (
        file_hash(root / "experiments/duplexlab/DELETION_EVAL_ADDENDUM.md")
        != policy["execution_addendum_sha256"]
    ):
        raise ValueError("Execution addendum changed")
    export = runs / "export"
    export_training(arms, data, export, parts["dev"])
    source = runs / "source"
    source.mkdir(exist_ok=False)
    paths = [
        "scripts/run_duplexlab_deletion.py",
        "scripts/run_duplexlab_multiseed.py",
        "src/speaker_lab/duplex/coverage.py",
        "src/speaker_lab/duplex/disfluency.py",
        "src/speaker_lab/duplex/deletion.py",
        "src/speaker_lab/duplex/retention_sft.py",
        "src/speaker_lab/duplex/study_sft.py",
        "experiments/duplexlab/DELETION_EVAL_ADDENDUM.md",
        "src/speaker_lab/duplex/datasets.py",
        "src/speaker_lab/duplex/seeded_sft.py",
        "src/speaker_lab/duplex/training.py",
        "src/speaker_lab/duplex/multiseed.py",
        "src/speaker_lab/duplex/metrics.py",
        "experiments/duplexlab/DELETION_PROTOCOL.md",
        ".cache/upstream/Qwen3-ASR/finetuning/qwen3_asr_sft.py",
    ]
    for name in paths:
        shutil.copy2(root / name, source / Path(name).name)
    write_json(runs / "SOURCE.json", {name: file_hash(root / name) for name in paths})

    def transcribe(name, records, checkpoint):
        wait_for_gpu()
        manifest = runs / f"{name}.jsonl"
        manifest.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
        config = runs / f"{name}.toml"
        config.write_text(
            f"""[project]
root = {json.dumps(str(root))}
[asr]
condition = {json.dumps(name)}
model = {json.dumps(str(checkpoint))}
model_revision = {json.dumps(model_revision(checkpoint, model))}
python = {json.dumps(sys.executable)}
device = "cuda:0"
dtype = "bfloat16"
language = ""
aligner = ""
attn_implementation = "sdpa"
max_new_tokens = 256
timeout_seconds = 7200
""",
            encoding="utf-8",
        )
        protocol = runs / "protocols" / name
        freeze_protocol(manifest, data, config, protocol)
        state = run_offline(protocol, data, runs / name)
        if state["status"] != "complete":
            raise RuntimeError(f"Incomplete ASR: {name}")
        score = score_run(runs / name, protocol, data)
        write_json(runs / name / "summary.json", score)
        print(
            json.dumps(
                {
                    "run": name,
                    "status": "complete",
                    "count": state["succeeded"],
                    "seconds": state["batch_subprocess_seconds"],
                    "overall": score["origins"]["real"]["overall"],
                }
            ),
            flush=True,
        )
        return score

    if (runs / "test-base").exists():
        raise ValueError("New test already observed")
    write_json(
        runs / "FREEZE_BINDING.json",
        dict(
            timestamp_unix=time.time(),
            test_outputs_absent=True,
            policy_sha256=file_hash(data / "POLICY.json"),
            protocol_sha256=policy["protocol_sha256"],
            loss_weights_sha256=policy["loss_weights_sha256"],
            execution_addendum_sha256=policy["execution_addendum_sha256"],
            source_sha256=read_json(runs / "SOURCE.json"),
        ),
    )
    devbase = transcribe("dev-base", parts["dev"], model)
    for row in parts["dev"]:
        pred = next(x for x in devbase["per_clip"] if x["clip_id"] == row["clip_id"])
        edits = alignment_edits(row["reference"], pred["hypothesis"])
        row.update(
            base_deletion_count=edits["D"],
            base_tail_deletion_count=sum(
                x["index"] >= 0.75 * edits["N"] for x in edits["deleted"]
            ),
        )
    diagnostics = {
        "train_pool": feature_marginals(parts["train"]),
        "arms": {a: feature_marginals(r) for a, r in arms.items()},
        "dev_base": devbase["origins"]["real"],
        "dev_marginals": feature_marginals(
            [
                dict(
                    r,
                    selection_mer=next(
                        x["mer"]
                        for x in devbase["per_clip"]
                        if x["clip_id"] == r["clip_id"]
                    ),
                )
                for r in parts["dev"]
            ]
        ),
        "dev_text_features": {
            r["clip_id"]: text_features(r["reference"]) for r in parts["dev"]
        },
        "dev_filler_counts": dict(
            Counter(
                f"{r['language_group']}|{int(has_filler(r['reference']))}"
                for r in parts["dev"]
            )
        ),
        "strategy_changes_after_dev": False,
    }
    diagnostics.update(
        test_outcomes_observed=False,
        selection_fixed_before_dev=True,
        strategy_adjusted=False,
    )
    write_json(runs / "TRAIN_DEV_DIAGNOSTICS.json", diagnostics)
    write_json(runs / "TRAIN_DEV_FEATURES.json", diagnostics)
    import torch

    actual, devpairs, records = {}, {}, {}
    states = runs / "training-state"
    states.mkdir(exist_ok=False)
    for seed in [42, 43, 44]:
        devpairs[seed] = {}
        for arm in ["random", "selected", "filtered"]:
            wait_for_gpu()
            config = read_json(
                root
                / f"runs/duplexlab/continuation-v1/train-{'random' if arm == 'random' else 'filtered'}.json"
            )
            config.update(
                arm=arm,
                study_eval_batch_size=1,
                study_adapter_sha256=file_hash(study_sft.__file__),
                seed=seed,
                seed_adapter_sha256=file_hash(seeded_sft.__file__),
                output_dir=str(
                    root / f"checkpoints/duplexlab/deletion-v1/{arm}-{seed}"
                ),
                record_dir=str(runs / f"train-{arm}-{seed}-record"),
            )
            if arm == "filtered":
                config.update(
                    objective="retention-v1",
                    objective_adapter_sha256=file_hash(retention_sft.__file__),
                    loss_weights=str(data / "LOSS_WEIGHTS.json"),
                    loss_weights_sha256=policy["loss_weights_sha256"],
                    execution_addendum_sha256=policy["execution_addendum_sha256"],
                )
            write_json(runs / f"train-{arm}-{seed}.json", config)
            result = run_training(config, export, dry_run=False)
            if result["status"] != "checkpoint_saved_unverified":
                raise RuntimeError("Training failed; retain evidence")
            checkpoint = Path(result["checkpoint"])
            binary = torch.load(
                checkpoint / "training_args.bin", map_location="cpu", weights_only=False
            )
            keys = [
                "seed",
                "data_seed",
                "bf16",
                "fp16",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "learning_rate",
                "warmup_ratio",
                "dataloader_num_workers",
                "dataloader_persistent_workers",
                "dataloader_drop_last",
                "save_steps",
                "save_total_limit",
                "logging_steps",
                "max_grad_norm",
                "gradient_checkpointing",
            ]
            observed = {k: getattr(binary, k) for k in keys}
            if observed["seed"] != seed or observed["data_seed"] is not None:
                raise ValueError("Actual seed mismatch")
            actual[f"{arm}-{seed}"] = observed
            write_json(runs / "ACTUAL_ARGUMENTS.json", actual)
            shutil.copy2(
                checkpoint / "trainer_state.json", states / f"{arm}-{seed}.json"
            )
            devpairs[seed][arm] = transcribe(
                f"dev-{arm}-{seed}", parts["dev"], checkpoint
            )
            result.update(
                actual_training_arguments=observed,
                checkpoint_reloaded=True,
                status="trained_and_dev_reloaded",
            )
            if arm == "filtered":
                runtime = read_json(Path(config["output_dir"]) / "LOSS_RUNTIME.json")
                if (
                    runtime["train_calls"] != 500
                    or runtime["boosted_subwords"] <= 0
                    or runtime["sidecar_sha256"] != policy["loss_weights_sha256"]
                ):
                    raise ValueError("Actual weighted-loss execution not established")
                result["loss_runtime"] = runtime
                write_json(runs / f"LOSS_RUNTIME-{seed}.json", runtime)
            records[f"{arm}-{seed}"] = result
            write_json(Path(config["record_dir"]) / "training.json", result)
            print(
                json.dumps(
                    {
                        "train": f"{arm}-{seed}",
                        "steps": result["optimizer_steps"],
                        "status": result["status"],
                        "seconds": result["subprocess_seconds"],
                    }
                ),
                flush=True,
            )
    template = {k: v for k, v in actual["random-42"].items() if k != "seed"}
    if not all(
        {k: v for k, v in row.items() if k != "seed"} == template
        for row in actual.values()
    ):
        raise ValueError("Actual training budgets differ")
    contrasts = {
        "joint": ("random", "filtered"),
        "selection": ("random", "selected"),
        "objective": ("selected", "filtered"),
    }

    def comparisons(role, pairs):
        for name, (left, right) in contrasts.items():
            selected = {
                s: {"random": p[left], "filtered": p[right]} for s, p in pairs.items()
            }
            write_json(
                runs / f"COMPARISON-{role}-{name}.json",
                compare_seeds(selected, parts[role]),
            )

    comparisons("dev", devpairs)
    write_json(
        runs / "DEV_COMPLETE.json",
        {
            "complete_before_test": True,
            "timestamp_unix": time.time(),
            "nine_fresh_checkpoints": True,
            "strategy_changed": False,
            "dev_summary_sha256": {
                f"{a}-{s}": file_hash(runs / f"dev-{a}-{s}/summary.json")
                for s in [42, 43, 44]
                for a in arms
            },
        },
    )
    base = transcribe("test-base", parts["test"], model)
    pairs = {}
    groups = {}
    for seed in [42, 43, 44]:
        pairs[seed] = {}
        groups[str(seed)] = {}
        for arm in ["random", "selected", "filtered"]:
            record = records[f"{arm}-{seed}"]
            score = transcribe(
                f"test-{arm}-{seed}", parts["test"], Path(record["checkpoint"])
            )
            pairs[seed][arm] = score
            record.update(
                status="trained_and_reloaded",
                test_complete=True,
                evaluation_run=f"test-{arm}-{seed}",
            )
            write_json(runs / f"train-{arm}-{seed}-record/training.json", record)
            metadata = {r["clip_id"]: r for r in parts["test"]}
            groups[str(seed)][arm] = {}
            for filler in [False, True]:
                rows = [
                    r
                    for r in score["per_clip"]
                    if has_filler(metadata[r["clip_id"]]["reference"]) == filler
                ]
                counts = {k: sum(r[k] for r in rows) for k in ["S", "D", "I", "N"]}
                groups[str(seed)][arm][str(filler)] = {
                    **counts,
                    "clips": len(rows),
                    "mer": (counts["S"] + counts["D"] + counts["I"]) / counts["N"]
                    if counts["N"]
                    else None,
                }
    comparisons("test", pairs)
    write_json(runs / "FILLER_GROUPS.json", groups)
    write_json(
        runs / "BASE.json",
        {"dev": devbase["origins"]["real"], "test": base["origins"]["real"]},
    )
    import importlib.metadata as metadata

    pins = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
    )
    (runs / "requirements-frozen.txt").write_text("\n".join(pins) + "\n")
    write_json(
        runs / "END_HEALTH.json",
        {
            "timestamp_unix": time.time(),
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            "apps": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
        },
    )


if __name__ == "__main__":
    main()
