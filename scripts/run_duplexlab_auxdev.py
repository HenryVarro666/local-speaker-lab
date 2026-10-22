"""Evaluate a pre-test-frozen dev coverage supplement after the main pipeline."""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from run_duplexlab_multiseed import load, model_revision, wait_for_gpu
from speaker_lab.duplex.protocol import (
    file_hash,
    freeze_protocol,
    read_json,
    write_json,
)
from speaker_lab.duplex.runner import run_offline
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.multiseed import compare_seeds


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    base = a.model.resolve()
    runs = root / "runs/duplexlab/disfluency-v1"
    data = root / "data/duplexlab/disfluency-v1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    policy = read_json(data / "AUXDEV_POLICY.json")
    assert file_hash(data / "auxdev.jsonl") == policy["manifest_sha256"]
    assert (
        file_hash(root / "experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md")
        == policy["protocol_sha256"]
    )
    freeze = read_json(runs / "AUXDEV_FREEZE.json")
    assert freeze["auxdev_manifest_sha256"] == policy["manifest_sha256"]
    assert freeze["test_base_exists"] is False
    # The primary study must be fully complete. This guard also prevents GPU races.
    assert (runs / "COMPARISON-test.json").exists() and (
        runs / "END_HEALTH.json"
    ).exists()
    source = runs / "auxdev-source"
    source.mkdir(exist_ok=False)
    for path in [
        "scripts/run_duplexlab_auxdev.py",
        "scripts/prepare_duplexlab_auxdev.py",
        "experiments/duplexlab/DISFLUENCY_DEV_ADDENDUM.md",
    ]:
        shutil.copy2(root / path, source / Path(path).name)
    write_json(
        runs / "AUXDEV_SOURCE.json",
        {
            n: file_hash(source / n)
            for n in [
                "run_duplexlab_auxdev.py",
                "prepare_duplexlab_auxdev.py",
                "DISFLUENCY_DEV_ADDENDUM.md",
            ]
        },
    )
    rows = load(data / "auxdev.jsonl")
    scores = {}
    pairs = {s: {} for s in [42, 43, 44]}
    models = {"base": base}
    for seed in [42, 43, 44]:
        for arm in ["random", "filtered"]:
            record = read_json(runs / f"train-{arm}-{seed}-record/training.json")
            assert (
                record["status"] == "trained_and_reloaded" and record["test_complete"]
            )
            models[f"{arm}-{seed}"] = Path(record["checkpoint"])
    for suffix, model in models.items():
        wait_for_gpu()
        name = "auxdev-" + suffix
        config = runs / f"{name}.toml"
        config.write_text(f"""[project]
root = {json.dumps(str(root))}
[asr]
condition = {json.dumps(name)}
model = {json.dumps(str(model))}
model_revision = {json.dumps(model_revision(model, base))}
python = {json.dumps(sys.executable)}
device = "cuda:0"
dtype = "bfloat16"
language = ""
aligner = ""
attn_implementation = "sdpa"
max_new_tokens = 256
timeout_seconds = 7200
""")
        protocol = runs / "protocols" / name
        freeze_protocol(data / "auxdev.jsonl", data, config, protocol)
        state = run_offline(protocol, data, runs / name)
        assert state["status"] == "complete"
        score = score_run(runs / name, protocol, data)
        write_json(runs / name / "summary.json", score)
        scores[suffix] = score
        if suffix != "base":
            arm, seed = suffix.split("-")
            pairs[int(seed)][arm] = score
        print(
            json.dumps(
                {
                    "run": name,
                    "count": state["succeeded"],
                    "seconds": state["batch_subprocess_seconds"],
                    "overall": score["origins"]["real"]["overall"],
                }
            ),
            flush=True,
        )
    write_json(
        runs / "AUXDEV_RESULTS.json",
        {
            "policy": policy,
            "comparison": compare_seeds(pairs, rows),
            "base": scores["base"]["origins"]["real"],
            "data_frozen_before_main_test": True,
            "evaluation_after_main_test": True,
            "strategy_or_models_changed": False,
            "timestamp_unix": time.time(),
        },
    )
    print("AUXDEV_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
