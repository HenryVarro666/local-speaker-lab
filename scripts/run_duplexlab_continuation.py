"""Run frozen continuation stages in an isolated GPU experiment directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from speaker_lab.duplex.datasets import export_training, sample_source_records, select_training_arms, validate_partitions
from speaker_lab.duplex.comparison import compare_runs
from speaker_lab.duplex.metrics import score_run
from speaker_lab.duplex.protocol import content_hash, file_hash, freeze_protocol, read_json, write_json
from speaker_lab.duplex.runner import run_offline
from speaker_lab.duplex.training import run_training


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--phase", choices=["pool", "train", "evaluate"], required=True)
    args = p.parse_args()
    root, model = args.root.resolve(), args.model.resolve()
    data = root / "data/duplexlab/continuation-v1"
    runs = root / "runs/duplexlab/continuation-v1"
    runs.mkdir(parents=True, exist_ok=True)
    policy = read_json(data / "POLICY.json")
    parts = {role: load_rows(data / f"{role}.jsonl") for role in policy["manifest_sha256"]}
    for role, sha in policy["manifest_sha256"].items():
        if file_hash(data / f"{role}.jsonl") != sha:
            raise ValueError("Manifest changed after pool/quotas freeze")
    validate_partitions(parts)

    def transcribe(name, records, checkpoint=model):
        manifest = runs / f"{name}.jsonl"
        manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        revision = model.name if checkpoint == model else content_hash({p.name: file_hash(p) for p in checkpoint.iterdir() if p.suffix in {".json", ".safetensors"}})
        text = f'''[project]
root = {json.dumps(str(root))}
[asr]
condition = {json.dumps(name)}
model = {json.dumps(str(checkpoint))}
model_revision = {json.dumps(revision)}
python = {json.dumps(sys.executable)}
device = "cuda:0"
dtype = "bfloat16"
language = ""
aligner = ""
attn_implementation = "sdpa"
max_new_tokens = 256
timeout_seconds = 7200
'''
        config = runs / f"{name}.toml"
        config.write_text(text, encoding="utf-8")
        protocol = runs / "protocols" / name
        freeze_protocol(manifest, data, config, protocol)
        state = run_offline(protocol, data, runs / name)
        if state["status"] != "complete":
            raise RuntimeError(f"Incomplete ASR run: {name}, {state['succeeded']}/{state['expected']}")
        score = score_run(runs / name, protocol, data)
        write_json(runs / name / "summary.json", score)
        print(json.dumps({"run": name, "succeeded": state["succeeded"], "batch_seconds": state["batch_subprocess_seconds"]}), flush=True)
        return state

    if args.phase == "pool":
        transcribe("pool-base", parts["train"])
        return
    if args.phase == "evaluate":
        transcribe("test-base", parts["test"])
        for arm in ["random", "filtered"]:
            record = read_json(runs / f"train-{arm}-record/training.json")
            state = transcribe("test-" + arm, parts["test"], Path(record["checkpoint"]))
            record.update(checkpoint_reloaded=True, evaluation_run=state["run_id"], status="trained_and_reloaded")
            write_json(runs / f"train-{arm}-record/training.json", record)
        comparison = compare_runs({name: runs / ("test-" + name) for name in ["base", "random", "filtered"]})
        write_json(runs / "SFT_COMPARISON.json", comparison)
        write_json(runs / "SFT_FILTERED_VS_RANDOM.json", compare_runs({name: runs / ("test-" + name) for name in ["random", "filtered"]}))
        return

    predictions = load_rows(runs / "pool-base/predictions.jsonl")
    selection_audit = {}
    arms = select_training_arms(parts["train"], predictions, [], policy, audit=selection_audit)
    durations = {k: sum(r["duration_seconds"] for r in v) for k, v in arms.items()}
    write_json(runs / "SELECTION.json", {"policy_hash": content_hash(policy), "arms": arms, "audit": selection_audit,
               "duration_seconds": durations, "counts": {k: len(v) for k, v in arms.items()},
               "overlap": len({r["clip_id"] for r in arms["random"]} & {r["clip_id"] for r in arms["filtered"]})})
    export = runs / "export"
    export_training(arms, data, export, parts["dev"])
    smoke = runs / "smoke-export"
    export_training({"random": sample_source_records(parts["train"], {"mixed": 4, "zh": 2, "en": 2}, 20261020)},
                    data, smoke, sample_source_records(parts["dev"], {"mixed": 2, "zh": 1, "en": 1}, 20261020))
    script = root / ".cache/upstream/Qwen3-ASR/finetuning/qwen3_asr_sft.py"
    common = {"python": sys.executable, "script": str(script), "script_sha256": file_hash(script),
              "upstream_commit": "7c6daf77a2421100f5fb066495372c00129d39ff", "trainer_default_seed": 42,
              "model_path": str(model), "batch_size": 1, "grad_acc": 8, "lr": 2e-5, "epochs": 1,
              "save_total_limit": 2, "num_workers": 0, "persistent_workers": 0,
              "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                              "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}}
    steps = [("smoke", "random", smoke, 1), ("random", "random", export, 50), ("filtered", "filtered", export, 50)]
    for name, arm, source, save_steps in steps:
        configuration = {**common, "arm": arm, "save_steps": save_steps,
                         "output_dir": str(root / "checkpoints/duplexlab" / name),
                         "record_dir": str(runs / f"train-{name}-record")}
        write_json(runs / f"train-{name}.json", configuration)
        result = run_training(configuration, source, dry_run=False)
        print(json.dumps({k: result.get(k) for k in ["status", "optimizer_steps", "subprocess_seconds", "checkpoint"]}), flush=True)
        if result["status"] != "checkpoint_saved_unverified":
            raise RuntimeError(f"Training failed: {name}; preserve logs and stop")
        if name == "smoke":
            transcribe("smoke-reload", read_json(source / "EXPORT.json")["records"]["dev"][:1], Path(result["checkpoint"]))
            result.update(checkpoint_reloaded=True, status="trained_and_reloaded")
            write_json(runs / "train-smoke-record/training.json", result)


if __name__ == "__main__":
    main()
