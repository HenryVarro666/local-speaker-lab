"""Hash-bound train-only token-weighted CE adapter; upstream source stays intact."""

from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .deletion import offset_weights
from .protocol import file_hash, write_json


def weighted_causal_loss(logits, labels, weights):
    import torch
    import torch.nn.functional as F

    if labels.shape != weights.shape or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("Loss shape mismatch")
    target = labels[:, 1:].contiguous()
    w = weights[:, 1:].float() * (target != -100)
    if not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid or empty loss weights")
    ce = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(w)
    return (ce * w).sum() / w.sum()


def execute(upstream, seed, sidecar, arguments):
    import torch
    from transformers import set_seed

    if not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    policy = json.loads(sidecar.read_text())
    entries = {r["audio"]: r for r in policy["entries"]}
    if len(entries) != len(policy["entries"]):
        raise ValueError("Duplicate training weight key")
    spec = importlib.util.spec_from_file_location(
        "duplexlab_retention_upstream", upstream
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_args, original_collator, original_trainer = (
        module.TrainingArguments,
        module.DataCollatorForQwen3ASRFinetuning,
        module.CastFloatInputsTrainer,
    )
    output = Path(arguments[arguments.index("--output_dir") + 1])
    audit = dict(
        objective="retention-v1",
        normalization="per-example sum(weight*causal CE)/sum(valid weight); Trainer gradient accumulation",
        weight=2.0,
        sidecar_sha256=file_hash(sidecar),
        adapter_sha256=file_hash(Path(__file__)),
        train_calls=0,
        dev_calls=0,
        boosted_subwords=0,
        supervised_subwords=0,
        exact_suffix_checks=0,
    )

    def seeded_args(*a, **kw):
        if "seed" in kw or "data_seed" in kw:
            raise ValueError("Unexpected upstream seed")
        return original_args(*a, seed=seed, per_device_eval_batch_size=1, **kw)

    class WeightedCollator(original_collator):
        def __call__(self, features):
            batch = super().__call__(features)
            tokenizer = self.processor.tokenizer
            weights = torch.zeros_like(batch["labels"], dtype=torch.float32)
            for i, feature in enumerate(features):
                target = feature["target"]
                entry = entries.get(Path(feature["audio"]).name)
                if entry is not None and entry["target"] != target:
                    raise ValueError("Training target changed after weight freeze")
                encoded = tokenizer(
                    target + (tokenizer.eos_token or ""),
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
                keep = [j for j, t in enumerate(ids) if t != tokenizer.pad_token_id]
                positions = (batch["labels"][i] != -100).nonzero().flatten()
                if [ids[j] for j in keep] != batch["labels"][i, positions].tolist():
                    raise ValueError(
                        "Processor supervision suffix differs from tokenizer offsets"
                    )
                values = offset_weights(
                    [offsets[j] for j in keep],
                    [ids[j] for j in keep],
                    entry["spans"] if entry else [],
                )
                weights[i, positions] = torch.tensor(values)
                audit["train_calls" if entry else "dev_calls"] += 1
                audit["boosted_subwords"] += sum(v > 1 for v in values)
                audit["supervised_subwords"] += len(values)
                audit["exact_suffix_checks"] += 1
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "LOSS_RUNTIME.json", audit)
            batch["retention_weights"] = weights
            return batch

    class WeightedTrainer(original_trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            weights = inputs.pop("retention_weights")
            outputs = model(**inputs)
            loss = weighted_causal_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    module.TrainingArguments = seeded_args
    module.DataCollatorForQwen3ASRFinetuning = WeightedCollator
    module.CastFloatInputsTrainer = WeightedTrainer
    previous = sys.argv
    try:
        sys.argv = [str(upstream), *arguments]
        set_seed(seed)
        module.main()
    finally:
        sys.argv = previous


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--loss-weights", type=Path, required=True)
    args, remaining = p.parse_known_args()
    execute(args.upstream, args.seed, args.loss_weights, remaining)


if __name__ == "__main__":
    main()
