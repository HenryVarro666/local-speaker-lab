"""Study-only seed/eval-batch adapter; legacy seed adapter is unchanged."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def execute(upstream: Path, seed: int, arguments: list[str]) -> None:
    from transformers import set_seed

    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    spec = importlib.util.spec_from_file_location("duplexlab_pinned_sft", upstream)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_arguments = module.TrainingArguments

    def seeded_arguments(*args, **kwargs):
        if "seed" in kwargs or "data_seed" in kwargs:
            raise ValueError(
                "Upstream now controls seeds; review adapter compatibility"
            )
        # data_seed=None preserves the upstream default: Trainer falls back to seed.
        return original_arguments(
            *args, seed=seed, per_device_eval_batch_size=1, **kwargs
        )

    module.TrainingArguments = seeded_arguments
    previous = sys.argv
    try:
        sys.argv = [str(upstream), *arguments]
        set_seed(seed)
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args, remaining = parser.parse_known_args()
    execute(args.upstream, args.seed, remaining)


if __name__ == "__main__":
    main()
