"""Train only an explicitly executed, verified DuplexLab export."""
import argparse
import json
from pathlib import Path

from speaker_lab.duplex.protocol import read_json
from speaker_lab.duplex.training import run_training


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    result = run_training(read_json(args.config), args.export_dir, dry_run=not args.execute)
    print(json.dumps({k: result.get(k) for k in ["status", "exit_code", "subprocess_seconds", "checkpoint", "optimizer_steps"]}))
    return 0 if result["status"] in {"dry_run", "checkpoint_saved_unverified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
