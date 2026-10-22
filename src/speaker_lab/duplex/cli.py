"""CPU-safe command routing; model and audio dependencies are loaded on demand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DuplexLab speech data and offline ASR experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Validate and freeze a new protocol")
    for name in ("manifest", "data-root", "config", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    audit = commands.add_parser("audit", help="Audit source audio without model inference")
    for name in ("protocol", "data-root", "output"):
        audit.add_argument(f"--{name}", type=Path, required=True)
    run = commands.add_parser("run", help="Run audited offline ASR in an isolated environment")
    for name in ("protocol", "data-root", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--config", type=Path, help="Optional check: must match the frozen config")
    score = commands.add_parser("score", help="Score a complete run, separating audio origins")
    score.add_argument("--run", type=Path, required=True)
    for name in ("protocol", "data-root", "output"):
        score.add_argument(f"--{name}", type=Path)
    for name in ("review", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "report":
            command.add_argument("--public", action="store_true")
    reviews = commands.add_parser("import-reviews")
    reviews.add_argument("--run", type=Path, required=True)
    reviews.add_argument("--file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from .protocol import freeze_protocol, read_json, verify_protocol, write_json
        if args.command == "freeze":
            result = freeze_protocol(args.manifest, args.data_root, args.config, args.output)
        elif args.command == "audit":
            import tomllib
            from .audio import audit_clip, find_exact_duplicates
            records = verify_protocol(args.protocol, args.data_root)
            config = tomllib.loads((args.protocol / "config.toml").read_text(encoding="utf-8"))
            args.output.mkdir(parents=True, exist_ok=False)
            rows = [audit_clip(r, args.data_root, args.output / "audio", config.get("audit", {})) for r in records]
            duplicates = find_exact_duplicates(rows)
            write_json(args.output / "audit.json", rows)
            write_json(args.output / "duplicates.json", duplicates)
            result = {"audited": len(rows), "duplicates": len(duplicates)}
        elif args.command == "run":
            from .runner import run_offline
            if args.config and args.config.read_bytes() != (args.protocol / "config.toml").read_bytes():
                raise ValueError("Runtime config must match the frozen protocol; freeze a new protocol to change it")
            state = run_offline(args.protocol, args.data_root, args.output, args.resume)
            result = {k: state.get(k) for k in ("status", "expected", "succeeded", "batch_subprocess_seconds", "error")}
            print(json.dumps(result, ensure_ascii=False))
            return 0 if state["status"] == "complete" else 1
        elif args.command == "score":
            from .metrics import score_run
            state = read_json(args.run / "run.json")
            result = score_run(args.run, args.protocol or Path(state["protocol_dir"]),
                               args.data_root or Path(state["data_root"]))
            write_json(args.output or args.run / "summary.json", result)
            result = {"status": result["status"], "origins": result["origins"]}
        elif args.command == "review":
            from .review import build_review_packet
            result = {"path": str(build_review_packet(args.run, args.output))}
        elif args.command == "import-reviews":
            from .review import import_reviews
            result = import_reviews(args.run, args.file)
        else:
            from .report import render_report
            result = {"path": str(render_report(args.run, args.output, args.public))}
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
