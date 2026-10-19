from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="本地双人分离与转写")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("audio")
    run.add_argument("--config", default="configs/default.toml")
    run.add_argument("--reference-1")
    run.add_argument("--reference-2")
    run.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        from .pipeline import run_pipeline
        references = {speaker: value for speaker, value in (
            ("SPEAKER_00", args.reference_1), ("SPEAKER_01", args.reference_2)) if value}
        try:
            result = run_pipeline(args.audio, args.config, references, progress=lambda message: print(message, flush=True),
                                  summarize=not args.no_summary)
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 1
        print(json.dumps({key: value for key, value in result.items() if key != "segments"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
