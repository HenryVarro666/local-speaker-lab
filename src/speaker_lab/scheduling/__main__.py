"""Explicit startup only: python -m speaker_lab.scheduling --help."""

from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PersonaPlex simulated scheduling Demo (loopback only)"
    )
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--model-url", default="http://127.0.0.1:8998")
    parser.add_argument(
        "--tool-voice",
        choices=("native", "native-controlled", "device"),
        default="native-controlled",
        help="Prefer exact tool wording with native audio; device voice remains the fallback",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Explicit text-only tool testing; no native model",
    )
    parser.add_argument("--vosk-model", type=Path)
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        help="Optional fixed business date for repeatable demos",
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        help="Explicit opt-in to local transcript/tool event storage",
    )
    args = parser.parse_args(argv)
    from aiohttp import web
    from .server import create_app

    if not args.diagnostic and not args.vosk_model:
        parser.error("Live mode requires --vosk-model for voice-operated tools")
    app = create_app(
        model_url=None if args.diagnostic else args.model_url,
        vosk_path=args.vosk_model,
        today=args.today,
        event_dir=args.event_dir,
        tool_voice=args.tool_voice,
    )
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
