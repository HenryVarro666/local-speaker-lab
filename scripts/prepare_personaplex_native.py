#!/usr/bin/env python3
"""Build the verified local PersonaPlex extension without changing upstream source."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

EXPERIMENT = (
    Path(__file__).resolve().parents[1] / "experiments/scheduling/native-injection"
)


def prepare(source: Path, output: Path, experiment: Path = EXPERIMENT, *, variant="cached"):
    names = {
        "cached": ("BACKEND.json", "server-native.patch"),
        "prefill": ("PREFILL_BACKEND.json", "server-prefill.patch"),
        "calendar": ("CALENDAR_BACKEND.json", "server-calendar.patch"),
    }
    if variant not in names:
        raise ValueError("Unknown backend variant")
    metadata, patch = names[variant]
    meta = json.loads((experiment / metadata).read_text())
    data = source.read_bytes().replace(
        b"web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)",
        b"web.run_app(app, port=args.port, ssl_context=ssl_context)",
    )
    if hashlib.sha256(data).hexdigest() != meta["original_server_sha256"]:
        raise ValueError("Upstream source hash differs from the tested revision")
    if output.exists():
        if (
            hashlib.sha256(output.read_bytes()).hexdigest()
            == meta["candidate_server_sha256"]
        ):
            return output
        raise FileExistsError(
            "Candidate exists with different contents; choose a new --output"
        )
    with tempfile.TemporaryDirectory(prefix="personaplex-native-") as folder:
        folder = Path(folder)
        original = folder / "server.py"
        original.write_bytes(data)
        candidate = folder / "server_native.py"
        subprocess.run(
            [
                "patch",
                "-o",
                str(candidate),
                str(original),
                str((experiment / patch).resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = candidate.read_bytes()
        if hashlib.sha256(result).hexdigest() != meta["candidate_server_sha256"]:
            raise ValueError("Patched backend hash differs from the reviewed candidate")
        compile(result, str(output), "exec")
        output.write_bytes(result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=("cached", "prefill", "calendar"), default="cached",
                        help="cached retains the main backend; prefill/calendar are explicit experiments")
    args = parser.parse_args()
    target = args.output or args.source.with_name("server_native.py")
    print(prepare(args.source.resolve(), target.resolve(), variant=args.variant))


if __name__ == "__main__":
    main()
