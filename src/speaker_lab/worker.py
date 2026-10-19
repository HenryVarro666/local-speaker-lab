"""One-stage subprocess entry point, allowing incompatible model environments."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback

from .backends import load_backend
from .redaction import Redactor, RedactingStream


def execute(request: dict) -> dict:
    if not isinstance(request, dict):
        raise TypeError("Worker request must be an object.")
    for key in ("stage", "backend"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"Worker request {key} must be a non-empty string.")
    for key in ("payload", "options"):
        if not isinstance(request.get(key, {}), dict):
            raise TypeError(f"Worker request {key} must be an object.")
    backend = load_backend(request["stage"], request["backend"])
    result = backend.run(request.get("payload", {}), request.get("options", {}))
    if not isinstance(result, dict):
        raise TypeError("Backend run() must return a JSON-serializable dict")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.request.is_absolute() or not args.response.is_absolute():
        parser.error("--request and --response must be absolute paths")
    # Never print requests/options: they may contain local paths or model credentials.
    redactor = Redactor()
    stdout = RedactingStream(sys.stdout, redactor)
    stderr = RedactingStream(sys.stderr, redactor)
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        redactor = Redactor(request)
        stdout.redactor = stderr.redactor = redactor
        with redirect_stdout(stdout), redirect_stderr(stderr):
            response = {"ok": True, "result": execute(request)}
        encoded = json.dumps(redactor.value(response), ensure_ascii=False, allow_nan=False)
        status = 0
    except Exception as exc:
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        response = redactor.value(response)
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        print(response["error"], file=stderr, flush=True)
        status = 1
    finally:
        stdout.finish()
        stderr.finish()
    args.response.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.response.with_name(args.response.name + f".{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.response)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
