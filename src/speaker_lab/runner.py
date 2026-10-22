from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .backends import BUILTINS
from .redaction import Redactor


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def write_json(path: Path, value, *, redactor=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_value = (redactor or Redactor()).value(value)
    path.write_text(json.dumps(safe_value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class WorkerRunner:
    """One process per stage: independent dependencies and deterministic GPU release."""

    def __init__(self, config: dict, run_dir: Path):
        self.config, self.run_dir = config, Path(run_dir).resolve()
        self.redactor = Redactor(config)

    def run(self, stage: str, payload: dict) -> dict:
        if not isinstance(stage, str) or stage not in BUILTINS:
            raise ValueError(f"Unknown worker stage: {stage!r}")
        if not isinstance(payload, dict):
            raise TypeError("Worker payload must be an object.")
        timeout = self.config["workers"].get("timeout_seconds", 3600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("workers.timeout_seconds must be a positive finite number.")
        timeout = float(timeout)
        # Preserve the virtualenv interpreter path: resolving its symlink would
        # launch the base Python and discard this worker's installed packages.
        configured_python = Path(self.config["workers"][f"{stage}_python"]).expanduser()
        python = Path(os.path.abspath(Path(self.config["_root"]) / configured_python))
        if not python.is_file():
            raise RuntimeError(f"Worker Python not found: {python}. Run scripts/install_ubuntu.sh first.")
        stage_dir = self.run_dir / "work" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        request_path, response_path = stage_dir / "request.json", stage_dir / "response.json"
        options = dict(self.config[stage])
        backend = options.pop("backend")
        write_json(request_path, {"stage": stage, "backend": backend, "options": options, "payload": payload},
                   redactor=self.redactor)
        # A retry must never mistake a previous success for the current child's
        # response (e.g. a plugin exits early without writing a response).
        response_path.unlink(missing_ok=True)
        environment = os.environ.copy()
        src = str(Path(self.config["_root"]) / "src")
        environment["PYTHONPATH"] = src + os.pathsep + environment.get("PYTHONPATH", "")
        environment.setdefault("HF_HOME", str(Path(self.config["_root"]) / ".cache" / "huggingface"))
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["PYANNOTE_METRICS_ENABLED"] = "0"
        environment["GRADIO_ANALYTICS_ENABLED"] = "False"
        log_path = stage_dir / "worker.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [str(python), "-m", "speaker_lab.worker", "--request", str(request_path), "--response", str(response_path)],
                    cwd=self.config["_root"], env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                deadline = time.monotonic() + timeout
                lines = queue.Queue()

                def read_output():
                    try:
                        for line in process.stdout:
                            lines.put(line)
                    finally:
                        lines.put(None)

                reader = threading.Thread(target=read_output, daemon=True)
                reader.start()
                try:
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(process.args, timeout)
                        try:
                            line = lines.get(timeout=min(0.2, remaining))
                        except queue.Empty:
                            continue
                        if line is None:
                            break
                        log.write(self.redactor.text(line))
                        log.flush()
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except BaseException:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                    raise
                finally:
                    reader.join(timeout=1)
                    # Killing a timed-out child closes its pipe and releases a
                    # final line without a newline; preserve that diagnostic too.
                    while True:
                        try:
                            tail = lines.get_nowait()
                        except queue.Empty:
                            break
                        if tail is not None:
                            log.write(self.redactor.text(tail))
                    log.flush()
                    if not reader.is_alive():
                        process.stdout.close()
        except subprocess.TimeoutExpired:
            raise RuntimeError(self.redactor.text(f"{stage} timed out. Log: {log_path}")) from None
        if not response_path.is_file():
            raise RuntimeError(f"{stage} exited with code {process.returncode} without a response. Log: {log_path}")
        try:
            response = self.redactor.value(json.loads(response_path.read_text(encoding="utf-8"),
                                                      parse_constant=_reject_json_constant))
        except (ValueError, UnicodeError):
            raise RuntimeError(f"{stage}: worker returned invalid JSON. Log: {log_path}") from None
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise RuntimeError(f"{stage}: invalid worker response envelope. Log: {log_path}")
        if process.returncode or response.get("ok") is not True:
            raise RuntimeError(self.redactor.text(f"{stage}: {response.get('error', 'worker failed')}. Log: {log_path}"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"{stage}: worker result must be an object. Log: {log_path}")
        return result
