"""Audited offline ASR through the existing isolated model worker."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import platform
import time
import tomllib
import uuid

from speaker_lab.runner import WorkerRunner

from .audio import audit_clip, find_exact_duplicates
from .protocol import content_hash, file_hash, read_json, verify_protocol, write_json
from .store import RunStore


def run_offline(protocol_dir: Path, data_root: Path, output_dir: Path, resume: bool = False) -> dict:
    protocol_dir, data_root, output_dir = map(lambda p: Path(p).resolve(), (protocol_dir, data_root, output_dir))
    records = verify_protocol(protocol_dir, data_root)
    frozen = read_json(protocol_dir / "FREEZE.json")
    config = tomllib.loads((protocol_dir / "config.toml").read_text(encoding="utf-8"))
    options = dict(config.get("asr", {}))
    for key in ("model", "python", "condition"):
        if not isinstance(options.get(key), str) or not options[key].strip():
            raise ValueError(f"Missing asr.{key}")
    if not isinstance(options.get("language"), str) or options.get("aligner") != "":
        raise ValueError("Offline ASR requires an explicit language string and aligner=''")
    condition = options.pop("condition")
    python = options.pop("python")
    backend = options.pop("backend", "qwen")
    model_artifacts = {}
    if backend == "qwen":
        model_path = Path(options["model"]).expanduser()
        if not model_path.is_dir():
            raise ValueError("Qwen requires a fixed local snapshot, not an unpinned remote model ID")
        if model_path.parent.name == "snapshots" and model_path.name != options.get("model_revision"):
            raise ValueError("Local snapshot does not match the declared model revision")
        model_artifacts = {p.name: file_hash(p) for p in sorted(model_path.iterdir())
                           if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}}
    timeout = options.pop("timeout_seconds", 1800)
    options.setdefault("max_new_tokens", 256)
    cfg_hash = content_hash(config)
    source_root = Path(__file__).resolve().parents[1]
    asr_files = ("runner.py", "worker.py", "redaction.py", "backends/common.py", "backends/qwen.py",
                 "duplex/protocol.py", "duplex/audio.py", "duplex/store.py", "duplex/runner.py")
    asr_sources = {name: file_hash(source_root / name) for name in asr_files}
    if output_dir.exists():
        if not resume:
            raise FileExistsError(output_dir)
        previous = read_json(output_dir / "run.json")
        if previous.get("protocol_hash") != frozen["protocol_hash"] or previous.get("config_hash") != cfg_hash:
            raise ValueError("Cannot resume: protocol or config changed")
        if previous.get("model_artifacts") != model_artifacts:
            raise ValueError("Cannot resume: model provenance missing or changed; start a new run")
        if previous.get("asr_source_hashes") != asr_sources:
            raise ValueError("Cannot resume: ASR source provenance missing or changed; start a new run")
    else:
        output_dir.mkdir(parents=True)
        for name in asr_files:
            snapshot = output_dir / "source" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((source_root / name).read_bytes())
        previous = {
            "schema_version": 1, "run_id": output_dir.name, "status": "running", "mode": "offline",
            "protocol_hash": frozen["protocol_hash"], "config_hash": cfg_hash,
            "protocol_dir": str(protocol_dir), "data_root": str(data_root), "expected": len(records),
            "condition": condition, "model": options["model"], "model_revision": options.get("model_revision"),
            "device": options.get("device"), "platform": platform.platform(),
            "model_artifacts": model_artifacts,
            "asr_source_hashes": asr_sources,
            "timing_scope": "whole model subprocess including model load; no per-clip latency available",
            "config": config,
        }
        write_json(output_dir / "run.json", previous)
    store = RunStore(output_dir / "state.sqlite")
    complete_ids = {r["clip_id"] for r in store.iter_results()}
    audits = [audit_clip(row, data_root, output_dir / "audio", config.get("audit", {})) for row in records]
    duplicates = find_exact_duplicates(audits)
    (output_dir / "audit.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits), encoding="utf-8")
    write_json(output_dir / "duplicates.json", duplicates)
    pending, attempts = [], {}
    by_id = {r["clip_id"]: r for r in records}
    for audit in audits:
        clip_id = audit["clip_id"]
        if clip_id in complete_ids:
            continue
        attempt = store.start_attempt(clip_id, condition)
        attempts[clip_id] = attempt
        if audit["decision"] == "reject":
            store.fail_attempt(clip_id, condition, attempt, ",".join(audit["reasons"]))
            continue
        pending.append({"id": clip_id, "audio_path": audit["normalized_path"], "start": 0, "speaker": "SPEAKER_00"})
    if pending:
        project_root = Path(config.get("project", {}).get("root", Path.cwd())).expanduser().resolve()
        worker_config = {
            "_root": str(project_root), "workers": {"transcribe_python": python, "timeout_seconds": timeout},
            "transcribe": {"backend": backend, **options},
        }
        batch_dir = output_dir / "attempts" / f"batch-{uuid.uuid4().hex[:12]}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        batch = {"directory": str(batch_dir.relative_to(output_dir)), "clip_ids": [j["id"] for j in pending], "status": "running"}
        previous.setdefault("batches", []).append(batch)
        write_json(output_dir / "run.json", previous)
        start = time.perf_counter()
        try:
            result = WorkerRunner(worker_config, batch_dir).run("transcribe", {"jobs": pending})
            elapsed = time.perf_counter() - start
            segments = result.get("segments")
            if not isinstance(segments, list):
                raise ValueError("Worker did not return segments")
            returned = {}
            for segment in segments:
                key = segment.get("job_id")
                if key in returned or key not in {job["id"] for job in pending}:
                    raise ValueError("Worker returned duplicate or unknown job_id")
                if not isinstance(segment.get("text"), str):
                    raise ValueError("Worker returned invalid transcription")
                returned[key] = segment["text"]
            write_json(batch_dir / "worker_result.json", result)
            batch.update(status="returned", subprocess_seconds=elapsed)
            for job in pending:
                cid = job["id"]
                if cid not in returned:
                    store.fail_attempt(cid, condition, attempts[cid], "No final transcription returned")
                    continue
                record = by_id[cid]
                store.finish_attempt(cid, condition, attempts[cid], {
                    "schema_version": 1, "run_id": output_dir.name, "clip_id": cid, "condition": condition,
                    "attempt": attempts[cid], "hypothesis": returned[cid], "mode": "offline",
                    "origin": record["origin"], "audio_sha256": record["audio_sha256"],
                    "model": options["model"], "model_revision": options.get("model_revision"),
                    "config_hash": cfg_hash, "inference_seconds": None, "batch_subprocess_seconds": elapsed,
                })
            previous["batch_subprocess_seconds"] = elapsed
        except Exception as error:
            batch.update(status="failed", error=str(error), subprocess_seconds=time.perf_counter() - start)
            for job in pending:
                cid = job["id"]
                current = next(row for row in store.iter_jobs() if row["clip_id"] == cid)
                if current["status"] == "running":
                    store.fail_attempt(cid, condition, attempts[cid], f"{type(error).__name__}: {error}")
            previous["error"] = str(error)
    results = store.iter_results()
    (output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in results), encoding="utf-8")
    previous.update(status="complete" if len(results) == len(records) else "incomplete", succeeded=len(results),
                    audit_counts=dict(Counter(r["decision"] for r in audits)))
    if previous["status"] == "complete":
        previous.pop("error", None)
    source = Path(__file__).resolve().parents[1]
    previous.setdefault("source_hashes", {str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*.py"))})
    write_json(output_dir / "run.json", previous)
    return previous
