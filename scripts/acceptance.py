#!/usr/bin/env python3
"""Run a real recording and verify GPU requests, separation, ASR and summary artifacts.

This checks execution and artifact integrity, not perceptual speech quality.
Run with the app environment; workers execute in their own configured environments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import numpy as np
import soundfile as sf

SPEAKERS = {"SPEAKER_00", "SPEAKER_01"}
STAGES = ("diarize", "separate", "match", "transcribe")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message: str):
    if not condition:
        raise RuntimeError(message)


def verify_run(run_dir: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    require(manifest.get("status") == "completed", "Pipeline did not complete.")
    responses = {}
    for stage in STAGES:
        stage_dir = run_dir / "work" / stage
        require((stage_dir / "response.json").is_file(),
                f"{stage} did not run. The test recording must contain detectable overlapping speech.")
        response = read_json(stage_dir / "response.json")
        require(response.get("ok") is True, f"{stage} worker failed.")
        request = read_json(stage_dir / "request.json")
        device = request.get("options", {}).get("device", "")
        require(isinstance(device, str) and (device == "cuda" or device.startswith("cuda:")),
                f"{stage} was not requested on CUDA: {device!r}")
        require(bool(manifest.get("models", {}).get(stage)), f"Missing {stage} model metadata.")
        responses[stage] = response["result"]

    matching = read_json(run_dir / "speaker_matching.json")
    matches = matching.get("jobs", [])
    separated = responses["separate"].get("jobs", [])
    match_ids, source_ids = [row["id"] for row in matches], [row["id"] for row in separated]
    require(bool(matches) and len(match_ids) == len(set(match_ids))
            and len(source_ids) == len(set(source_ids)) and set(match_ids) == set(source_ids),
            "Separation and speaker matching need nonempty, corresponding unique job IDs.")
    require(all(set(row["speakers"]) == SPEAKERS and len(row["speakers"]) == 2 for row in matches),
            "Separated sources were not assigned to two distinct speakers.")

    original = sf.info(run_dir / "work" / "input.wav")
    duration = original.frames / original.samplerate
    require(original.samplerate == 16000 and original.channels == 1 and original.frames > 0,
            "Invalid normalized input audio.")
    for speaker in sorted(SPEAKERS):
        path = run_dir / f"{speaker}.wav"
        info = sf.info(path)
        require(info.samplerate == 16000 and info.channels == 1 and info.frames == original.frames,
                f"{speaker} is not a 16 kHz mono track aligned to the input duration.")
        peak = 0.0
        for block in sf.blocks(path, blocksize=16000 * 10, dtype="float32"):
            require(bool(np.isfinite(block).all()), f"{speaker} contains NaN/infinity.")
            peak = max(peak, float(np.abs(block).max(initial=0)))
        require(peak > 1e-7, f"{speaker} is silent.")

    segments = read_json(run_dir / "segments.json")["segments"]
    ids = [row["id"] for row in segments]
    require(bool(ids) and len(ids) == len(set(ids)), "Missing or duplicate segment IDs.")
    require({row["speaker"] for row in segments} == SPEAKERS, "Transcript must contain both speakers.")
    previous = -1.0
    for row in segments:
        start, end = float(row["start"]), float(row["end"])
        require(math.isfinite(start) and math.isfinite(end)
                and 0 <= start <= end <= duration + 0.1 and start >= previous,
                f"Invalid segment timestamps: {row['id']}")
        previous = start
        require(isinstance(row.get("text_raw"), str) and bool(row["text_raw"].strip()),
                f"Missing raw transcript: {row['id']}")
        words = row.get("words", [])
        require(isinstance(words, list) and bool(words),
                f"No aligned words for {row['id']}; this acceptance test requires ForcedAligner.")
        previous_word = -1.0
        for word in words:
            begin, stop = float(word["start"]), float(word["end"])
            require(math.isfinite(begin) and math.isfinite(stop)
                    and max(0, start - 0.1) <= begin <= stop <= min(duration, end) + 0.1
                    and begin >= previous_word and isinstance(word.get("text"), str),
                    f"Invalid aligned word timestamps: {row['id']}")
            previous_word = begin

    summary_path = run_dir / "summary.md"
    require(summary_path.is_file(), "Summary is missing. Inspect manifest.warnings and the Ollama service.")
    summary = summary_path.read_text(encoding="utf-8").strip()
    citations = set(re.findall(r"\bseg_\d+\b", summary))
    require(bool(summary) and bool(citations) and citations.issubset(ids),
            "Summary needs nonempty text and valid transcript segment references.")
    return {"run_dir": str(run_dir.resolve()), "duration_seconds": duration,
            "separation_jobs": len(matches), "transcript_segments": len(segments),
            "summary_citations": sorted(citations), "uncertain_matches": matching.get("uncertain", []),
            "warnings": manifest.get("warnings", []), "automatic_checks": "PASS",
            "manual_review_required": ["speaker identity and separation quality", "transcript accuracy", "summary factual support"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="Real two-speaker audio containing solo turns and overlapping speech")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--reference-1")
    parser.add_argument("--reference-2")
    parser.add_argument("--log-dir", type=Path, help="New directory for GPU trace, result.json and acceptance-report.json")
    parser.add_argument("--run-dir", type=Path, help="Only verify an existing run; no new inference")
    args = parser.parse_args(argv)
    run_dir = args.run_dir
    log_dir = None
    try:
        if run_dir is None:
            require(bool(args.audio), "Provide an audio path, or use --run-dir to verify a completed run.")
            nvidia_smi = shutil.which("nvidia-smi")
            require(bool(nvidia_smi), "nvidia-smi is unavailable. Install/check the driver and run doctor.py first.")
            log_dir = args.log_dir or Path("acceptance_logs") / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
            log_dir.mkdir(parents=True)
            from speaker_lab.pipeline import run_pipeline
            references = {key: path for key, path in (("SPEAKER_00", args.reference_1), ("SPEAKER_01", args.reference_2)) if path}
            print(f"GPU trace: {(log_dir / 'gpu.csv').resolve()}", flush=True)
            with (log_dir / "gpu.csv").open("w", encoding="utf-8") as gpu_log:
                monitor = subprocess.Popen(
                    [nvidia_smi, "--query-gpu=timestamp,name,memory.used,utilization.gpu", "--format=csv", "-l", "1"],
                    stdout=gpu_log, stderr=subprocess.STDOUT,
                )
                try:
                    result = run_pipeline(args.audio, args.config, references or None,
                                          progress=lambda message: print(message, flush=True), summarize=True)
                    run_dir = Path(result["run_dir"])
                    require(monitor.poll() is None, "GPU trace monitor exited early; inspect gpu.csv.")
                finally:
                    if monitor.poll() is None:
                        monitor.terminate()
                        try:
                            monitor.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            monitor.kill()
                            monitor.wait(timeout=10)
            (log_dir / "result.json").write_text(json.dumps({key: value for key, value in result.items() if key != "segments"},
                                                           ensure_ascii=False, indent=2), encoding="utf-8")
        report = verify_run(run_dir)
        destination = (log_dir or run_dir) / "acceptance-report.json"
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"PASS: execution/artifact checks only. Listen to both tracks and verify the transcript and summary. Report: {destination}")
        return 0
    except Exception as error:
        print(f"ACCEPTANCE FAILED: {error}", file=sys.stderr)
        if run_dir:
            print(f"Run artifacts preserved: {run_dir}", file=sys.stderr)
        if log_dir:
            print(f"Acceptance logs: {log_dir.resolve()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
