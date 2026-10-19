"""Single-speaker local Qwen-ASR glossary pilot; no imports of ML tools until run.

Manifest JSONL: id, audio (relative to manifest), reference, split, group.
Glossary JSON: list of literal terms, shared by every clip in an experiment.
Predictions contain raw ASR text; scoring always reads the separate reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time
import tomllib

from .glossary_metrics import aggregate_scores, normalize_tokens, score_pair


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                records.append(value)
    if not records:
        raise ValueError(f"No records: {path}")
    return records


def load_manifest(path: Path, *, check_audio: bool = False) -> list[dict]:
    rows, seen = read_jsonl(path), set()
    for row in rows:
        for key in ("id", "audio", "reference", "split", "group"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Each row requires a nonempty string {key!r}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate clip id: {row['id']}")
        seen.add(row["id"])
        if not normalize_tokens(row["reference"]):
            raise ValueError(f"Empty normalized reference: {row['id']}")
        row["audio_path"] = str((path.parent / row["audio"]).resolve())
        if check_audio:
            import soundfile as sf
            info = sf.info(row["audio_path"])
            if info.samplerate != 16000 or info.channels != 1 or info.frames == 0:
                raise ValueError(f"Expected nonempty mono 16kHz audio: {row['audio_path']}")
            row["duration_seconds"] = info.frames / info.samplerate
    return rows


def load_terms(path: Path) -> list[str]:
    terms = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(terms, list) or not terms:
        raise ValueError("Glossary must be a nonempty JSON list")
    seen = set()
    for term in terms:
        if not isinstance(term, str) or not term.strip() or "\n" in term or "\r" in term:
            raise ValueError("Glossary terms must be single-line, nonempty strings")
        key = tuple(normalize_tokens(term))
        if not key or key in seen:
            raise ValueError(f"Empty/duplicate normalized glossary term: {term!r}")
        seen.add(key)
    return terms


def make_contexts(personal: list[str], unrelated: list[str]) -> dict[str, str]:
    if len(personal) != len(unrelated):
        raise ValueError("Personal and unrelated lists must have equal item counts")
    if {tuple(normalize_tokens(t)) for t in personal} & {
        tuple(normalize_tokens(t)) for t in unrelated
    }:
        raise ValueError("Personal and unrelated lists must not overlap")
    # Plain vocabulary context, rather than instructions to edit or translate.
    return {"none": "", "personal": "\n".join(personal), "unrelated": "\n".join(unrelated)}


def prediction_index(predictions: list[dict], rows: list[dict]) -> dict:
    ids = {r["id"] for r in rows}
    index = {}
    for prediction in predictions:
        key = prediction.get("id"), prediction.get("condition")
        if key[0] not in ids or key[1] not in {"none", "personal", "unrelated"}:
            raise ValueError(f"Unknown prediction id/condition: {key}")
        if key in index or not isinstance(prediction.get("text"), str):
            raise ValueError(f"Duplicate or invalid prediction: {key}")
        index[key] = prediction
    expected = {(i, c) for i in ids for c in ("none", "personal", "unrelated")}
    if set(index) != expected:
        raise ValueError(f"Incomplete paired predictions; missing {len(expected - set(index))} rows")
    return index


def summarize(rows: list[dict], predictions: list[dict], personal: list[str], unrelated: list[str]) -> dict:
    index = prediction_index(predictions, rows)
    slices = {"all": rows}
    for field in ("group", "split"):
        for value in sorted({r[field] for r in rows}):
            slices[f"{field}:{value}"] = [r for r in rows if r[field] == value]
    report = {"normalization": "nfkc_han_latin_digits_v1", "slices": {}}
    for name, subset in slices.items():
        conditions = {}
        for condition in ("none", "personal", "unrelated"):
            scored = [score_pair(r["reference"], index[r["id"], condition]["text"], personal)
                      for r in subset]
            result = aggregate_scores(scored)
            result["all_candidates"] = aggregate_scores([
                score_pair(r["reference"], index[r["id"], condition]["text"], personal + unrelated)
                for r in subset
            ])
            timings = [index[r["id"], condition].get("seconds") for r in subset]
            if all(isinstance(t, (int, float)) and t >= 0 for t in timings):
                ordered = sorted(timings)
                result["warm_p50_seconds"] = statistics.median(ordered)
                result["warm_p95_seconds"] = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
                duration = sum(index[r["id"], condition].get("duration_seconds", 0) for r in subset)
                if duration > 0:
                    result["rtf"] = sum(timings) / duration
            conditions[condition] = result
        report["slices"][name] = conditions
    return report


def run(args) -> None:
    config_path = Path(args.config).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))["asr"]
    rows = load_manifest(Path(args.manifest).resolve(), check_audio=True)
    personal, unrelated = load_terms(Path(args.personal)), load_terms(Path(args.unrelated))
    contexts = make_contexts(personal, unrelated)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Write provenance before importing or downloading any model.
    provenance = {
        "status": "started", "config": config, "config_sha256": file_hash(config_path),
        "manifest_sha256": file_hash(Path(args.manifest)),
        "personal_sha256": file_hash(Path(args.personal)),
        "unrelated_sha256": file_hash(Path(args.unrelated)), "contexts": contexts,
        "seed": args.seed, "offline": args.offline,
        "audio_sha256": {r["id"]: file_hash(Path(r["audio_path"])) for r in rows},
        "normalization": "nfkc_han_latin_digits_v1",
    }
    log = output / "run.json"
    log.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel
    from .backends.common import read_audio, torch_device

    torch.manual_seed(args.seed)
    if "torch_threads" in config:
        threads = config["torch_threads"]
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("torch_threads must be a positive integer")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch_device(torch, config.get("device", "cuda:0"))
    dtype_name = config.get("dtype", "bfloat16")
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16 or bfloat16")
    source = config["model"]
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        model_path = str(candidate.resolve())
    else:
        model_path = snapshot_download(source, revision=config.get("revision") or None,
                                       local_files_only=args.offline)
    provenance["resolved_model_path"] = model_path
    provenance["versions"] = {name: importlib.metadata.version(name) for name in
                              ("qwen-asr", "transformers", "torch", "numpy", "huggingface-hub")}
    log.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    start = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        model_path, device_map=str(device), dtype=getattr(torch, dtype_name),
        max_inference_batch_size=1, max_new_tokens=config.get("max_new_tokens", 512),
    )
    model.model.eval()
    model.model.generation_config.do_sample = False
    model.model.generation_config.num_beams = 1

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def infer(audio_path: str, context: str):
        # Validate actual samples; both positive and negative arms see identical audio.
        audio = read_audio(audio_path)
        sync()
        begin = time.perf_counter()
        with torch.inference_mode():
            result = model.transcribe(audio=(audio, 16000), context=context,
                                      language=config.get("language") or None,
                                      return_time_stamps=False)
        sync()
        if len(result) != 1:
            raise RuntimeError("Expected exactly one transcription per clip")
        return result[0], time.perf_counter() - begin

    sync()
    provenance["model_load_seconds"] = time.perf_counter() - start
    print(f"Model loaded in {provenance['model_load_seconds']:.2f}s on {device}", flush=True)
    _, provenance["warmup_seconds"] = infer(rows[0]["audio_path"], "")
    provenance["generation_config"] = model.model.generation_config.to_dict()
    log.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Warm-up completed in {provenance['warmup_seconds']:.2f}s; starting paired inference", flush=True)
    predictions = []
    rng.shuffle(rows)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            arms = list(contexts)
            rng.shuffle(arms)
            for condition in arms:
                result, seconds = infer(row["audio_path"], contexts[condition])
                prediction = {"id": row["id"], "condition": condition, "text": result.text,
                              "detected_language": result.language, "seconds": seconds,
                              "duration_seconds": row["duration_seconds"]}
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"{len(predictions) // 3}/{len(rows)} clips complete", flush=True)
    # Use union in all arms so unrelated-list inventions are also counted.
    report = summarize(rows, predictions, personal, unrelated)
    (output / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance["status"] = "complete"
    log.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "scores.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local Mandarin-English personal glossary pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "run", "score"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", required=True)
        item.add_argument("--personal", required=True)
        item.add_argument("--unrelated", required=True)
        if command == "run":
            item.add_argument("--config", default="configs/glossary.toml")
            item.add_argument("--output-dir", required=True)
            item.add_argument("--seed", type=int, default=20261018)
            item.add_argument("--offline", action="store_true")
        if command == "score":
            item.add_argument("--predictions", required=True)
            item.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run(args)
        else:
            rows = load_manifest(Path(args.manifest).resolve(), check_audio=args.command == "check")
            personal, unrelated = load_terms(Path(args.personal)), load_terms(Path(args.unrelated))
            contexts = make_contexts(personal, unrelated)
            if args.command == "check":
                print(json.dumps({"clips": len(rows), "terms_per_list": len(personal),
                                  "context_characters": {k: len(v) for k, v in contexts.items()},
                                  "context_normalized_tokens": {k: len(normalize_tokens(v))
                                                                for k, v in contexts.items()},
                                  "groups": sorted({r["group"] for r in rows})}, ensure_ascii=False))
            else:
                report = summarize(rows, read_jsonl(Path(args.predictions)), personal, unrelated)
                with Path(args.output).open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
