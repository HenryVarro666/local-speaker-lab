from __future__ import annotations

import math
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audio import normalize_audio, read_source, write_wav
from .config import load_config, resolve_path
from .runner import WorkerRunner, write_json
from .redaction import Redactor
from .timeline import SPEAKERS, activity_intervals, overlap_jobs, speaker_spans, split_interval

_GPU_QUEUE = threading.Lock()
SR = 16000


def timestamp(seconds: float, srt: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{',' if srt else '.'}{milliseconds:03d}"


def _references(audio, intervals, supplied, config, work_dir) -> dict:
    references = {}
    settings = config["pipeline"]
    minimum = round(settings["min_reference_seconds"] * SR)
    maximum = round(settings["reference_seconds"] * SR)
    for speaker in SPEAKERS:
        output = work_dir / "references" / f"{speaker}.wav"
        if supplied.get(speaker):
            samples = normalize_audio(supplied[speaker], output, 60)[:maximum]
        else:
            solo = sorted(
                (interval for interval in intervals if interval["speakers"] == (speaker,)),
                key=lambda interval: interval["end"] - interval["start"], reverse=True,
            )
            # Use contiguous clean examples; concatenating many tiny turns harms embeddings.
            pieces, remaining = [], maximum
            for interval in solo:
                if interval["end"] - interval["start"] < minimum:
                    continue
                part = audio[interval["start"]:min(interval["end"], interval["start"] + remaining)]
                pieces.append(part)
                remaining -= len(part)
                if remaining <= 0:
                    break
            samples = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        if len(samples) < minimum or not len(samples) or np.max(np.abs(samples)) < 1e-7:
            raise ValueError(
                f"{speaker} lacks {settings['min_reference_seconds']:g}s of solo reference speech. "
                "Upload a clean reference for this person (numbered by first speaking order), then retry. "
                "No arbitrary channel-to-person assignment was made."
            )
        references[speaker] = write_wav(output, samples)
    return references


def _index_job_results(result, jobs, stage):
    """Validate a worker boundary before another model or audio stage starts."""
    expected_ids = {job["id"] for job in jobs}
    rows = result.get("jobs") if isinstance(result, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows):
        raise ValueError("Separation/matching jobs must be a list of objects with string IDs.")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError("Separation/matching result must cover every overlap job exactly once with matching IDs.")
    for row in rows:
        values = row.get("sources" if stage == "separate" else "speakers")
        if (not isinstance(values, list) or len(values) != 2
                or (stage == "match" and (not all(isinstance(value, str) for value in values)
                                         or set(values) != set(SPEAKERS)))):
            raise ValueError("Every overlap job needs two sources matched to two different speakers.")
        if stage == "separate" and not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("Separation/matching source paths must be non-empty strings.")
        if stage == "match":
            if "uncertain" in row and not isinstance(row["uncertain"], bool):
                raise ValueError("Matching uncertainty must be a boolean.")
            margin = row.get("margin")
            if "margin" in row and (isinstance(margin, bool) or not isinstance(margin, (int, float))
                                    or not math.isfinite(margin) or margin < 0):
                raise ValueError("Matching margin must be a finite nonnegative number.")
    return dict(zip(ids, rows))


def _validate_separation_sources(result, jobs):
    """Check files/timing before loading the next model environment."""
    by_id = _index_job_results(result, jobs, "separate")
    for job in jobs:
        sources = [Path(value) for value in by_id[job["id"]]["sources"]]
        if any(not path.is_absolute() or not path.is_file() for path in sources):
            raise ValueError("Separated source paths must be absolute, existing files.")
        if sources[0].samefile(sources[1]):
            raise ValueError("A separation job must return two different source files.")
        for path in sources:
            read_source(path, job["input_end"] - job["input_start"])
    return by_id


def _blend_jobs(audio, tracks, jobs, separated, matched):
    by_id = _index_job_results(separated, jobs, "separate")
    matches = _index_job_results(matched, jobs, "match")
    weights = np.zeros(len(audio), dtype=np.float32)
    uncertain = []
    for job in jobs:
        result, match = by_id[job["id"]], matches[job["id"]]
        begin, stop = job["input_start"], job["input_end"]
        count = stop - begin
        window = np.ones(count, dtype=np.float32)
        left, right = job["start"] - begin, stop - job["end"]
        if left:
            window[:left] = np.linspace(0, 1, left, endpoint=False)
        if right:
            window[-right:] = np.linspace(1, 0, right, endpoint=False)
        # Context stabilizes the model, but only the actual overlap region is replaced.
        write_start = max(begin, job["region_start"])
        write_stop = min(stop, job["region_end"])
        relative = slice(write_start - begin, write_stop - begin)
        weight = window[relative]
        weights[write_start:write_stop] += weight
        for source, speaker in zip(result["sources"], match["speakers"]):
            samples = read_source(source, count)
            tracks[speaker][write_start:write_stop] += samples[relative] * weight
        if match.get("uncertain"):
            uncertain.append({"start": job["start"] / SR, "end": job["end"] / SR,
                              "margin": match.get("margin"), "job_id": job["id"]})
    mask = weights > 0
    for speaker in SPEAKERS:
        tracks[speaker][mask] /= weights[mask]
    return uncertain


def _validate_segments(result, jobs, duration, uncertain):
    by_id = {job["id"]: job for job in jobs}
    segments = []
    rows = result.get("segments")
    if not isinstance(rows, list):
        raise ValueError("ASR segments must be a list.")
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("ASR segments must contain objects.")
        if not isinstance(item.get("job_id"), str) or item["job_id"] not in by_id:
            raise ValueError("ASR returned an unknown job_id.")
        job = by_id[item["job_id"]]
        if item.get("speaker") != job["speaker"]:
            raise ValueError("ASR changed a speaker label.")
        start, end = float(item["start"]), float(item["end"])
        if (not math.isfinite(start) or not math.isfinite(end) or end < start
                or start < job["start"] - 0.1 or end > job["end"] + 0.1
                or start < -0.1 or end > duration + 0.1):
            raise ValueError("ASR returned invalid original-timeline timestamps.")
        # Tolerance accommodates frame rounding, but must not create inverted
        # or outside-job segments when a timestamp is clipped for export.
        start, end = max(0, job["start"], start), min(duration, job["end"], end)
        if end < start:
            raise ValueError("ASR segment lies outside its original-timeline job.")
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError("ASR segment text must be a string.")
        text = text.strip()
        if not text:
            continue
        words = item.get("words", [])
        if not isinstance(words, list):
            raise ValueError("ASR words must be a list.")
        previous_start = -1.0
        normalized_words = []
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                raise ValueError("ASR word must contain text and timestamps.")
            word_start, word_end = float(word["start"]), float(word["end"])
            if (not math.isfinite(word_start) or not math.isfinite(word_end)
                    or word_end < word_start or word_start < start - 0.1 or word_end > end + 0.1
                    or word_start < previous_start):
                raise ValueError("ASR word timestamps must be ordered and within their parent segment.")
            previous_start = word_start
            word_start, word_end = max(start, word_start), min(end, word_end)
            if word_end < word_start:
                raise ValueError("ASR word lies outside its parent segment.")
            normalized_words.append({**word, "start": word_start, "end": word_end})
        entry = {**item, "start": start, "end": end, "text": text, "words": normalized_words,
                 "text_raw": text, "source_audio": f"{job['speaker']}.wav"}
        entry["uncertain"] = any(start < span["end"] and end > span["start"] for span in uncertain)
        segments.append(entry)
    segments.sort(key=lambda entry: (entry["start"], entry["end"], entry["speaker"]))
    for index, entry in enumerate(segments, 1):
        entry["id"] = f"seg_{index:05d}"
    return segments


def _export_transcript(run_dir, segments):
    markdown = ["# 双人转写", "", "人物编号按首次发言顺序；疑似串音/人物匹配不确定的片段需回听。", ""]
    subtitles = []
    for index, segment in enumerate(segments, 1):
        text = segment["text"].replace("\r", " ").replace("\n", " ")
        note = " [人物匹配待复核]" if segment["uncertain"] else ""
        markdown.append(f"**{segment['id']} · {timestamp(segment['start'])}–{timestamp(segment['end'])} · {segment['speaker']}{note}**  \n{text}\n")
        subtitles.append(f"{index}\n{timestamp(segment['start'], True)} --> {timestamp(segment['end'], True)}\n[{segment['speaker']}] {text}\n")
    transcript, srt = run_dir / "transcript.md", run_dir / "transcript.srt"
    transcript.write_text("\n".join(markdown), encoding="utf-8")
    srt.write_text("\n".join(subtitles), encoding="utf-8")
    return transcript, srt


def run_pipeline(audio_path: str, config_path: str = "configs/default.toml", reference_paths: dict | None = None,
                 progress=None, summarize: bool = True) -> dict:
    """Run real inference through isolated workers. There is no simulated fallback."""
    with _GPU_QUEUE:
        return _run(audio_path, config_path, reference_paths or {}, progress or (lambda message: None), summarize)


def _run(audio_path, config_path, references, progress, summarize):
    config = load_config(config_path)
    redactor = Redactor(config)
    settings = config["pipeline"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = resolve_path(config, settings["output_dir"]) / run_id
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True)
    warnings, models = [], {}
    try:
        progress("解码为 16 kHz 单声道，检查长度")
        normalized = work_dir / "input.wav"
        audio = normalize_audio(audio_path, normalized, settings["max_duration_seconds"])
        runner = WorkerRunner(config, run_dir)
        progress("分析两位说话人及重叠区间（首次运行会下载模型）")
        diarization = runner.run("diarize", {"audio_path": str(normalized)})
        intervals, label_mapping = activity_intervals(diarization["turns"], len(audio))
        models["diarize"] = diarization.get("model", config["diarize"].get("model"))
        write_json(run_dir / "diarization.json", {"turns": diarization["turns"], "label_mapping": label_mapping,
                                                  "intervals_samples": intervals, "sample_rate": SR})
        tracks = {speaker: np.zeros_like(audio) for speaker in SPEAKERS}
        for interval in intervals:
            if len(interval["speakers"]) == 1:
                tracks[interval["speakers"][0]][interval["start"]:interval["end"]] = audio[interval["start"]:interval["end"]]
        jobs = overlap_jobs(intervals, len(audio), round(settings["separation_chunk_seconds"] * SR),
                            round(settings["context_seconds"] * SR))
        uncertain = []
        if jobs:
            progress("提取单独发言参考音频")
            anchors = _references(audio, intervals, references, config, work_dir)
            for job in jobs:
                job["input_path"] = write_wav(work_dir / "clips" / f"{job['id']}.wav", audio[job["input_start"]:job["input_end"]])
                job["output_dir"] = str((work_dir / "separated" / job["id"]).resolve())
            progress(f"分离 {len(jobs)} 个重叠片段")
            separated = runner.run("separate", {"jobs": jobs})
            _validate_separation_sources(separated, jobs)
            models["separate"] = separated.get("model", config["separate"].get("model"))
            progress("用声纹将每块的两个声道对应回同一人物")
            matched = runner.run("match", {"references": anchors, "jobs": separated["jobs"]})
            _index_job_results(matched, jobs, "match")
            models["match"] = matched.get("model", config["match"].get("model"))
            uncertain = _blend_jobs(audio, tracks, jobs, separated, matched)
            write_json(run_dir / "speaker_matching.json", {"jobs": matched["jobs"], "uncertain": uncertain})
            if uncertain:
                warnings.append(f"{len(uncertain)} 个重叠片段的人物匹配分数接近，已标记待复核；详见 speaker_matching.json。")
            warnings.append("分离部分为 16 kHz 模型输出，可能有串音、音色或电平变化；浮点 WAV 保留模型电平。")
        else:
            warnings.append("未检测到重叠发言：已按说话人归轨，分离模型未调用。")
        track_paths = {speaker: write_wav(run_dir / f"{speaker}.wav", tracks[speaker]) for speaker in SPEAKERS}
        asr_jobs = []
        for speaker in SPEAKERS:
            for start, end in speaker_spans(intervals, speaker):
                for begin, stop in split_interval(start, end, round(settings["asr_chunk_seconds"] * SR)):
                    job_id = f"asr_{len(asr_jobs):05d}"
                    path = write_wav(work_dir / "asr_clips" / f"{job_id}.wav", tracks[speaker][begin:stop])
                    asr_jobs.append({"id": job_id, "audio_path": path, "speaker": speaker,
                                     "start": begin / SR, "end": stop / SR})
        progress(f"转写 {len(asr_jobs)} 个片段并对齐时间戳")
        transcription = runner.run("transcribe", {"jobs": asr_jobs})
        models["transcribe"] = transcription.get("model", config["transcribe"].get("model"))
        segments = _validate_segments(transcription, asr_jobs, len(audio) / SR, uncertain)
        if not segments:
            warnings.append("ASR 未产生非空文字。音轨已保留，请检查模型日志和录音。")
        segments_path = run_dir / "segments.json"
        write_json(segments_path, {"sample_rate": SR, "duration": len(audio) / SR, "segments": segments,
                                   "speaker_order": list(SPEAKERS), "models": models})
        transcript_path, subtitles_path = _export_transcript(run_dir, segments)
        summary_path = None
        if summarize and segments:
            progress("调用本地 Ollama 生成带出处的纪要")
            from .summarize import summarize_segments
            try:
                summary = summarize_segments(segments, config.get("summary", {}), run_dir / "summary.md")
                warnings.extend(summary.get("warnings", []))
                summary_path = str((run_dir / "summary.md").resolve())
            except Exception as error:
                warnings.append(redactor.text(f"纪要生成失败，音轨和转写已保留：{error}"))
        manifest = {"run_id": run_id, "duration": len(audio) / SR, "sample_rate": SR, "models": models,
                    "config_path": config["_path"], "warnings": warnings, "status": "completed"}
        write_json(run_dir / "manifest.json", manifest)
        archive_path = run_dir / "results.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.iterdir()):
                if path.is_file() and path != archive_path:
                    archive.write(path, path.name)
        progress("处理完成")
        return {"run_dir": str(run_dir), "tracks": track_paths, "segments": segments,
                "segments_path": str(segments_path), "transcript_path": str(transcript_path),
                "subtitles_path": str(subtitles_path), "summary_path": summary_path,
                "archive_path": str(archive_path), "warnings": warnings}
    except Exception as error:
        write_json(run_dir / "error.json", {"error": str(error), "run_dir": str(run_dir)}, redactor=redactor)
        raise RuntimeError(redactor.text(f"{error}\n运行目录及日志：{run_dir}")) from None
