"""Local Qwen3-ASR Transformers backend with actual forced-alignment times."""

import math

from .common import absolute_path, dependency, read_audio, torch_device


def aligned_words(result, offset: float, duration: float) -> list[dict]:
    """Convert upstream ForcedAlignResult; never synthesize word timestamps."""
    timestamps = getattr(result, "time_stamps", None)
    if timestamps is None:
        return []
    items = timestamps.items
    if callable(items):
        raise TypeError("Expected Qwen ForcedAlignResult.items list, received mapping-like timestamps")
    words = []
    for item in items:
        start, end = float(item.start_time), float(item.end_time)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Forced aligner returned an invalid timestamp")
        # Clamp model rounding at clip boundaries; original timeline offset is explicit.
        start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        words.append({"text": str(item.text), "start": offset + start, "end": offset + end})
    return words


class Transcriber:
    def run(self, payload: dict, options: dict) -> dict:
        torch = dependency("torch", "ASR")
        qwen = dependency("qwen_asr", "ASR")
        device = torch_device(torch, options.get("device", "cuda:0"))
        model_name = options.get("model", "Qwen/Qwen3-ASR-1.7B")
        aligner = options.get("aligner", "Qwen/Qwen3-ForcedAligner-0.6B") or None
        dtype_name = options.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype must be bfloat16, float16 or float32")
        dtype = getattr(torch, dtype_name)
        kwargs = {
            "dtype": dtype,
            "device_map": str(device),
            "max_inference_batch_size": 1,
            "max_new_tokens": int(options.get("max_new_tokens", 512)),
        }
        if aligner:
            kwargs.update(forced_aligner=aligner, forced_aligner_kwargs={
                "dtype": dtype, "device_map": str(device)
            })
        if options.get("attn_implementation"):
            kwargs["attn_implementation"] = options["attn_implementation"]
        model = qwen.Qwen3ASRModel.from_pretrained(model_name, **kwargs)
        segments = []
        for job in payload["jobs"]:
            path = str(absolute_path(job["audio_path"]))
            duration = len(read_audio(path)) / 16000
            offset = float(job["start"])
            with torch.inference_mode():
                results = model.transcribe(
                    audio=path, language=options.get("language", "Chinese") or None,
                    return_time_stamps=bool(aligner),
                )
            if len(results) != 1:
                raise RuntimeError("Expected one ASR result per input clip")
            result = results[0]
            text = str(result.text).strip()
            if not text:
                continue
            words = aligned_words(result, offset, duration)
            segments.append({
                "job_id": job["id"], "speaker": job["speaker"],
                "start": min(word["start"] for word in words) if words else offset,
                "end": max(word["end"] for word in words) if words else offset + duration,
                "text": text, "words": words,
            })
        return {"segments": segments, "model": model_name, "aligner": aligner}
