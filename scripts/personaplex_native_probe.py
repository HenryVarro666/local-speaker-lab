#!/usr/bin/env python3
"""Direct native-backend probe. Opt-in synthetic evidence, never browser acceptance."""

from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import time


async def run(args):
    import numpy as np
    import soundfile as sf
    from speaker_lab.scheduling.personaplex import PersonaPlexClient

    args.output.mkdir(parents=True, exist_ok=False)
    script = json.loads(args.script.read_text())
    started = time.monotonic()
    events = []
    chunks = []
    stage = 0
    output_epoch = 0
    eventfile = (args.output / "events.jsonl").open("w")

    async def audio(pcm, epoch):
        chunks.append((stage, epoch, (time.monotonic() - started) * 1000, pcm.copy()))

    async def event(e):
        row = {"ms": round((time.monotonic() - started) * 1000, 3), "stage": stage, **e}
        events.append(row)
        eventfile.write(json.dumps(row) + "\n")
        eventfile.flush()

    client = PersonaPlexClient(args.url, audio, event)
    try:
        for stage, spec in enumerate(script):
            events.append(
                {
                    "kind": "probe_stage",
                    "ms": round((time.monotonic() - started) * 1000, 3),
                    "stage": stage,
                }
            )
            if "prompt" in spec:
                output_epoch = stage + 1
                await client.connect(spec["prompt"], output_epoch, seed=spec.get("seed"))
            if "command" in spec:
                # A zero-PCM initial stage may return before capabilities arrive.
                deadline = time.monotonic() + 2
                while not client.capabilities and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                output_epoch = spec["command"]["epoch"]
                await client.control(
                    spec["command"]["mode"],
                    spec["command"].get("text", ""),
                    output_epoch,
                    pace=spec["command"].get("pace", 3),
                    prefix=spec["command"].get("prefix"),
                    date_pace=spec["command"].get("date_pace"),
                    weekday_pace=spec["command"].get("weekday_pace"),
                )
            spec["expected_epoch"] = output_epoch
            wave = np.zeros(0, dtype=np.float32)
            if spec.get("audio"):
                wave, sr = sf.read(spec["audio"], dtype="float32")
                assert sr == 24000 and wave.ndim == 1
            end = int(spec.get("seconds", 12) * 24000)
            deadline = time.monotonic()
            for offset in range(0, end, 480):
                pcm = np.zeros(480, dtype=np.float32)
                selected = wave[offset : offset + 480]
                pcm[: len(selected)] = selected
                if not await client.feed(pcm):
                    raise ConnectionError(
                        "Native probe lost the backend; inspect events.jsonl"
                    )
                deadline += 0.02
                await asyncio.sleep(max(0, deadline - time.monotonic()))
    finally:
        await client.close()
        eventfile.close()
        (args.output / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        summary = []
        for index, spec in enumerate(script):
            selected = [
                pcm
                for s, e, t, pcm in chunks
                if s == index and e == spec.get("expected_epoch", index + 1)
            ]
            pcm = (
                np.concatenate(selected) if selected else np.zeros(0, dtype=np.float32)
            )
            path = args.output / f"stage-{index}.wav"
            sf.write(path, pcm, 24000)
            summary.append(
                {
                    "stage": index,
                    "label": spec.get("label"),
                    "text": "".join(
                        e.get("text", "")
                        for e in events
                        if e["stage"] == index
                        and e["kind"] == "model_text"
                        and e.get("epoch") == spec.get("expected_epoch", index + 1)
                    ),
                    "samples": len(pcm),
                    "rms": float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0,
                    "first_voiced_ms": next(
                        (
                            round(t, 3)
                            for s, e, t, p in chunks
                            if s == index
                            and e == spec.get("expected_epoch", index + 1)
                            and float(np.sqrt(np.mean(p**2))) > 0.008
                        ),
                        None,
                    ),
                }
            )
        if args.vosk_model:
            from vosk import Model, KaldiRecognizer, SetLogLevel
            from scipy.signal import resample_poly

            SetLogLevel(-1)
            model = Model(str(args.vosk_model))
            for item in summary:
                wave, _ = sf.read(
                    args.output / f"stage-{item['stage']}.wav", dtype="float32"
                )
                if len(wave) == 0:
                    item["waveform_asr"] = ""
                    continue
                rec = KaldiRecognizer(model, 16000)
                rec.AcceptWaveform(
                    (np.clip(resample_poly(wave, 2, 3), -1, 1) * 32767)
                    .astype("<i2")
                    .tobytes()
                )
                item["waveform_asr"] = json.loads(rec.FinalResult()).get("text", "")
        (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "SCRIPT.json").write_text(json.dumps(script, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8998")
    p.add_argument("--vosk-model", type=Path)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(run(p.parse_args()))
