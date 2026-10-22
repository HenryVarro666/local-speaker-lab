#!/usr/bin/env python3
"""Paced synthetic-speech or diagnostic flow against the actual scheduling gateway.

This saves evidence, never substitutes a fake model, and requires native audio
when --require-model is set. Audio recordings are explicitly opt-in via --output.
"""

from __future__ import annotations
import argparse
import asyncio
from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import time
import re


def native_fact_matches(text, start, reference=None, *, strict_date=False):
    """Narrow development guard; waveform ASR/listening remain separate evidence."""
    value = datetime.fromisoformat(start)
    text = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    text = re.sub(r"\b([ap])\s+m\b", r"\1m", text)
    if strict_date:
        ordinals = ("zeroth first second third fourth fifth sixth seventh eighth ninth "
                    "tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth "
                    "seventeenth eighteenth nineteenth").split()
        day = value.day
        spoken_day = ordinals[day] if day < 20 else (
            {20: "twentieth", 30: "thirtieth"}.get(day)
            or ("twenty " if day < 30 else "thirty ") + ordinals[day % 10]
        )
        cardinals = ("zero one two three four five six seven eight nine ten eleven twelve "
                     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
        cardinal_day = cardinals[day] if day < 20 else (
            ("twenty" if day < 30 else "thirty")
            + (" " + cardinals[day % 10] if day % 10 else "")
        )
        calendar_date = (
            rf"\b{value.strftime('%B').lower()}\s+"
            rf"(?:{day}(?:st|nd|rd|th)?|{spoken_day}|{cardinal_day})\b"
        )
        if not re.search(calendar_date, text.replace("-", " ")):
            return False
    hour = value.hour % 12 or 12
    words = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    )
    suffix = "pm" if value.hour >= 12 else "am"
    number = (
        rf"\b{hour}(?::00)?\s*{suffix}\b"
        if value.minute == 0
        else rf"\b{hour}:30\s*{suffix}\b"
    )
    spoken = (
        rf"\b{words[hour]}\s+"
        + (r"thirty\s+" if value.minute else r"(?:o.clock\s+)?")
        + suffix
        + r"\b"
    )
    if value.strftime("%A").lower() not in text or not (
        re.search(number, text) or re.search(spoken, text)
    ):
        return False
    if reference:
        for count, word in ((3, "three"), (2, "two")):
            text = re.sub(
                rf"\b(?:{count}|{word})\s+(?:zeros|zeroes|0s)\b(?:\s+followed\s+by\b)?",
                " ".join(["zero"] * count), text,
            )
        for spoken, expanded in (("triple", "zero zero zero"), ("double", "zero zero")):
            text = re.sub(rf"\b{spoken}[\s-]+(?:zero|0)\b", expanded, text)
        for digit, word in enumerate(words):
            text = re.sub(rf"\b{word}\b", str(digit), text)
        identifier = re.fullmatch(r"([a-z]+)-?(\d+)", reference.lower())
        if not identifier:
            return False
        letters, expected_digits = identifier.groups()
        pattern = (
            r"\b" + r"[\s.]*".join(re.escape(c) for c in letters)
            + r"[\s,:-]*((?:\d[\s,-]*)+)"
        )
        identifiers = [re.sub(r"\D", "", m.group(1)) for m in re.finditer(pattern, text)]
        alias_pattern = (
            r"\b" + r"[\s.]*".join(re.escape(c) for c in letters)
            + r"[\s,:-]+number\s+((?:\d[\s,-]*)+)"
        )
        identifiers.extend(
            "{:04d}".format(int(re.sub(r"\D", "", m.group(1))))
            for m in re.finditer(alias_pattern, text)
        )
        return bool(identifiers) and all(digits == expected_digits for digits in identifiers)
    return True


async def run(args):
    import aiohttp
    import numpy as np
    import soundfile as sf

    validator = None
    if args.native_check == "waveform":
        import torch
        from qwen_asr import Qwen3ASRModel
        from huggingface_hub import snapshot_download

        torch.set_num_threads(4)
        snapshot = snapshot_download(
            "Qwen/Qwen3-ASR-0.6B",
            revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
            local_files_only=True,
        )
        validator = Qwen3ASRModel.from_pretrained(
            snapshot,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    args.output.mkdir(parents=True, exist_ok=False)
    events = []
    queue = deque()
    chunks = []
    latest = {}
    ready = False
    current_epoch = 0
    last_voice = -100
    closed = False
    started = time.monotonic()
    actions = []
    audio_manifest = []
    readbacks = []
    proposal_readbacks = []
    native_jobs = []
    waveform_checks = []
    eventfile = (args.output / "events.jsonl").open("w")

    def record(event):
        event = {
            "client_elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            **event,
        }
        events.append(event)
        eventfile.write(json.dumps(event) + "\n")
        eventfile.flush()

    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(args.url.rstrip("/") + "/ws") as ws:

            async def read():
                nonlocal latest, ready, current_epoch, last_voice
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(message.data)
                        record(event)
                        if event["kind"] == "state":
                            latest = event["state"]
                        if event["kind"] == "playback_reset":
                            current_epoch = event["epoch"]
                            ready = False
                        if event["kind"] == "model_ready":
                            ready = True
                        if (
                            event["kind"] == "native_control"
                            and event.get("phase") == "applied"
                        ):
                            ready = True
                        if event["kind"] == "native_readback":
                            native_jobs.append(event)
                        if (
                            event["kind"] == "model_text"
                            and native_jobs
                            and args.receipt_mode == "native"
                        ):
                            current_text = "".join(
                                e.get("text", "")
                                for e in events
                                if e["kind"] == "model_text"
                                and e.get("epoch") == current_epoch
                            ).lower()
                            if re.search(
                                r"welcome to|how can i (?:help|assist)|thank you for calling",
                                current_text,
                            ):
                                raise RuntimeError(
                                    "Repeated introduction after tool update"
                                )
                        if event["kind"] == "receipt_readback":
                            readbacks.append(event)
                        if event["kind"] == "proposal_readback":
                            proposal_readbacks.append(event)
                        if (
                            event["kind"] in ("proposal_readback", "receipt_readback")
                            and args.require_model
                            and args.receipt_mode == "native"
                        ):
                            raise RuntimeError(
                                "Device voice fallback in a native-required run"
                            )
                        if event["kind"] == "model_error" and args.require_model:
                            raise RuntimeError(event.get("message", event.get("error")))
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        epoch = struct.unpack("<I", message.data[:4])[0]
                        pcm = np.frombuffer(message.data[4:], dtype="<f4").copy()
                        chunks.append((time.monotonic() - started, epoch, pcm))
                        if (
                            epoch == current_epoch
                            and float(np.sqrt(np.mean(pcm**2))) > 0.01
                        ):
                            last_voice = time.monotonic()

            async def feed():
                deadline = time.monotonic()
                while not closed:
                    pcm = queue.popleft() if queue else np.zeros(480, dtype=np.float32)
                    await ws.send_bytes(pcm.astype("<f4").tobytes())
                    deadline += 0.02
                    await asyncio.sleep(max(0, deadline - time.monotonic()))

            reader = asyncio.create_task(read())
            feeder = asyncio.create_task(feed()) if args.audio_dir else None

            async def wait(predicate, label):
                deadline = time.monotonic() + 60
                while not predicate():
                    if reader.done():
                        await reader
                    if time.monotonic() > deadline:
                        raise TimeoutError(label)
                    await asyncio.sleep(0.02)

            def current_native_fact(receipt=False):
                fact = latest.get("booking" if receipt else "proposal")
                text = "".join(
                    e.get("text", "")
                    for e in events
                    if e["kind"] == "model_text" and e.get("epoch") == current_epoch
                )
                return (
                    ready
                    and bool(fact)
                    and native_fact_matches(
                        text, fact["start"], fact.get("id") if receipt else None
                    )
                )

            async def wait_native_fact(label, receipt=False):
                if validator is None:
                    return await wait(lambda: current_native_fact(receipt), label)
                deadline = time.monotonic() + 60
                attempt = 0
                while time.monotonic() < deadline:
                    if reader.done():
                        await reader
                    fact = dict(latest.get("booking" if receipt else "proposal") or {})
                    epoch = current_epoch
                    selected = [pcm for _, e, pcm in chunks if e == epoch]
                    if ready and fact and sum(len(p) for p in selected) >= 48000:
                        attempt += 1
                        wave = np.concatenate(selected)
                        path = args.output / f"qa-epoch-{epoch}-attempt-{attempt}.wav"
                        sf.write(path, wave, 24000)
                        result = await asyncio.to_thread(
                            validator.transcribe,
                            audio=str(path.resolve()),
                            language="English",
                            return_time_stamps=False,
                        )
                        text = str(result[0].text)
                        if epoch == current_epoch and native_fact_matches(
                            text, fact["start"], fact.get("id") if receipt else None,
                            strict_date=True,
                        ):
                            check = {
                                "kind": "native_waveform_verified",
                                "epoch": epoch,
                                "purpose": "receipt" if receipt else "proposal",
                                "asr_text": text,
                                "expected": fact,
                                "file": path.name,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "model": "Qwen3-ASR-0.6B",
                                "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
                            }
                            waveform_checks.append(check)
                            record(check)
                            return
                        record(
                            {
                                "kind": "native_waveform_check_pending",
                                "epoch": epoch,
                                "asr_text": text,
                            }
                        )
                    await asyncio.sleep(1.5)
                raise TimeoutError(label + " (independent waveform ASR)")

            async def utterance(name, text):
                overlap = time.monotonic() - last_voice < 0.3
                actions.append(
                    {
                        "action": name,
                        "text": text,
                        "client_elapsed_ms": round(
                            (time.monotonic() - started) * 1000, 3
                        ),
                        "agent_voice_within_previous_300ms": overlap,
                        "epoch": current_epoch,
                    }
                )
                record(
                    {
                        "kind": "smoke_input",
                        "action": name,
                        "origin": "synthetic_speech"
                        if args.audio_dir
                        else "diagnostic_text",
                        "agent_voice_within_previous_300ms": overlap,
                    }
                )
                if args.audio_dir:
                    path = args.audio_dir / f"{name}.wav"
                    wave, sr = sf.read(path, dtype="float32")
                    if sr != 24000 or wave.ndim != 1:
                        raise ValueError("Expected 24 kHz mono input")
                    audio_manifest.append(
                        {
                            "name": name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "samples": len(wave),
                            "sample_rate": sr,
                        }
                    )
                    for i in range(0, len(wave), 480):
                        queue.append(
                            np.pad(
                                wave[i : i + 480],
                                (0, max(0, 480 - len(wave[i : i + 480]))),
                            )
                        )
                    while queue:
                        await asyncio.sleep(0.02)
                else:
                    await ws.send_json({"kind": "text", "text": text})

            result = {
                "status": "failed",
                "input_origin": "synthetic_macOS_Samantha"
                if args.audio_dir
                else "diagnostic_text",
                "real_microphone": False,
                "native_model_required": args.require_model,
            }
            try:
                await wait(lambda: bool(latest), "initial session")
                if args.require_model:
                    await wait(lambda: ready, "native model handshake")
                    await wait(
                        lambda: time.monotonic() - last_voice < 0.3,
                        "initial native greeting audio",
                    )
                await utterance("book", "Book Thursday at three pm")
                await wait(
                    lambda: latest.get("status") == "awaiting_confirmation",
                    "first proposal",
                )
                if args.require_model and args.receipt_mode == "native":
                    await wait(
                        lambda: ready and time.monotonic() - last_voice < 0.3,
                        "native speech before correction",
                    )
                    await asyncio.sleep(0.2)
                elif args.require_model:
                    await wait(
                        lambda: ready and bool(proposal_readbacks),
                        "first controlled proposal instruction",
                    )
                await utterance("correct", "Actually Friday at four thirty")
                await wait(
                    lambda: (
                        latest.get("status") == "awaiting_confirmation"
                        and "16:30" in latest["proposal"]["start"]
                    ),
                    "corrected proposal",
                )
                if args.require_model and args.receipt_mode == "native":
                    await wait_native_fact("native corrected proposal speech")
                    await asyncio.sleep(1)
                elif args.require_model:
                    await wait(
                        lambda: (
                            ready
                            and any(
                                e["proposal"] == latest["proposal"]
                                for e in proposal_readbacks
                            )
                        ),
                        "corrected controlled proposal instruction",
                    )
                await utterance("confirm", "Yes confirm")
                await wait(lambda: latest.get("status") == "booked", "first booking")
                first = dict(latest["booking"])
                if args.require_model:
                    if args.receipt_mode == "native":
                        await wait_native_fact("native receipt speech", receipt=True)
                        await asyncio.sleep(1)
                    else:
                        await wait(
                            lambda: (
                                ready and any(e["receipt"] == first for e in readbacks)
                            ),
                            "device-voice receipt instruction",
                        )
                await utterance("move", "Move it to Thursday at two pm")
                await wait(
                    lambda: (
                        latest.get("status") == "awaiting_confirmation"
                        and "14:00" in latest["proposal"]["start"]
                    ),
                    "reschedule proposal",
                )
                if args.require_model and args.receipt_mode == "native":
                    await wait_native_fact("native reschedule proposal speech")
                    await asyncio.sleep(1)
                elif args.require_model:
                    await wait(
                        lambda: (
                            ready
                            and any(
                                e["proposal"] == latest["proposal"]
                                for e in proposal_readbacks
                            )
                        ),
                        "reschedule controlled proposal instruction",
                    )
                await utterance("confirm", "Yes confirm")
                await wait(
                    lambda: latest.get("status") == "booked", "reschedule receipt"
                )
                final = dict(latest["booking"])
                if args.require_model:
                    if args.receipt_mode == "native":
                        await wait_native_fact("final native speech", receipt=True)
                        await asyncio.sleep(8)
                    else:
                        await wait(
                            lambda: (
                                ready and any(e["receipt"] == final for e in readbacks)
                            ),
                            "final device-voice receipt instruction",
                        )
                        await asyncio.sleep(3)
                assert first["id"] == final["id"] == "BK-0001"
                assert "16:30" in first["start"] and "14:00" in final["start"]
                result.update(
                    status="passed",
                    first_booking=first,
                    final_booking=final,
                    native_pcm_frames=len(chunks),
                    receipt_mode=args.receipt_mode,
                    native_tool_jobs=native_jobs,
                    native_check=args.native_check,
                    waveform_checks=waveform_checks,
                    device_readback_instructions=[
                        {"text": e["text"], "receipt": e["receipt"]} for e in readbacks
                    ],
                    browser_readback_tested=False,
                    correction_during_native_voice=actions[1][
                        "agent_voice_within_previous_300ms"
                    ],
                )
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            finally:
                closed = True
                if feeder:
                    feeder.cancel()
                reader.cancel()
                await asyncio.gather(
                    *([feeder] if feeder else []), reader, return_exceptions=True
                )
                eventfile.close()
                for epoch in sorted({e for _, e, _ in chunks}):
                    selected = [p for _, e, p in chunks if e == epoch]
                    sf.write(
                        args.output / f"agent-epoch-{epoch}.wav",
                        np.concatenate(selected),
                        24000,
                    )
                result.update(
                    native_check=args.native_check,
                    waveform_checks=waveform_checks,
                    native_tool_jobs=native_jobs,
                    actions=actions,
                    input_audio=audio_manifest,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    timing_boundary="paced synthetic input to gateway PCM arrival; no browser or audio hardware",
                )
                (args.output / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(json.dumps(result, indent=2), flush=True)
            return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument(
        "--native-check",
        choices=("metadata", "waveform"),
        default="metadata",
        help="Waveform mode requires the pinned local Qwen ASR environment and verifies actual PCM before confirming",
    )
    parser.add_argument(
        "--receipt-mode",
        choices=("device", "native"),
        default="native",
        help="Native checks model output; waveform mode verifies actual audio. Device checks readback instructions.",
    )
    args = parser.parse_args()
    if args.require_model and not args.audio_dir:
        parser.error("--require-model requires paced --audio-dir")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
