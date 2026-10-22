# PersonaPlex scheduling Demo

[简体中文](README.zh-CN.md) · [Verification and retained failures](native-injection/RESULTS.md)

An English appointment desk using the unmodified PersonaPlex checkpoint, a separate CPU Vosk observer and versioned simulated booking tools. It supports creation, interruption/correction, explicit confirmation and rescheduling the same reference.

**Proposals and receipts use PersonaPlex's native voice.** Tool updates restore a clean voice/persona prefix inside the same WebSocket and prefill only current facts. The default `native-controlled` mode conditions the complete authoritative tool wording; PersonaPlex generates all audio. **B K number one** maps to the displayed **BK-0001**, and tool context/opening share the same alias. Free conversation history is reset; this is not native function calling. Device voice remains a labeled failure fallback.

Two final paced synthetic-speech flows passed eight key waveform checks; additional Tuesday11AM/Wednesday9:30AM facts passed four more, plus cancellation during prefill. Ten update → first voiced gateway PCM measurements were **1.474–1.734 s**, median **1.552 s**, versus 3.338 s previously (**53.52% lower**). Main-entry native audio was separately verified with diagnostic text and paced zero PCM. [Exact results and retained failures](native-injection/LATENCY_RESULTS.md). These timings exclude observer/query waits, browser/hardware and strict cold start; human microphone, echo and subjective listening remain pending.

Further opt-in prefill testing reached **0.828–0.976 s, median0.898 s** over ten updates, but an additional date waveform guard failed (**11/12 key checks**). It was **not adopted**; the default and main service above remain unchanged. [Candidate, failures and reproduction](native-injection/PREFILL_RESULTS.md).

The date/weekday-pacing follow-up reached **0.778–0.942s, median0.868s** over ten updates and passed20 core fact waveforms. **Actual gateway entry61 failed**, and the user heard day28 rather than20, with the rest unclear except11o'clock. The candidate remains rejected and the1.552s main service stays unchanged. Same-token decoder replay did not establish a fix. [Full follow-up and human evidence](date-clarity-2026-10-21/RESULTS.md).

## Local diagnostic mode

From the repository root:

```bash
uv venv --python 3.12 .venvs/scheduling
uv pip install --python .venvs/scheduling/bin/python -r requirements/scheduling.txt
uv pip install --python .venvs/scheduling/bin/python --no-deps -e .
.venvs/scheduling/bin/python -m speaker_lab.scheduling --diagnostic --today 2026-10-20 --port 8768
```

Open `http://127.0.0.1:8768`. Connect, expand **Diagnostic text controls**, submit `Book Thursday at three pm`, then immediately `Actually Friday at four thirty`. Confirm the current proposal; submit `Move it to Thursday at two pm` and confirm again. Reference BK-0001 stays unchanged. Diagnostics do not establish model speech or microphone acceptance.

## Live GPU configuration

Prepare an isolated Linux GPU environment with the official [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) runtime at revision `3428dfd95309a7f3c84fd93259ded0f810d1ff91`. Accept the [model access conditions](https://huggingface.co/nvidia/personaplex-7b-v1) with your own account, then log into a dedicated HF_HOME interactively. Never put tokens in source, prompts or command arguments. Tested checkpoint revision: `fdaf4090a61cb315c138a1faee287ffd6c716309`.

Download the Apache 2.0 [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models). Vosk supplies business transcripts only; PersonaPlex still receives the original 24 kHz user audio. The optional Vosk wheel in the requirements file targets Linux; Mac text diagnostics do not need it.

Create the local backend extension from the reviewed upstream source:

```bash
python scripts/prepare_personaplex_native.py \
  upstream/personaplex/moshi/moshi/server.py \
  --output upstream/personaplex/moshi/moshi/server_native_cached.py
```

This verifies source/patch hashes, preserves `server.py`, and creates `server_native_cached.py`. A different existing candidate is not overwritten. The extension also makes the host binding effective. Start after the pinned assets have been prepared in the isolated cache:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/hf-home" HF_HUB_CACHE="$PWD/models/hub" \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m moshi.server_native_cached --host 127.0.0.1 --port 8998 --static none

PYTHONPATH="$PWD/src" .venv/bin/python -m speaker_lab.scheduling \
  --model-url http://127.0.0.1:8998 --tool-voice native-controlled \
  --vosk-model models/vosk-model-small-en-us-0.15 \
  --today 2026-10-20 --port 8768
```

Forward the gateway with `ssh -N -L 8769:127.0.0.1:8768 USER@GPU_HOST` and open `http://127.0.0.1:8769`. Click **Connect**, then **Start microphone** when ready. Use headphones. Both remote services bind only loopback.

Voice options:

- `native`: experimental guided opening/alias and sampled date/time; extra waveform checks failed, so it is not the selected Demo mode.
- `native-controlled` (default): complete tool wording conditioned in PersonaPlex's text stream; all audio generated natively.
- `device`: explicit device-voice path. An unsupported/unavailable native backend also uses the labeled device fallback.

The displayed business date is fixed for the Demo, America/Chicago, weekdays 09:00–17:00 in half-hour slots. Hours 1–5 default to afternoon without am/pm. Say weekday and time together; **yes confirm** authorizes a write. Plain **yes**, **okay** and backchannels do not. **Never mind** withdraws a pending change. Bookings are in-memory per session; no actual calendar or customer data is used.

## Verification

```bash
.venvs/scheduling/bin/python -m pytest tests/test_scheduling_*.py -q

# Native PCM and model-text check; waveform verification is still separate.
.venvs/scheduling/bin/python scripts/scheduling_smoke.py \
  --url http://127.0.0.1:8769 --audio-dir runs/scheduling/synthetic \
  --require-model --receipt-mode native --output runs/scheduling/native-01

# Stronger check, in a separate QA environment with the pinned cached Qwen ASR model.
# This ASR validator is not part of the Demo runtime.
HF_HUB_OFFLINE=1 .venvs/asr-mac/bin/python scripts/scheduling_smoke.py \
  --url http://127.0.0.1:8769 --audio-dir runs/scheduling/synthetic \
  --require-model --receipt-mode native --native-check waveform \
  --output runs/scheduling/native-waveform-01
```

The synthetic folder requires `book/correct/confirm/move.wav`, each 24 kHz mono; actual input is paced. Output folders are never overwritten. Waveform mode uses cached Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, CPU FP32, English, and stores PCM hashes and independent transcripts. A text-channel omission can coexist with correctly spoken audio; matching text tokens can also coexist with incorrect audio. These are checked separately.

The gateway does not record audio by default. `--event-dir runs/events` opts into local transcript/tool events; the browser exports the same trace. Its native first-chunk scheduling/end callbacks establish browser lifecycle only, not hardware audibility or a full-utterance listening review. Keep runs, recordings, tokens and weights out of source releases. Exact behavior, failures, timing boundaries and remaining acceptance are in [RESULTS.md](native-injection/RESULTS.md).
