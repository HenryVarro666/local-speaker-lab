# Native tool-result injection: results

**Latest: [clean-prefix latency and current speech acceptance](LATENCY_RESULTS.md).** The following measurements describe the preserved pre-cache native version. Current source metadata is in BACKEND.json; the historical patch/metadata are preserved separately.

## Pre-cache native version

2026-10-20, America/Chicago. No fine-tuning. The selected Demo mode is **guided native speech**: a fixed reply opening and the saved reference steer the text stream; PersonaPlex generates the date/time continuation and all audio. An optional fully controlled native readback conditions the complete tool sentence. Browser device voice remains the failure fallback.

## Delivered change

Tool updates use a local extension of the official binary WebSocket protocol. They recondition LM/Mimi state **inside the existing connection**, using the cached voice embeddings and only current tool facts. The controller's booking state stays intact. This prevents old conversational facts from competing with a corrected time. It is a model-context reset, not preserved KV history or native function calling.

The gateway invalidates stale playback immediately, waits for the control acknowledgement, then releases audio for that epoch. A lock keeps the new Opus header behind its acknowledgement. The client tolerates a temporarily empty/None decoder result and attributes disconnection to the current epoch. Meaningful new user speech releases the prior readback. An eight-second no-voiced-PCM watchdog falls back to labeled device voice.

`--tool-voice native` prefers guided speech; `native-controlled` prefers the exact tool sentence with explicit PAD/EPAD word timing; `device` retains the original fallback path. The exact backend patch and hashes are in [server-native-uncached.patch](server-native-uncached.patch) and [PRE_CACHE_BACKEND.json](PRE_CACHE_BACKEND.json). The original upstream source is preserved; [prepare_personaplex_native.py](../../../scripts/prepare_personaplex_native.py) verifies its revision/hash before creating a separate module.

## Actual audio acceptance

| Run | Input and verification | Result |
| --- | --- | --- |
| guided-waveform-08 | Paced synthetic English audio, actual Vosk/tools, independent CPU Qwen ASR before each confirmation | Full flow passed, 62.765 s, 638 native PCM chunks |
| guided-waveform-repeat-10 | Same frozen audio/protocol, separate session | Full flow passed, 62.710 s, 637 chunks |
| novel-guided-09 | Additional Tuesday 11:00 AM and Wednesday 9:30 AM tool facts; separate native waveform ASR | Both proposals and both receipts matched |
| full-controlled-native-07 | Complete tool text conditioned, native audio generated; separate waveform ASR | Full flow passed, 51.942 s, 498 chunks; four fact checks matched |

Both guided full runs executed Thursday 15:00 → interruption to Friday 16:30 → confirmation → reschedule to Thursday 14:00 → confirmation. The saved reference remained BK-0001. Each run had one model handshake, zero device fallbacks and correction during a native voiced frame within the previous 300 ms. This is a streamed synthetic overlap condition, not a hardware audible-stop measurement.

The selected guided mode has **12 matching key waveform checks**: four in each full run plus four additional-date checks. They cover four unique date/time values; repeated observations are not new independent tasks. The first proposal in each full run was intentionally interrupted and excluded from the full-readback denominator. Correct text tokens alone never count as audio acceptance.

Qwen3-ASR-0.6B used pinned revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, CPU FP32, English, no aligner or ASR fine-tuning. It is an independent automated content check, not human listening or a guarantee of natural prosody. Exact texts and WAV hashes are in [SUMMARY.json](SUMMARY.json) and ignored raw runs.

## Timing boundaries

Across ten tool updates in the two guided full sessions, gateway update-event → first same-epoch PCM with RMS > 0.008 was **3232.1–3457.0 ms**, median **3338.2 ms**. These measurements exclude user end-of-speech, observer/query delay, browser scheduling and audio hardware. Weights were already loaded; this is not strict cold start or stable p95. The earlier 4.0–5.9 s numbers ended at model-ready after reconnection and are not a paired speed comparison.

Full-run elapsed times include paced inputs, protocol waits and independent ASR validation. The controlled run's ASR happened afterward, so its elapsed time cannot be compared as a speed win.

## Retained failures and corrections

- The reconnect baseline spoke a new introduction instead of the saved receipt. An initial probe also used a wrong synthetic path; that harness failure is retained and excluded from model conclusions.
- Initial append-context probes failed on Opus stream ordering/nullable decoder handling. These are transport failures; they do not prove model silence.
- Corrected append probe 04 did speak Friday 16:30 and BK-0001, supported by both Vosk and Qwen waveform checks. Full-flow 01/02 still restarted introductions. One-word steering in 03 retained the obsolete Thursday time. None passed acceptance.
- Fresh context plus one word in 04 still introduced the desk. A complete fixed opening in 05 produced the correct time but omitted the reference. The saved reference therefore became part of the guided opening.
- Metadata-guard run 06 stopped when the text channel omitted the hour 2. **Independent waveform ASR heard “two p.m.” correctly.** It remains a failed/incomplete run; the failure was an over-strict text-based harness, not established spoken-hour omission. Subsequent runs verify actual PCM before confirming.
- The first full-text forcing experiment used repeated EPAD 0 and generated wrong waveform content despite matching text tokens. Both ASRs rejected it. The corrected controlled alternative uses PAD 3 and word-end EPAD 0. Its native audio passed separate fact checks.

Engineering protocols changed between these runs; they must not be pooled into a success rate. Failed source snapshots, logs and audio remain under ignored `runs/scheduling/native-injection/` locally and `runs/native-injection/` in the isolated H100 runtime.

## Limits and remaining work

Human headset/microphone, acoustic echo, subjective speech quality and hardware latency are still untested. Unconstrained dynamic function calling is not implemented. Guided openings/reference wording are controlled; date/time continuation is sampled. The optional controlled mode conditions all tool wording. The UI/tool receipt remains authoritative, and the small development pilot is not the planned frozen benchmark or a production-readiness claim.

Runtime uses only GPU 0 on the separate H100 host. No apex service, shared package, training job, Git commit or push was changed. Setup and experiment metadata contain no credentials. The prior device-voice baseline is preserved in [BASELINE_REPORT.md](BASELINE_REPORT.md).

The final local suite passed **581 tests**, with **1 optional skip** and **22 subtests**; scheduling tests passed **52**. The source archive builds/installs, the patch reproduces exact reviewed backend bytes, and eight deployed scheduling source hashes match. Python/JavaScript syntax and Ruff F checks passed. Browser lifecycle evidence and final cleanup are recorded in the numeric summary.

## Final browser and cleanup check

The promoted localhost Demo was separately exercised with labeled diagnostic text. It ignored the obsolete query, created Friday 16:30 and rescheduled the same BK-0001 to Thursday 14:00. Four native readbacks each emitted first-voiced-chunk scheduled/end callbacks: eight callbacks, one handshake, zero device fallbacks. This establishes browser Web Audio lifecycle, not hardware audibility, complete listening quality or microphone acceptance. Raw browser evidence is in the workspace samples directory; exact hashes and callbacks are in SUMMARY.

The page is reset for a new user session. Only loopback gateway 8768, native backend 8999 and the 8769 SSH forward remain; the owned 8998 baseline, 8767 candidate and 8766 tunnel were stopped. GPU 0 was about 19623 MiB of 95830 MiB after cleanup. Other GPU workloads were not modified. Health model_ready is false while no browser session is connected; the backend remains loaded and listening. The final focused distribution/package check passed all 12 tests.
