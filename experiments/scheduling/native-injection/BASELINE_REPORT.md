# PersonaPlex scheduling Demo verification

Date: 2026-10-20, America/Chicago. This is an experimental English hybrid voice Agent, without fine-tuning. [English setup](../README.md) · [中文使用](../README.zh-CN.md) · [Numeric evidence](BASELINE_SUMMARY.json) · [Environment and deployed source hashes](BASELINE_ENVIRONMENT.json).

## Delivered behavior

The browser captures 24 kHz mono user audio. PersonaPlex receives it directly and returns native audio/text; a separate local Vosk observer drives the simulated booking controller. Corrections invalidate previous confirmations and stale availability responses. Explicit confirmation creates a booking; rescheduling updates the same booking reference. Tool proposals and final receipts use controlled browser device-voice readback. All three sources are labeled separately.

The Demo is single-user and loopback-only. A local SSH forward exposes the remote gateway to the user's browser. No actual calendar, customer data, clinical task, fine-tuning or DSH deployment is involved.

## Tests and actual model runs

At the recorded source snapshot, the full local suite passed **542 tests**, with **1 optional Gradio UI skip** and **22 subtests passed**. The new scheduling module has **26 passing tests**. Checks include stale tool results, obsolete confirmation, duplicate writes, rescheduling, cancellation of pending proposals, backchannels, wire codecs, bad handshakes/PCM, connection release, cross-origin rejection, and exact controlled speech from saved tool facts. Source distribution builds/installs with the browser page included. Python syntax and Ruff F checks passed.

The initial unchanged suite had 16 bootstrap failures under macOS BSD `realpath`; rerunning with GNU coreutils in PATH passed. No bootstrap implementation was modified. Other DuplexLab work continued in the shared checkout; its edits were preserved. A final focused check passed all 12 packaging/distribution tests. Its earlier attempt had 2 failures while a concurrent DuplexLab manifest referenced a not-yet-created VALIDATION.md; after that report appeared, the unchanged tests passed. The scheduling suite was separately rerun: 26 passed.

Four unique inputs were synthesized using the Mac Samantha voice and streamed at the actual sample rate: booking Thursday 3 pm, correction to Friday 4:30 pm, explicit confirmation, and rescheduling to Thursday 2 pm. Confirmation audio was reused. These are synthetic recordings, not human microphone tests.

| Run | Observed result | Interpretation |
| --- | --- | --- |
| voice-tools-preflight-01 | Passed in 12.582 s, no native output required | Actual Vosk and tools completed both writes, reference BK-0001 preserved |
| native-speech-01 | Transport/tool flow passed, 140 PCM chunks | Context resets produced repeated greetings; accurate native receipt speech was not established |
| native-speech-02 | Failed during acoustic replay | Opus rejected a 1440-sample tail. Reproduced in a real wire test, then packetized/padded to allowed frame sizes |
| native-speech-03 | Flow passed in 57.242 s, 387 PCM chunks; correction during recent native voice | Native text accurately restated Friday 4:30, but final confirmations still restarted the greeting |
| native-speech-04 | Failed waiting for native receipt speech | Tool had saved the booking; native voice became silent after the update |
| hybrid-speech-05 | Failed waiting for corrected-proposal native speech | Extending controlled readback only to final receipts was insufficient |
| hybrid-speech-06 | Passed current hybrid contract in 47.726 s; 223 native PCM chunks | Native greeting/audio, actual Vosk, all tool transitions and exact device-voice instructions completed. Final booking remained BK-0001, Thursday 2026-10-22 14:00 Central |

These runs used changing engineering protocols and cannot be pooled into a success rate. Earlier failed runs remain in the ignored evidence bundle. Run 06 explicitly reports `browser_readback_tested=false`: a separate browser check observed device speech start and completion, using diagnostic text input. That browser check is not a microphone test. A final browser flow against the live H100 gateway also ignored the obsolete query, saved Friday 16:30, then rescheduled the same BK-0001 to Thursday 14:00. Both receipt readbacks and the second proposal reached their end callback. The first proposal was intentionally cancelled by confirmation; its cancellation callback is retained. Device callbacks do not establish hardware audibility.

## Why the implementation is hybrid

The official PersonaPlex endpoint accepts its role prompt at connection startup; this integration refreshes the connection with current facts and replays bounded recent user audio. Native role grounding improved in one corrected-proposal example, but speech after a result was inconsistent. Current controlled tool readback follows the committed tool facts, while PersonaPlex remains the native conversation backend.

Run 06 reset-to-model-ready durations were 4023.7, 5110.9, 5148.9, 5906.0, 5821.7 and 5762.1 ms. These are connection-refresh measurements, not end-of-user-speech to audible-response latencies. The first value is not a strict model cold start: weights were already loaded. Tool readback can start before native refresh finishes. Reducing these resets and restoring reliable native fact-grounded responses remain future work.

## Resources and reproducibility

The user supplied access to a pair of H100 NVL GPUs, each reporting 95830 MiB. This deployment uses only CUDA-visible GPU 0; its initial loaded-model allocation was about 19597 MiB. GPU 1 later had a separate workload, which was not stopped or changed. The original apex services were not accessed or altered.

The isolated data-volume runtime uses Python 3.12.13, PyTorch 2.4.1+cu121 and pinned PersonaPlex source/model revisions recorded in ENVIRONMENT. The upstream copy has one host-binding fix so its server honors loopback. A separate HF_HOME was logged in by the user; credentials are not part of this report, source or evidence exports.

Raw generated audio, paced events, failed attempts and upstream patch remain under local `runs/scheduling/h100/` and the isolated remote runtime. No weights, credentials or user recordings are included in source releases.

## Remaining acceptance

- Human headset microphone use, accents, soft speech, browser/OS permission behavior and acoustic echo are untested.
- Controlled speech uses the browser's device voice, so its voice differs from PersonaPlex. Speech start/end callbacks do not prove hardware audibility or human quality.
- Natural native speech after arbitrary tool updates is not reliable; this version uses explicit context resets and controlled business readback.
- The parser accepts a displayed business week and limited weekday/time expressions. Plain yes/backchannels do not authorize a write. In-memory bookings disappear with a new browser session.
- There is no frozen 60-task benchmark, calibrated p95, training result or production-readiness claim.

No Git commit or push was performed.
