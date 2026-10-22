# PersonaPlex scheduling Demo implementation plan

> Execute inline with `superpowers:executing-plans`; the user authorized implementation and requested no subagents. Do not commit or push.

**Goal:** An English speech Demo that creates a simulated appointment, accepts an interruption changing its time, explicitly confirms the current proposal, and reschedules an existing booking.

**Architecture:** Raw 24 kHz microphone PCM reaches PersonaPlex while a CPU Vosk observer produces business transcripts. A versioned controller owns simulated tools. Official PersonaPlex only accepts the role prompt at connection setup: tool facts therefore refresh its connection with the current proposal, receipt and short transcript history. Log these resets; this is a hybrid Agent, not native model function calling.

**Tech stack:** Python 3.12, aiohttp, sphn, Vosk, NumPy/SciPy, browser Web Audio. No fine-tuning, external calendar, paid API, DSH change or apex operation.

**Spec:** `/path/to/projects/voice_ai/docs/superpowers/specs/2026-10-20-native-duplex-scheduling-proposal.md`.

## Constraints and rulings

- Existing `feat/duplexlab` has uncommitted experiments: add a separate `speaker_lab.scheduling` package and tests; preserve existing edits. No worktree/commit is needed for these additive files.
- GPU: user supplied a reachable H100 server. Only the new environment on a writable data volume and GPU 0 may be used. Verify availability before launching. Tokens remain in the user-managed, isolated HF_HOME credential store.
- Initial model access returned HTTP 403. The user completed a separate HF_HOME login; checkpoint access is now verified and the pinned model has been downloaded.
- Baseline initially had 16 `tests/test_bootstrap.py` failures because macOS `realpath` lacks `-m`. With GNU coreutils prepended to PATH the unchanged baseline is 493 passed, 1 skipped, 22 subtests passed. Do not modify bootstrap code.
- Use a standalone scheduling requirements file/environment, not global package changes or automatic model downloads on import.
- No audio recording by default; explicit evidence runs save synthetic audio and event logs under ignored `runs/`. No credentials in prompts, URLs, source or reports.
- First version supports a displayed, fixed business week and 09:00–17:00 slots. It does not infer arbitrary timezones or ambiguous authorization.

## Review focus

1. A correction arrives before an old query completes: stale result cannot replace the latest proposal.
2. Confirmation of an obsolete version or duplicate commit: no unintended or duplicate booking.
3. Backchannel/hesitation/negative answer: no write authorization.
4. PersonaPlex refresh fails: tools remain truthful; clear visible error, no fake voice success.
5. Audio from a cancelled generation/old connection: no stale playback, bound queues, explicit epoch.

## Tasks

### 1. Versioned scheduling domain and simulated tools

Files: `src/speaker_lab/scheduling/{__init__,domain}.py`, `tests/test_scheduling_domain.py`.

- [x] Write failing tests for create/correct/confirm/reschedule, stale query, stale confirmation, unavailable slot, ambiguous time, withdrawal and idempotency.
- [x] Implement `BookingStore`, `SchedulingSession`, `parse_request`, and typed snapshots/events. Tools operate only on current proposal versions.
- [x] Run domain tests and record RED/GREEN.

### 2. PersonaPlex transport, CPU observation and server

Files: `src/speaker_lab/scheduling/{audio,personaplex,server,__main__}.py`, `requirements/scheduling.txt`, `tests/test_scheduling_transport.py`, `tests/test_scheduling_server.py`.

- [x] Write failing protocol tests with a local aiohttp WebSocket fixture: handshake, Opus/PCM, text, reconnect, cancellation, upstream failure, bounded input and Origin checks.
- [x] Implement `PersonaPlexClient` against official `/api/chat` binary kinds 0/1/2. Reset on tool facts; client epochs discard stale audio.
- [x] Implement Vosk final/partial observer, optional text diagnostics and same session business API. Serialize controller mutations; do not block audio on a simulated tool delay.
- [x] Run transport/server tests and a real Vosk synthetic-audio smoke.

### 3. Browser Demo and complete tool-flow evidence

Files: scheduling page assets, `scripts/scheduling_smoke.py`, paired `experiments/scheduling/README*.md`.

- [x] Build a microphone/connect interface, current proposal/receipt, available slots, activity transcript, interruption controls and event trace. Text diagnostics are visibly labeled.
- [x] Run create → correction before confirmation → confirm → reschedule → confirm with actual HTTP/WS tools; save evidence and explicitly identify inference origin.
- [x] Browser-test usable layout, labeled diagnostic input, stale-result cancellation, same-reference rescheduling, live model reconnect and device speech callbacks. Protocol fixtures do not count as real PersonaPlex inference.
- [ ] Human microphone permission, headset speech, acoustic echo, hardware playback and subjective quality acceptance. These require a real-user trial and remain explicitly pending.

### 4. H100 integration and final verification

- [x] Verify model authorization, install isolated upstream runtime and download only licensed assets to the data volume.
- [x] Start loopback PersonaPlex on GPU 0 and scheduling gateway; tunnel to localhost for browser microphone support.
- [x] Run a synthetic speech flow through actual PersonaPlex and Vosk, including speaking-time correction and tool-result refresh; retain audio/events/model SHA and identify any failures.
- [x] Run the full local suite with GNU coreutils PATH, lint/type/syntax checks appropriate to new code, and new server/GPU checks.
- [x] Update workspace status/EVOLUTION and paired guides; report separately automated flow, actual GPU speech and human microphone acceptance. No claim of human validation from synthesized input.

## Execution ledger

- 2026-10-20: Authorized by user to implement without fine-tuning. Baseline verified with GNU coreutils. H100 pair is NVL, 95830 MiB each, idle. Model authorization HTTP 403; account identity supplied to user without exposing the token. Implementation not yet started.
- Domain RED/GREEN: create, correction, current-version confirmation, stale query/confirmation rejection, reschedule/idempotency and early correction invalidation pass. Transport and gateway RED/GREEN also pass; new scheduling suite 25 tests.
- User completed a separate HF login; model access and pinned checkpoint download succeeded. Dedicated runtime on data volume, GPU 0; upstream one-line host-binding fix installed editable and verified loopback listeners. GPU 1 later acquired an unrelated workload; it was not altered.
- Real Vosk + paced synthetic Mac Samantha input completed create → correction → confirm → reschedule → confirm, reference BK-0001 preserved. This preflight did not require native output.
- Native run 01 completed tools and speech overlap, but generated mostly repeated greetings after reconnect; it is transport/flow evidence, not accurate receipt speech acceptance.
- Ruling: shorten the role prompt and replay bounded recent user audio on context refresh — the native model lacks a dynamic tool-result API and needs conversational acoustic context. Replay is explicitly logged and never sent back through the business observer.
- Native run 02 exposed an irregular 1440-sample replay tail rejected by Opus. Added an actual wire regression (RED), packetized/padded to allowed codec frame sizes (GREEN). Native run 03 is running. Keep failed evidence.
- Full local suite at the preceding source snapshot passed 535 tests, 1 optional Gradio skip and 22 subtests using GNU coreutils PATH. Additional unrelated DuplexLab files appeared from concurrent work; preserve them.
- Native run 03 passed real speech/tool flow with 387 received PCM frames and correction during model voice. Corrected Friday 4:30 proposal was spoken accurately, but final receipt speech remained a repeated greeting.
- Ruling: use controlled browser device-voice readback of successful tool receipts, explicitly labeled separately from PersonaPlex — allowed by the proposal's hybrid fallback, needed for a truthful complete Demo. Native output is muted until the next user correction; receipt readback callbacks are logged. Human microphone/speaker acceptance remains pending.
- Added RED/GREEN for receipt text coming from the saved booking rather than an unconfirmed proposal; wire flow checks both first and rescheduled receipt payload. Final scheduling suite is 26 tests. Native run 04 verifies the final server source; browser device-voice callback validation follows.

- Final native run 04 failed waiting for receipt speech; hybrid run 05 also failed waiting for corrected-proposal native speech. Preserve both. Controlled browser voice now reads proposals as well as saved receipts, each labeled by source.
- Final hybrid run 06 passed in 47.726 s, with 223 native PCM chunks, actual Vosk and both writes to BK-0001. Its correction was not during recent native voice; native run 03 remains separate overlap evidence. The script checks controlled speech instructions, not browser playback.
- Separate browser validation against the live H100 gateway used diagnostic text and confirmed a stale query was ignored, Friday 16:30 booking and Thursday 14:00 rescheduling retained BK-0001. Device speech callbacks completed for both receipts and the rescheduling proposal. Confirmation intentionally interrupted the preceding proposal readback. No microphone was enabled, and no hardware audibility claim is made. AX traces and screenshots are in the workspace samples directory.
- Full local regression at the recorded snapshot: 542 passed, 1 optional skip, 22 subtests. Final new-module check: 26 passed. Final packaging/distribution check: 12 passed. An earlier packaging retry caught a temporary missing concurrent DuplexLab report; it passed after that work completed without editing its files. Syntax and Ruff F checks passed. Workspace status, EVOLUTION and both language guides are synchronized.

- Native-injection optimization subsequently deployed guided PersonaPlex proposals/receipts inside one WebSocket. Current architecture, retained failures and waveform acceptance supersede the original reconnect/device-first implementation above; see experiments/scheduling/native-injection/RESULTS.md. Two full voice sessions plus two new time values passed 12 key PCM checks, without fine-tuning.


## 2026-10-20 clean-prefix latency completion

Fixed voice/first-silence/18-token persona snapshot restores KV/offset/provided in place; latest tool text only. Full controlled native wording is now the default after guided-waveform failures; speech alias BK number1 is shared by context/opening, displayed BK-0001 remains unchanged. Runs33/34 +35 passed12 key waveform checks across4 date/time values, including speaking-time correction, same-ID reschedule and prefill cancellation. Update→gateway voiced PCM median3338.1645→1551.531ms (53.52% lower); excludes observer/query/browser/hardware/cold start. Main8769 diagnostic audio check36 passed. Gateway8768/backend8998 remain; old8999/candidate8767/forward8766 stopped. Nine deployed source hashes match; 83 scheduling tests and12 package/distribution checks passed; full618 passed,1 skip,22subtests,3 concurrent DuplexLab warnings. Human microphone/echo/listening remain pending. See [latency report](../../../experiments/scheduling/native-injection/LATENCY_RESULTS.md). No fine-tuning, apex, subagents, commit or push.
