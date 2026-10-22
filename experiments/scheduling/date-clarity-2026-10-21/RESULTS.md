# Date clarity and sub-second prefill follow-up

The calendar-pacing candidate is **rejected for the main entry**. It reached a median **867.684ms** from gateway tool-readback event to first same-epoch native PCM with RMS>.008, and passed20 core key waveform checks, but actual gateway entry61 failed. The user heard **“28 11点，别的没听清”** where the expected date/time was **Tuesday October20 11AM**. Main8998/8768 and local8769 retain the previous accepted1.552s version; no model fine-tuning.

Experiment started2026-10-21 and continued through2026-10-22, America/Chicago. The simulated calendar stays fixed at2026-10-20. See [plan and amendments](PLAN.md), [machine-readable results](SUMMARY.json), [final environment](ENVIRONMENT.json), [candidate metadata](../native-injection/CALENDAR_BACKEND.json) and [previous rejected prefill experiment](../native-injection/PREFILL_RESULTS.md).

## Changes and controls

The opt-in calendar backend retains the28-token clean persona/tool-header cache and fully forced prefill optimization with measured CUDA RNG advancement. It increases English weekday pacing3→5frames and date ordinals after a month/before “at”3→5frames; ordinary words, time words and the existing B/K/zero pacing stay unchanged. The complete authoritative tool wording conditions PersonaPlex audio; this is not native function calling. There are no changed weights or per-tool seed resets. Default preparation remains `cached`, not `calendar`.

Library date/weekday pacing controls are optional integers3–7 (bool rejected). Their default remains3. Initial-connection seed is optional uint32 and omitted by default; it exists only for declared controlled comparisons. The diagnostic ASR CLI now rejects an empty glob before loading optional dependencies and records relative filenames. No domain, business-controller or page changes.

Candidate source SHA256: `947acc18c7287b7e5e7323b810f2463193168fd7daf0d4c993080e75f8eb381d`. The15-file source snapshot predates prospective60; full hashes are in SUMMARY. Later capture instrumentation is an ignored diagnostic module, not the packaged backend.

## Results

| Cohort | Input and comparison | Result | Interpretation |
|---|---|---|---|
| Frozen50 | Original49 waveform, no regeneration | Qwen full twice28th; Qwen first3s crop twentieth; Vosk/Whisper full twentieth | ASR/context disagreement; original49 stays rejected |
| A/B52 | Seeds42/123/2026, date3 vs5, weekday3 | Both2/3 strict facts | Extra ordinal time alone did not fix weekday loss |
| C53 | Same three seeds, date5+weekday5 |3/3 strict facts | Development evidence only; restart unseeded before acceptance |
| Parity51/54 | Reference vs optimized prefill |100 tensors,65 scalars, RNG and next4 audio steps exact | Computational parity does not establish waveform correctness |
| Flows55/56 | Two paced synthetic-user flows with speaking-time correction and same-reference reschedule |8/8 key waveforms; one handshake and zero device fallbacks each | BK-0001 preserved; not human microphone input |
| Extra57/58 | Two repetitions of Tuesday11AM/Wednesday9:30AM proposal/receipt plus prefill cancellation |8/8 facts; superseded epoch2 no ACK/audio | Simulated facts and zero PCM, not extra spoken tasks |
| Prospective60 | Source frozen before new MondayOct26 4PM/ThursdayOct22 9AM proposal/receipt |4/4 facts | Single once-only probe, no tuning afterward |
| Actual entry61 | Diagnostic text on capabilities, continuous zero PCM | **Failed** independent ASR and human date readback | Candidate remains rejected |

The core total is20/20 factual waveforms across five runs and six unique slots, **not20 independent user tasks**. First interrupted/superseded proposals are excluded from complete-readback counts. Only BK-0001 was validated in generated speech; no general arbitrary-reference reliability claim.

### Timing

Two full flow durations were58.693/58.634s, including independent QA work; these are not per-turn latency. Ten update→first-voiced-gateway-PCM values in ms:

`812.136, 864.663, 924.830, 839.701, 919.133, 870.705, 778.270, 942.095, 844.192, 925.681`

Range778.270–942.095ms; median867.684ms,44.08% below the previous deployed ten-update median1551.531ms. Weights were warm. This excludes observer/transcription, tool-query wait, browser scheduling, network-to-hardware playback and strict cold start. The0.898→0.868s difference between unpaired candidate runs is not a causal estimate of the pacing change. No p95 claim.

### Entry61 and human evidence

Model text exactly matched “Tuesday October twentieth at eleven A M is available. Say yes confirm to save.”, while full waveform checks read:

- Qwen: “Is twenty to twentieth at eleven a.m. is available. Say yes, confirm to book to save.”
- Whisper: “is 20th at11a.m. is available. Say yes, confirm to book to save.”
- Vosk: “a straight to twentieth at eleven am is available say yes confirmed a book to save”.
- Human user: **“28 11点，别的没听清”**.

Expected weekday/month/date/time are Tuesday/October/20/11AM. The user heard day28 and hour11; other facts were unclear. This is a failed clip, not merely an ASR ambiguity. Original full WAV SHA256 `cb73947699ef91b8c0fe591c69e34907180fddafe49054210afd3caa167a7fc3`; no crop, replacement or retry is substituted for it. Entry61 native-readback→voiced PCM was825.961ms, separately reported and excluded from the ten-flow timing set.

All ASR systems were independently conditioned on the audio/language only, without expected facts, grammar or hotwords. Qwen0.6B revision`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, Whisper-large-v3-turbo revision`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, CPU FP32; Vosk small-en-us0.15. The human reviewed only61, not the whole cohort.

## Decoder diagnostic

62 captures four predeclared seeds7/11/19/23. For every case, replaying the same agent codes from reset reproduced live pre-Opus PCM with **maximum absolute error0**. Reset/6-frame-history replay both passed4/4 strict full-waveform Qwen checks; the original client Opus outputs also passed4/4. This did not reproduce the entry fault and provides no evidence to adopt decoder priming.

63 completed all four fresh unseeded actual-gateway sessions with the entry61 event sequence. Qwen strict weekday/month/date/time checks passed4/4 on received gateway PCM,4/4 on reset replay and4/4 on6-frame-history replay; all reset baseline PCM comparisons were exact. There was no transport-only failure in this cohort and no demonstrated priming benefit. Both62 and63 failed to reproduce61, so the root cause remains unresolved; **the known human-confirmed61 failure is retained and the candidate stays rejected**. Instrumented timing is excluded from performance claims.

These are8 generated token sequences with alternate decoder outputs, not24 independent successes or newly accepted user tasks. Captures, maps, complete independent ASR, strict guard results and debug-source hash are retained. No additional parameter grid is run in this round.

## Reproduction and decision

The reviewed extension is reproducible explicitly, while the default stays the previously accepted backend:

```bash
python scripts/prepare_personaplex_native.py \
  upstream/personaplex/moshi/moshi/server.py \
  --variant calendar \
  --output upstream/personaplex/moshi/moshi/server_calendar_prefill.py
```

This verifies pinned upstream and candidate hashes, preserves the original file and refuses to overwrite a different output. Use an isolated scratch port/GPU0 only if reproducing; **this candidate is not selected for the Demo**. Raw WAVs/tokens/events, initial code snapshots, the frozen49 failure and all failed candidates remain under ignored `runs/scheduling/date-clarity-2026-10-21/` (49 remains in the previous native-injection evidence). Source releases contain only explicitly listed code/reports/patches, never recordings, credentials, models or runtime logs.

Next work must target startup-word alignment with a frozen actual-entry regression before retiming the model further. Preserve interruption, cancellation, all exact tool facts and no-greeting behavior; no generic filler may be counted as successful fact readback. Human microphone/headphone/AEC, broad natural-language coverage, subjective quality and runtime speech verification remain unvalidated. No apex access, other-service changes, subagents, staging, commit or push.

## Final verification and environment

Focused scheduling/package checks114passed; full suite650passed,2skipped and22subtests (17.09s), with3 existing unrelated DuplexLab NumPy warnings. The first full run hit16 known bootstrap failures with macOS BSD `realpath`; rerunning unchanged code with the existing GNU coreutils directory prepended to PATH passed. Both logs are retained. Ruff F, whitespace,15 frozen-source hashes and real pinned-upstream calendar-patch reproduction passed. Nine related Markdown files/235 local links had zero missing targets.

Final remote snapshot2026-10-22 05:19:49UTC: main backend PID2154793/8998 and gateway PID2184143/8768 retained;8999/8767 and local8766 closed. Local8769 gateway health passed (no active model WebSocket, so `model_ready=false` is normal). Main backend SHA remains`331578aa28761cf34c966d1edc71fd5349d94762d23536d758313846a6739238`. GPU0 returned22697MiB/0%; GPU1 observed14MiB/0% at final snapshot, versus an unrelated active task earlier; no operations against GPU1. Disk has backward-compatible experimental controls; unchanged running main processes retain their previously loaded modules. See ENVIRONMENT for exact runtime versions. This is environment health and source verification, not a fresh native-speech acceptance or hardware listening test.
