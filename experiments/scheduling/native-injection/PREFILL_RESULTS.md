# Subsecond tool-prefill experiment — not adopted

2026-10-20. **The candidate met the latency target but failed an additional waveform date guard. The main Demo remains on the previously verified 1.552 s version.** [Measurements](PREFILL_SUMMARY.json), [plan and hypotheses](PREFILL_PLAN.md), [candidate manifest](PREFILL_BACKEND.json), [current deployed version](LATENCY_RESULTS.md).

## Result

Two complete paced synthetic-speech flows (47/48) passed: Thursday 15:00 → interrupt during native voice and change to Friday 16:30 → confirm → move the same booking to Thursday 14:00 → confirm. Each used one model handshake and zero device-voice fallback. Eight key waveforms passed explicit calendar/time/reference checks. The first proposal is intentionally interrupted and excluded from complete-readback counts. The saved/displayed reference stays BK-0001.

Additional direct simulated-tool facts (49; paced zero PCM, no user speech) passed **3/4** waveform checks. Tuesday 11:00 proposal text says October 20th, but independent Qwen ASR decoded the actual waveform as **October 28th**. The Tuesday receipt and Wednesday 09:30 proposal/receipt passed. This is an unresolved waveform/ASR discrepancy, not a claim based on human listening. Superseded epoch 2 had no applied ACK or epoch-2 readback; applied epochs were 3/4/5/6/7. Cancellation passed.

Final candidate total: **11/12 key waveform checks** across four unique date/time values. The validator was unchanged. A correct text stream, fast first sound and two successful flows do not override the failed date guard; no promotion or main-entry candidate acceptance was attempted.

## Timing

Start: gateway `native_readback`. End: first same-epoch gateway PCM with RMS > .008 (`native_voice_started`). Recognition, simulated-tool wait, browser/hardware and strict cold start are excluded; model weights were loaded. Full flows55.501/53.588 s include paced inputs and independent CPU ASR, not per-turn latency.

| Version | Updates | Minimum / median / maximum, ms | Disposition |
| --- | ---: | --- | --- |
| Uncached reset, 08/10 | 10 | 3232.057 / 3338.1645 / 3456.979 | Historical |
| Voice/persona cache, 33/34 | 10 | 1474.007 / 1551.531 / 1734.432 | **Still deployed** |
| Skip discarded predictions, 38 | 5 | 934.672 / 1007.573 / 1090.031 | One flow passed |
| Header cache + fixed seed, 43/44 | 10 | 805.395 / 895.936 / 960.721 | Later failed 45 |
| Header cache + preserved RNG, 47/48 | 10 | **827.856 / 898.117 / 975.551** | Later failed 49 |

All ten final timing observations are below one second; median is **42.11% lower** than 33/34. Tool-text prefill host-stage median518.220 ms (508.833–603.239), versus about1.17–1.33 s in run33. Six trailing silence frames remain; their median120.255 ms. Prefix restore host-stage median1.223 ms. These stage times are host wall time without added CUDA synchronization, not independent GPU kernel timings.

Small development comparison, without paired cross-version RNG/audio inputs, a frozen benchmark, p95 or general speech guarantee. A new WebSocket still performs initial voice/system setup. First-sound measurements here are existing-connection tool updates.

## Candidate implementation

- Non-emitting tool-text prefill retains upstream `prepare_step_input`, main-transformer KV updates, provided-mask cleanup and offsets. Skip discarded text/depth samples only when every generated target is already supplied. Partial targets and requested logits/loss use the upstream processor. Live audio stays on the original step method.
- Cache the connection-local voice, first silence and token-identical **28-token persona/tool header**, versus18 persona tokens in the deployed version. The extra ten tokens contain only the fixed introduction; no appointment, date/time, reference or user facts. Restore tensors in place and retain CUDA graph addresses. Free conversation history resets; versioned controller booking state survives.
- Measure actual upstream CUDA RNG offset increments during clean-header construction. Require a constant stride, restore construction RNG state, then advance by the cached header's draws and each skipped fully forced step. The pinned runtime measured **68**, not a hardcoded constant. Final candidate uses the original numeric alias spelling and **no fixed seed**.
- Diagnostic46 compared **100 exposed streaming tensors, 65 scalars, complete CUDA RNG state and four subsequent native audio-token steps: all exact**. Snapshots/synchronization are excluded from timing. This is a same-start computational parity check, not a universal waveform-quality claim.

All speech audio remains native PersonaPlex; complete authoritative tool wording is conditioned in text. No model weights or LoRA training. This is not native function calling. Spoken B K number one maps explicitly to displayed BK-0001; unmarked BK1 and contradictory/extra digits still fail. Device voice remains a labeled failure fallback.

Independent waveform validator: Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, CPU FP32, four threads, English, no expected facts supplied. Human microphone, headphones/listening, echo/AEC and hardware first sound remain pending.

## Retained intermediate failures

| Run | Result |
| --- | --- |
| 37 | Streaming-state parity passed; reference ASR BK1 Number One failed; diagnostic excluded from timing |
| 38 | Four key waveforms passed; median1.008 s; one intermediate flow |
| 39 | Subsecond timing, but final reference BK1 Number One/BK1001 failed |
| 40 | Spelling counter one in context/text passed one flow |
| 41 | Repetition lost the spoken hour four in4:30 PM; failed |
| 42 | Planned probe not run because41 failed |
| 43/44 | Fixed seed42 passed two flows; no per-case seed search |
| 45 | Fixed-seed extra facts2/4; Tuesday weekday/reference failed; seed/spelling changes rejected |
| 46 | Preserved-RNG state and next-token parity passed; not timing evidence |
| 47/48 | Original spelling, no fixed seed; both full flows passed |
| 49 | Additional facts3/4; Tuesday proposal20th→ASR28th; candidate not adopted |

Earlier11–36 evidence stays in [the cache experiment](LATENCY_RESULTS.md). Local ignored `runs/scheduling/native-injection/evidence/` retains events, all failed/accepted WAVs, scripts, hashes and the exact13-file candidate source snapshot. The independent ASR discrepancy remains open; do not fix it by relaxing date matching.

## Reproduce explicitly

The default preparation variant remains the verified cached backend. Experimental prefill is opt-in:

```bash
.venv/bin/python scripts/prepare_personaplex_native.py \
  upstream/personaplex/moshi/moshi/server.py --variant prefill \
  --output upstream/personaplex/moshi/moshi/server_native_rng_prefill.py
```

`PREFILL_BACKEND.json` and `server-prefill.patch` are hash-checked; upstream and unrelated candidates are preserved. The CUDA-only candidate uses `speaker_lab.scheduling.prefill` from this source package. Run on an isolated scratch backend/port with one client, retaining the selected main service. `PERSONAPLEX_VERIFY_PREFILL=1` checks the first tool prefill per process, including RNG and next-token parity; exclude that diagnostic from timing.

Temporary candidate backend/gateway/tunnel were retired. The previous8998/8768 backend/gateway and local8769 entry are retained; verify current ports in the environment artifact. No subagents, apex access, training, commit or push.

Verification:103 focused scheduling/distribution tests;633 full-suite passed,2 skipped,22 subtests,3 existing concurrent DuplexLab NumPy warnings; Ruff F clean. Both backend preparation variants hash-verified.13 candidate source files frozen and10 remote disk source files matched.16 related Markdown documents,277 local links,0 missing. [Latest environment](PREFILL_ENVIRONMENT.json).
