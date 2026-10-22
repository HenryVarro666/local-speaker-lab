# PersonaPlex clean-prefix latency

2026-10-20, America/Chicago. Deployed and verified. The selected Demo uses complete controlled tool wording with native PersonaPlex audio. No fine-tuning or changes to model weights.

Each tool update restores a connection-local snapshot of the fixed voice prompt, first silence and **18 token-identical persona tokens**. The snapshot precedes all booking/user facts. It copies tensors into existing storage, restores generator/KV/position offsets and provided masks, and preserves CUDA graph objects. Only the latest authoritative tool text is then prefilled. It does not preserve free conversation history. A mismatched tool persona replays the original prefix; an unmatched initial persona uses the voice-only cache. Old reset/read modes remain available.

Cancellation is checked during prefill and again inside the output lock. Superseded controls must not publish an acknowledgement or start a reply. Ogg stream replacement remains ordered behind the matching epoch acknowledgement.

The canonical written reference remains BK-0001. Native speech uses the explicit alias **B K number one**, deterministically derived from that saved reference. Tool context and the reply opening share one alias function; they never mix the alias with a competing zero-padded form in model context. BK number N maps to the canonical BK-{N:04d}; the validator requires the word number and exact counter. Literal BK1 is not accepted as BK-0001, and extra digits or contradictory references fail. The selected mode now conditions the entire tool sentence, including dates and times; all waveform generation remains native. The guided sampled-date alternative failed additional checks and is retained for experimentation. B/K use longer native text timing (pace 5); ordinary words use pace 3. Device voice remains the labeled failure fallback.

## Timing

Start: gateway `native_readback` event. End: first same-epoch gateway PCM with RMS > .008 (`native_voice_started`). Excludes observer/query waits, browser, hardware and strict cold start. Full script times include paced input and independent waveform ASR and are not response latency.

| Version | Updates | Minimum / median / maximum ms |
| --- | ---: | --- |
| Previous uncached reset, runs 08/10 | 10 | 3232.057 / 3338.1645 / 3456.979 |
| Voice-only cache, intermediate run 14 | 5 | 2122.041 / 2148.011 / 2332.910 |
| Final controlled read_cached, runs 33/34 | 10 | 1474.007 / 1551.531 / 1734.432 |

The intermediate counted-reference runs 17/18 had a 1570.8185 ms median, but later reference checks were ambiguous, so their wording was not selected. Selected median reduction is **53.52%** (about 1.787 s). The selected source and changed wording differ from the earlier native baseline; this is a whole-Demo development comparison. Small development samples, different reference wording, unpaired RNG; no p95, frozen benchmark or general speech-accuracy claim.

## Retained failures and superseded candidates

| Run | Observed issue / disposition |
| --- | --- |
| 11 uncached recheck | Receipt said relative tomorrow; explicit-date gate failed |
| 12 instrumented reset | Corrected proposal incomplete; explicit CUDA synchronization also increased measured delay; not a fair latency baseline |
| 13 voice-only cache | Reference waveform ASR did not match; not deployed |
| 14/15 | Full flows passed, but later repetition 16 failed reference gate |
| 16 | Independent ASR rendered too many leading zeros; actual listening remains unverified |
| 17/18 | Counted triple-zero wording passed two full flows; later novel probe 19 disagreed between Qwen and Vosk on reference normalization |
| 20/21 | Three zeros wording was normalized as 30; not accepted |
| 22 | Probe ran before backend readiness; connection failure, no model conclusion |
| 23/24 | Slower individual spelling passed novel checks and one full flow; repetition 25 again failed exact reference gate |
| 25 | Independent ASR rendered extra zeros on final receipt; not relabeled as correct |
| 26 | Alias opening was mixed with a zero-padded tool-context reference; waveform gate failed |
| 27/28/29 | Shared alias passed a minimal receipt and two full guided flows; novel probe 30 still failed |
| 30 | Garbled Tuesday proposal and duplicated Wednesday reference; Vosk and three identical-wave Qwen repeats corroborated the failures |
| 31 | Missing local script dependency, infrastructure failure only |
| 32 | Corrected fully-controlled native receipt probe passed Tuesday11AM/reference |

These changing protocols must not be pooled into a success rate. Logs, complete source variants and raw PCM remain in ignored `runs/scheduling/native-injection/` and the isolated H100 runtime. No leading zeros are silently inferred to turn failed literal references into successes.

## Acceptance and limits

Runs **33/34** passed the original complete speech flow: Thursday15 → speaking-time correction Friday16:30 → confirm → reschedule Thursday14 → confirm, same BK-0001. They took 55.214/55.034 s including independent ASR waits, with one handshake and zero device fallback each. Four key waveforms per flow passed. Direct tool-fact probe **35** (zero user PCM) passed both Tuesday11AM and Wednesday9:30AM proposals/receipts, plus cancellation during prefill: obsolete epoch2 had no acknowledgement/audio, valid epochs3/4/5/6/7 applied. Total **12 key waveform checks across four unique date/time values**. Main-entry diagnostic probe **36**, separately labeled text plus paced zero PCM, passed native cache/PCM protocol and independent Tuesday11AM waveform verification. Waveform acceptance checks explicit month/day, weekday, time, am/pm and the exact canonical reference or its declared spoken alias. Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, CPU FP32, English; ASR decoding runs off the client event loop. No expected booking value is supplied to the ASR model.

Human microphone/headset, echo, subjective quality and hardware latency remain pending. This is a single-session Demo with an external Vosk/tool controller, not native function calling. [Numeric evidence and raw-event hashes](LATENCY_SUMMARY.json), [backend metadata](BACKEND.json), [uncached snapshot](PRE_CACHE_SUMMARY.json) and [uncached environment](PRE_CACHE_ENVIRONMENT.json) are preserved. Nine runtime source hashes and the backend hash match. 83 scheduling tests, 618 full-suite tests (1 skip, 22 subtests; 3 existing concurrent DuplexLab NumPy warnings) and 12 distribution/package checks passed. Ruff F passed. Main gateway8768 and verified backend8998 remain loopback-only with local forward8769; old8999, candidate gateway8767 and forward8766 are stopped. Active module server_prefix_paced and prepared reproduction module server_native_cached have identical SHA256 331578aa…. GPU0 was 22697/95830 MiB idle after cleanup; GPU1 was not operated. Page code is unchanged; this round does not add browser/hardware listening evidence.

Final documentation check: 15 related Markdown files, 272 local links, zero missing targets. Final focused run: 95 passed (83 scheduling + 12 distribution/package). Twelve frozen source files still match byte-for-byte.
