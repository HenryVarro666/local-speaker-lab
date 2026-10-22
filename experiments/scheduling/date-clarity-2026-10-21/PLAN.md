# Date-word clarity experiment

Experiment date2026-10-21, America/Chicago. Freeze the simulated Demo calendar at2026-10-20 for regression; no real appointments. Main8998/8768 stays unchanged; scratch8999 uses GPU0 only. No subagents, training, apex access, commits or pushes.

Feedback loop: retain failing run49 full WAV (SHA fec45cb73aa330d6950881a7abd156122eda205c26b04ee25b6dbd6c597ddca8), decode without expected-fact prompts. Ranked hypotheses announced before tests: ASR/context ambiguity; insufficient date-word articulation time; cancellation-state interaction. One variable per comparison.

Frozen50: Vosk full clip says twentieth. Qwen full clip twice says28th; its first3seconds says twentieth. These are diagnostic disagreements, not a replacement acceptance gate or proof from human listening. Fix the diagnostic CLI to reject empty glob matches before loading optional ASR dependencies; the initial new filenames did not match its stage-* default.

Candidate rule: only ordinal date words after an English month and before at receive5 instead of3 frames. All token IDs and EPAD order are preserved; calendar months, weekday, booking letters, times (including thirty after at) and other words retain their previous timings. No per-case seed search or extra text. A date-pacing control is optional, bounded3–7; default library behavior stays3. The scratch backend defaults5. No changes to prefill/RNG preservation.

Predeclare A/B order: (seed42,pace3), (42,5), (123,5), (123,3), (2026,3), (2026,5). One fresh WebSocket per case; same initial prompt, no initial user PCM or free greeting, same Tuesday October20 11AM proposal; paced zero PCM after control. All six complete original waveforms get identical independent Qwen decoding and strict unchanged calendar/time guards. Vosk full-waveform readings also retained. These are simulated-tool probes, not user-speech flows.

A/B seed is a connection-initialization experiment control, never a per-tool production policy. If candidate passes all three declared cases, restart a fresh unseeded scratch process before acceptance, then run two complete original synthetic user-speech interruption/rescheduling flows and two repetitions of four extra facts plus cancel-during-prefill. Do not pool failed candidates into success statistics. Require complete native waveforms/date/time/reference, one handshake per user-speech flow, no device fallback, same-reference reschedule, superseded epoch no ACK/audio, and latency reported with exact endpoints. First interrupted proposal is excluded from complete-readback denominator. Do not promote based only on text, fewer ASR errors or latency.

Timing: gateway native_readback to first same-epoch PCM RMS>.008. Exclude observer/tool wait, browser/hardware and strict cold start. A/B direct-client timings are separate and not pooled with gateway measurements. Human listening/microphone/AEC remain pending.

## Dated amendments and additional diagnostics

After completing A/B52, both date3 and date5 passed2/3 (seed42 lost the weekday in the primary full-waveform ASR). Before running53, add weekday5 while preserving date5 and reuse all three declared seeds42/123/2026; no seed selection. This C arm passed3/3, then restart a fresh unseeded process before55 onward. This is an engineering development sequence, not a frozen benchmark.

After55–58, freeze the exact15-file source snapshot before prospective60: Monday October26 4PM and Thursday October22 9AM, proposal/receipt for each, no seed. After those four complete waveforms, run actual gateway entry61 using diagnostic text Book Tuesday at eleven am immediately after capabilities plus paced zero PCM. Entry61 failed; retain the candidate as rejected.

After independent Qwen/Whisper/Vosk and human failure at61, test decoder-history hypothesis62: predeclare seeds7/11/19/23, initial0.64s zero PCM, capture full generated codes and pre-Opus PCM. Offline decode the identical codes after reset versus priming with the existing6 silence frames. Never generate replacement codes for this comparison. Check reset replay PCM exactly matches live capture. No timing or human acceptance claims from instrumented runs.

Predeclare63 as four fresh unseeded actual gateway sessions, exact entry61 event order. Capture codes/pre-Opus and gateway PCM, compare reset/6-frame replay without changing text/pacing, seeds or the strict fact guard. Complete all four, keep every failure, and do not use these diagnostic repetitions to erase61 or promote the backend. No more parameter search in this round.
