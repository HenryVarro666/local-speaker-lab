# Native tool-result injection experiment

2026-10-20, America/Chicago. User authorized improving the existing PersonaPlex Demo without fine-tuning. Only the isolated H100 environment and GPU 0; retain existing fallback and all failed evidence. No subagents, apex operations, commit or push.

## Reproduction and hypotheses

Direct backend probe first repeats the current reconnect strategy with the same voice and actual checkpoint, then records native text and PCM for the saved Friday 16:30 receipt. These are synthetic inputs, not microphone or browser hardware acceptance.

Ranked hypotheses:

1. Every update resets LM conversation state and replays voice/system prompts. Prediction: inserting current tool facts into the existing LM state will reduce repeated introductions and avoid the 4–6 s reset.
2. Replaying only the last short utterance loses preceding context. Prediction: retaining the full active session and avoiding acoustic replay will improve post-confirmation responses.
3. Playback epochs, suppression or delayed codec buffers hide valid speech. Prediction: direct backend PCM/text will distinguish generated speech from gateway/UI loss; a versioned control acknowledgement must gate new audio.
4. The model's free text generation does not reliably follow arbitrary dynamic facts. Prediction: overriding only the assistant text stream with validated tool text while leaving audio generation native will improve exact content. This must be labeled controlled native readback, not autonomous function calling.

## Sequential probes

- Reconnect baseline: initial greeting, then confirmed receipt prompt plus paced synthetic confirmation.
- In-place context prefill: current facts enter the existing cache, no voice-prompt replay or new greeting. Record update-to-first-voiced-PCM and generated content.
- If needed, native controlled speech: reuse upstream LMGen.step(text_token=...) with generated audio. Compare token pacing, retain unforced conversation before/after. Never call forced text itself proof that the waveform says it.

Development inputs cover Thursday 15:00, corrected Friday 16:30 and rescheduled Thursday 14:00; confirmation must preserve BK-0001. Add at least two other date/time values before selecting a candidate. A separate full flow must include speech-time interruption, newest-version cancellation and both receipts. Frozen benchmarking, human listening and microphone/AEC remain separate.

## Acceptance

- No greeting in tool readbacks; non-silent PCM after each update; day/time and saved reference supported by actual generated waveform ASR as well as visible source text.
- Stale control/output cannot overwrite or play as the current proposal; duplicate confirmation cannot repeat a write.
- Report forced/controlled text vs freely generated text, text availability vs native first PCM, and source/endpoint timing separately.
- Only switch the Demo default after actual model probes and the full simulated tool flow pass. Keep device voice on backend failure, with clear source/status events.

## Sources and evidence

Official source/model: https://github.com/NVIDIA/personaplex and https://huggingface.co/nvidia/personaplex-7b-v1. Exact installed revision stays in ../ENVIRONMENT.json. Native LMGen already accepts a text_token override; the official WebSocket currently accepts only user audio. Any added control messages are a local experimental extension.

Commands, manifests, WAVs, JSONL and failed attempts go under ignored runs/scheduling/native-injection locally and runs/native-injection on H100. Aggregate results and the patch are retained in this experiment folder.

## Engineering rulings (development, not acceptance)

- Baseline reconnect reproduced a full greeting instead of receipt facts. A first harness attempt had a wrong synthetic-file path; it is retained and excluded from model conclusions.
- Fixed-rate full text forcing with repeated EPAD 0 produced correct text tokens but incorrect waveform content. Both Vosk and independent Qwen3-ASR-0.6B CPU checks rejected it. That implementation was not selected; its source and patch remain in ignored evidence.
- In-place append initially hit new-stream codec races and nullable Linux decoder results. Serialize Opus reset with its control acknowledgement, tolerate None, and attribute disconnects to the current epoch. Wire tests reproduce these conditions.
- Corrected append-context probe 04 did naturally speak Friday 16:30 and BK-0001, verified from its waveform by both ASRs. However full-flow 01/02 restarted introductions. One-word steering in 03 avoided that introduction but retained the obsolete Thursday time. These runs failed acceptance and are not pooled as successes.
- Chosen next hypothesis: recondition LM/Mimi inside the existing WebSocket using cached voice embeddings plus only current tool facts. First-word steering in full run 04 still introduced the desk. Expand only the fixed reply opening, with actual PAD/EPAD word boundaries; dates, times and reference remain generated. No default deployment until the original full-flow and independent waveform checks pass.

## Final selection and acceptance

Default native uses in-connection model reconditioning plus a short opening and saved reference; date/time continuation and all speech are generated. Full controlled read mode remains an explicit alternative, device voice a failure fallback. Two full synthetic sessions (08/10) passed four actual-waveform checks each before confirming; additional Tuesday 11AM and Wednesday 9:30AM facts passed four waveform checks. Total 12 guided checks across four unique date/time values, one handshake and zero device fallbacks per full session; speaking-time correction in both. Controlled alternative 07 separately passed four waveform checks.

Important correction: guided metadata-guard run 06 omitted 2 in its text channel, but Qwen ASR heard two p.m. in its audio. Subsequent acceptance uses independent PCM ASR, not only the text stream. Failed run 06 stays incomplete; it is not relabeled a full success.

Local final source snapshot: 581 passed, 1 optional skip, 22 subtests; scheduling tests 52 passed. Eight deployed source hashes and the native backend hash match. The backend patch reconstructs exact reviewed bytes and preserves original source. Human microphone, echo/hardware and subjective voice quality remain pending. Browser lifecycle verification and final cleanup are recorded in SUMMARY/RESULTS when closing.
