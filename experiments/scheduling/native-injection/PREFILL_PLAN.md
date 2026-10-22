# Fully forced tool-prefill experiment

2026-10-20. Keep the selected 1.552 s backend 8998 and gateway 8768 while testing backend 8999 on GPU 0. No training, subagents, apex changes or Git mutations.

The remaining text prefill took 1.17–1.33 s in run 33. Ranked hypotheses: skip discarded depth/text samples for fully supplied targets; extend the fact-free static template cache; shorten trailing silence only after waveform validation. Change one variable at a time.

Candidate 1 keeps upstream prepare_step_input, main transformer KV update, provided-mask cleanup and offset increment. It omits sampling/depth only when every generated target mask is true. Partial targets/logit reporting use the original processor. This changes random-number consumption; it does not claim identical sampled future waveforms. Live audio generation stays on the original step method. Both six-frame silence sections remain.

Before timing, replay the same real-model tool prefill through the old and fast paths, restoring the same prefix, and compare every streaming tensor exactly (NaNs equal) plus all scalar offsets/masks. Synchronization and snapshots in this diagnostic are excluded from performance runs. Then measure gateway native_readback to same-epoch PCM RMS>.008 with paced synthetic audio, two complete scheduling flows and additional simulated-tool facts/cancellation. Independent Qwen ASR must confirm dates, times and explicit reference aliases from the actual waveform. Keep failed runs.

Subsecond is a target, not an acceptance result. Human microphone, browser/hardware first sound and subjective listening remain pending.

Candidate 1 (38) passed one complete flow at median 1007.573 ms. Candidate 2 added a token-identical 28-token role/tool header with no facts (39), achieving subsecond measurements but failing a receipt reference. Spelling the alias counter as English one instead of numeric 1 passed one flow (40), then repetition 41 failed the spoken hour. These remain failures, without relaxing the waveform checks.

Additional falsifiable hypothesis, shared before testing: discarded-sample removal changes RNG consumption. Fix tool-speech CUDA seed to the preselected value 42 after current-epoch validation and before the readback ACK, retaining all context, text pacing and waveform gates. Validate all dates/references and repeat without per-case seed search (43/44/45). This is a reproducible Demo policy, not a speech-quality or universal correctness guarantee. Guided/free audio is not reseeded. Keep the original 1.552 s service until all candidate gates pass.

Fixed-seed 43/44 passed two flows but novel facts 45 failed Tuesday weekday/reference guards (2/4 passed). Reject the fixed-seed policy and restore original numeric alias spelling. Do not promote it.

Next hypothesis: preserve the original CUDA RNG state progression while skipping predictions. At each connection's clean template construction, measure the actual upstream sampling offset increment for every static header token; require a constant stride and restore the preconstruction RNG state. At a cached tool update, advance by the cached ten tokens' measured draws, then advance once per fully forced fast step. Partial steps run the original processor. No fixed seed or hardcoded stride. In diagnostic 46 compare streaming tensors/scalars, CUDA RNG state and four following native audio-token steps exactly; exclude diagnostic timing. Then run 47/48 full speech flows and 49 additional facts/cancellation before promotion.
