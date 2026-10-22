# Experiment decisions / 实验决策索引

This index separates the interactive product prototype from offline ASR/data research. Protocols and detailed reports retain original failures and uncertainty. Cohorts differ between rows; do not compare MER across studies as a leaderboard.

## Interactive scheduling

| Record | Question | Finding | Decision |
|---|---|---|---|
| [Initial demo](../experiments/scheduling/RESULTS.md) | Can listening, tool state and corrected bookings form a loop? | Synthetic flows and versioned simulated state; original device-readback fallback | Historical baseline |
| [Native injection](../experiments/scheduling/native-injection/RESULTS.md) | Can native voice reflect updated tool facts? | Same-WebSocket update; exact facts checked in actual waveforms | Native controlled readback retained |
| [Prefix cache](../experiments/scheduling/native-injection/LATENCY_RESULTS.md) | Reduce redundant context rebuilding |10 tool updates, median3.338→1.552s;12 key waveform checks | Selected prototype; real microphone/quality pending |
| [Forced prefill](../experiments/scheduling/native-injection/PREFILL_RESULTS.md) | Remove redundant prediction work | Median0.898s;11/12 strict waveform checks | Rejected date failure |
| [Date pacing and actual entry](../experiments/scheduling/date-clarity-2026-10-21/RESULTS.md) | Improve clarity while retaining speed | Median0.868s;20 core facts, but human-confirmed actual-entry failure | Rejected; diagnostic replay does not erase failure |

Timing: tool-readback event → first same-epoch gateway PCM with RMS>.008. Excludes recognition, tool query, browser/hardware and strict cold start. Simulated calendar fixed at2026-10-20. No native function-calling or general task-success claim.

## DuplexLab: matched-budget ASR studies

| Report | Cohort and control | Principal result | Decision |
|---|---|---|---|
| [Mac CPU](../experiments/duplexlab/reports/2026-10-20-mac-cpu/RESULTS.md) |20 unique development clips; language diagnostics | One English clip drives a large small-group difference | Pilot only; no training/streaming claim |
| [Initial SFT](../experiments/duplexlab/reports/2026-10-20-continuation/RESULTS.md) | Base/random/selected;500 training clips, one seed;test100 | MER15.31/14.26/14.16%;selected only1 edit better than random | Expand paired uncertainty, not a selection success |
| [Three seeds](../experiments/duplexlab/reports/2026-10-20-multiseed/RESULTS.md) | Random/selected ×42/43/44;newtest300/3IDs | Mean10.41/10.90%;delta+0.4952pp, interval crosses0 | Selection advantage unsupported |
| [Coverage](../experiments/duplexlab/reports/2026-10-20-coverage/RESULTS.md) | Matched coverage;newtest300/3IDs |8.8984/8.6209%;interval[-0.5559,0.0000]pp | No reliable overall advantage; filler subgroup regresses |
| [Disfluency](../experiments/duplexlab/reports/2026-10-20-disfluency/RESULTS.md) | Frequency/repetition/switch-span;newtest320/4IDs |12.8807/12.6199%;interval[-0.9295,+0.2436]pp | Retention not established; total deletions increase |
| [Deletion/weighting](../experiments/duplexlab/reports/2026-10-20-deletion/RESULTS.md) | Random/selected/weighted ×3;newtest280/7IDs |10.7691/10.3624/10.9632%;same-data weighting worse in all seeds | Do not adopt weighting objective |
| [Normalization control](../experiments/duplexlab/reports/2026-10-21-normalization/RESULTS.md) | Random/native/uniform/2x-weighted ×3;newtest362/10IDs |9.7315/9.8353/9.8353/10.1329%;native/uniform match | Overall gains unsupported; freeze padding and ragged-batch contracts next |

Latest weighted-minus-uniform interval[-0.0556,+0.6673]pp and selected-minus-random[-0.3246,+0.5328]pp include0. All latest models use500 clips/63steps; train1015/dev52 fixed. IDs are isolated within the current split but have project history. Observed tests must not become later unseen tests. MER and deletion counts are interpreted together; human listening pending.

## Existing workflows

- [Glossary200](../experiments/glossary/ASCEND_200_RESULTS.md):200 recordings ×3 conditions; no demonstrated repair of original term misses. [Frozen protocol](../experiments/glossary/protocols/ascend-val200-train-vocab30-v2/README.md).
- [Remaining725 protocol](../experiments/glossary/ASCEND_REMAINING_EXECUTION.md): retained historical execution record; publication does not launch, rerun or newly certify this job.
- [Two-speaker workflow](TWO_SPEAKER_GUIDE.md): independent separation/ASR/notes pipeline and deployment evidence.

## Record format

Each new experiment needs a predeclared question, dataset/model/code revisions, frozen comparison and budget, failure accounting, uncertainty, adoption decision and verification. Public summaries do not redistribute raw speech, complete predictions, checkpoints or live credentials.
