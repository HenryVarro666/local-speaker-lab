# Speech Data Curation Lab

English · [简体中文](README.zh-CN.md)

Speech Data Curation Lab (formerly displayed as DuplexLab) extends the existing `speaker_lab` package with an offline loop: freeze sources and hashes, audit raw audio, detect exact duplicates, run ASR in an isolated process, score errors, replay audio, and import versioned listening reviews. The existing two-speaker and glossary workflows retain their original interfaces.

The implementation remains under `speaker_lab.duplex` and `experiments/duplexlab/` in the
[existing Local Speaker Lab repository](https://github.com/HenryVarro666/local-speaker-lab).
The public export contains reviewed source and aggregate evidence; private audio, weights
and run state remain outside it. Evidence reviewed on 2026-10-27; no new training or model call.

## Latest: normalization and token weighting

Twelve fresh base-model SFTs completed: Random native CE / selected native CE / identical selected data with uniform normalized CE / identical selected data with 2x token-weighted CE, 500 clips and 63 steps per model, seeds42/43/44. On 362 new recordings from 10 current-role-separated speaker IDs, mean MER is **9.7315% / 9.8353% / 9.8353% / 10.1329%**; three-repeat deletions are **221 / 237 / 237 / 271**. All three paired native/uniform learned-weight hashes, 1242 batch1 processor input pairs and final transcriptions match exactly. Saved tokenizer left/right padding differs and is documented separately; equivalence does not cover ragged batches. Weighted minus uniform is **+0.2976pp**, paired seed/whole-speaker interval **[-0.0556,+0.6673]pp**; selected minus Random is **+0.1038pp**, interval **[-0.3246,+0.5328]pp**. The interval includes zero; a reliable overall weighting benefit is not established. Train 1015 / dev52 and selection are inherited unchanged; all 12 dev reloads precede test. IDs have historical project exposure; source references remain and human listening is pending. [Results](reports/2026-10-21-normalization/RESULTS.md) · [Diagnosis](reports/2026-10-21-normalization/REGRESSION_ANALYSIS.md) · [Reproduction](reports/2026-10-21-normalization/REPRODUCE.md).

`prepare_duplexlab_normalization.py --root . --shards /path/to/pinned/ASCEND/main` → input/loss preflight → `run_duplexlab_normalization.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_normalization.py` → `summarize_duplexlab_normalization.py` → `report_duplexlab_normalization.py`.

## Previous: deletion retention and objective ablation

The deletion-retention ablation completed nine fresh base-model SFTs (Random CE / selected CE / identical selected data with retention CE; each500 clips,63 steps,seeds42/43/44). On280 new recordings from seven current-role-separated speaker IDs, mean MER is **10.7691% / 10.3624% / 10.9632%**. Joint minus Random is **+0.1941pp**, paired seed/whole-speaker interval **[-0.7538,+1.1337]pp**; total deletions over three repeats are **284 / 270 / 295**. Deletion and overall errors must be assessed together; fewer D alone does not establish restored content. Train1015 and upfront dev40+12=52 are fixed; all nine dev reloads precede test. IDs/file/PCM exclude earlier recordings, but these seven IDs have historical project exposure. Original references/strict scores remain; human listening pending. The interval includes zero; reliable overall superiority is not established. On identical selected data, retention CE has more total edits in every seed (+0.6009pp mean MER); this objective is not supported by the pilot. Selected CE increases filler-token deletions25→35. [Results](reports/2026-10-20-deletion/RESULTS.md) · [Diagnosis](reports/2026-10-20-deletion/REGRESSION_ANALYSIS.md) · [Reproduction](reports/2026-10-20-deletion/REPRODUCE.md).

`analyze_duplexlab_deletions.py` → `prepare_duplexlab_deletion.py --root . --shards /path/to/pinned/ASCEND/main` → input/loss preflight → `run_duplexlab_deletion.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_deletion.py` → `summarize_duplexlab_deletion.py` → `report_duplexlab_deletion.py`.

## Previous: filler frequency, repetition and switch spans

The frequency/repetition/switch-span study completed six fresh base-model SFTs (500 clips/63 steps each; seeds42/43/44). On320 new clips from four speakers disjoint from current train/dev, Random/Disfluency mean MER is **12.8807% / 12.6199%**, delta **-0.2608pp**, paired seed/speaker interval **[-0.9295,+0.2436]pp**. The interval includes zero; a reliable advantage is not established. Filler-present64 clips average19.6048%→18.8248%. Across the three repeats, deletions are296→318, filler-token deletions26→25; disfluency retention is not established as repaired. Existing coverage and new frequency/repetition totals/switch-span marginals match; audio budgets differ by0.9643%. Dev40 remains unchanged;12 extra dev challenges address missing coverage, frozen before new test but evaluated after the main pipeline, without changing training eval_file. Test IDs13/14/18/24 have prior confirmation/test exposure; these are not project-wide never-seen speakers. [Results](reports/2026-10-20-disfluency/RESULTS.md) · [Prior regressions](reports/2026-10-20-disfluency/REGRESSION_ANALYSIS.md) · [Reproduction](reports/2026-10-20-disfluency/REPRODUCE.md).

`analyze_duplexlab_disfluency.py` → `prepare_duplexlab_disfluency.py --root . --shards /path/to/pinned/ASCEND/main` → `run_duplexlab_disfluency.py --root . --model /path/to/pinned/snapshot` → `run_duplexlab_auxdev.py --root . --model /path/to/pinned/snapshot` → `verify_duplexlab_disfluency.py` / `verify_duplexlab_auxdev.py` → `summarize_duplexlab_disfluency.py` → `report_duplexlab_disfluency.py --root .`.

## Previous: coverage constraints and three fresh training seeds

The coverage study completed six fresh base-model SFTs (seeds42/43/44;500 clips and63 steps each). On300 new recordings from three speakers disjoint from the current train/dev, Random/Coverage mean MER is **8.8984% / 8.6209%**, delta **−0.2774pp**; the paired seed/speaker interval **[−0.5559,0.0000]pp** still includes zero. All three overall point estimates improve, but filler-present91 clips worsen from10.4099% to10.5731%. Speaker/difficulty/filler/language-duration marginals match exactly; audio budgets differ by0.954%. Dev40 was retained and completed before test. Test speakers10/16/26 appeared in the prior training pool: current roles are speaker-disjoint, all six models restart from base, and this is not a project-wide never-seen-speaker claim. [Results](reports/2026-10-20-coverage/RESULTS.md) · [Design](reports/2026-10-20-coverage/DESIGN.md) · [Protocol](COVERAGE_PROTOCOL.md).

## Previous: unconstrained reference-agreement study

Three paired seeds42/43/44 now evaluate the same frozen300 clips from three previously unused speaker IDs. Random/Filtered mean MER is **10.41% / 10.90%**; Filtered minus Random is **+0.4952 percentage points**, with a paired seed/speaker interval **[−0.1378,+1.1198]** crossing zero. This fixed reference-agreement selection shows no reliable advantage. The prior100 clips/two speakers remain secondary (400 clips/five IDs combined, not400 entirely new clips). A pre-outcome209-clip audit-pass sensitivity also has higher filtered mean MER (+0.4522pp), with an interval crossing zero. [Full results](reports/2026-10-20-multiseed/RESULTS.md) · [12-case analysis](reports/2026-10-20-multiseed/REGRESSION_ANALYSIS.md) · [Protocol](MULTISEED_PROTOCOL.md).

Seed42 reuses the original checkpoints; four new43/44 SFTs each complete63 steps and real reloads. Training selections remain fixed (500 per arm); these seeds do not measure selection-resampling variance. Only2 of the12 prior Chinese/mixed regression cases remain worse in all3 seeds. All source references stay unchanged and human listening remains pending.

## Earlier pilots

The initial Mac CPU slice contains 20 unique real recordings (74.4204 seconds). Automatic-language MER was 13.99% overall and 10.71% on 10 mixed-language clips. On the same five English clips, the post-hoc known-English diagnostic changed WER from 48.48% to 18.18%; all of that change came from one clip. Each condition was repeated once with identical predictions. These are development results, with no training or population-level claim. [Initial results](reports/2026-10-20-mac-cpu/RESULTS.md).

The continuation freezes **40 new English clips from two new speakers** before inference. WER is **20.92% with automatic language versus 20.50% with known English**, only one edit fewer; the large earlier difference did not recur. The training candidate pool has 1,465 clips, with separate 40-clip dev and 100-clip test sets using official speaker IDs. Both 500-clip SFT arms completed 63 optimizer steps and real checkpoint reloads on H100 GPU1. On the same frozen 100-clip test, MER is **15.31% / 14.26% / 14.16%** for Base / Random / Filtered. Filtering beats random by only one edit; paired intervals cross zero, so this pilot provides no clear evidence of superiority. [Preregistered protocol](CONTINUATION_PROTOCOL.md) · [Continuation results](reports/2026-10-20-continuation/RESULTS.md).

The official Parquet does contain speaker/session fields; the original importer omitted them. The old 20 clips came from speakers 1/2. Original frozen files and scores remain intact, with a correction in the reports. Model pretraining exposure to public ASCEND audio is unknown.

## Local commands

Run from the repository root:

```bash
.venv-dev/bin/python -m speaker_lab.duplex --help
.venv-dev/bin/python -m speaker_lab.duplex score --run runs/duplexlab/confirm40-cpu-auto-001
.venv-dev/bin/python -m speaker_lab.duplex report --run runs/duplexlab/confirm40-cpu-auto-001 --output runs/duplexlab/confirm40-cpu-auto-001/replay/index.html
.venv-dev/bin/python -m speaker_lab.duplex review --run runs/duplexlab/confirm40-cpu-auto-001 --output runs/duplexlab/confirm40-cpu-auto-001/replay
```

`freeze` creates a new content-verified protocol. `audit` does not load a model. `run` requires a local pinned model and explicit language/disabled aligner settings. Completed jobs are not repeated with `--resume`; changed model, config or ASR source rejects recovery. Failed attempts retain their logs. Stale `running` jobs are conservatively refused; automatic recovery is not implemented.

The review page starts pending. Actual listening, reviewer identity and corrected text are required for approval; browser playback checks do not count as human review. Import a downloaded review with:

```bash
.venv-dev/bin/python -m speaker_lab.duplex import-reviews --run runs/duplexlab/confirm40-cpu-auto-001 --file /absolute/path/to/reviews.json
```

Imports are source-bound, append-only and idempotent. Original references are retained. `report --public` emits aggregate text without source recordings, transcripts, hypotheses or local paths.

## Training entry points

`scripts/prepare_duplexlab_pilot.py` prepares the fixed source shard without ASR; it refuses an existing output. `scripts/run_duplexlab_continuation.py` executes explicit `pool`, `train` or `evaluate` stages in an isolated GPU experiment directory. Its training phase first runs a separate 8-train/4-dev smoke and real checkpoint reload, then trains random/filtered 500-clip arms from the same base. `scripts/duplexlab_train.py` defaults to validation only; actual launch requires `--execute`.

Training exports bind role, reference, audio and JSONL hashes; dev stays separate and test is never copied into a training export. The earlier unconstrained selector matches language/duration quotas and total audio time, with recorded within-stratum swaps if needed. It selects base-model/reference agreement, which may favor easy material rather than generally better data.

The new phases use `prepare_duplexlab_multiseed.py` and `run_duplexlab_multiseed.py --phase train|evaluate`. The former freezes speaker3/4/17 without model output; the latter supplements43/44 through a hash-bound explicit-seed adapter. `verify_duplexlab_multiseed.py` re-scores archived predictions without loading a model; `report_duplexlab_multiseed.py` renders public aggregates and private replay. See [reproduction and audit commands](reports/2026-10-20-multiseed/REPRODUCE.md).

## Evidence and remaining work

Model environments, audio, complete predictions and checkpoints remain local/ignored. Source packages explicitly include reviewed documentation, aggregate fingerprints and generated figures. The private 100-clip replay compares all three checkpoints, including regressions; it is generated under runs/duplexlab/h100-continuation-v1/replay/. ASCEND-derived material retains CC BY-SA 4.0; original code and model licenses remain separate. See [third-party notices](../../THIRD_PARTY_NOTICES.md).

Actual human listening, native streaming, expressive synthesis, microphone/voice latency and Avatar integration remain unverified in this branch. CPU subprocess durations are not first-text, TTS or voice-to-voice latency. [Design](../../docs/superpowers/specs/2026-10-20-duplexlab-design.md) · [Implementation status](../../docs/superpowers/plans/2026-10-20-duplexlab-implementation.md).


Coverage commands: `prepare_duplexlab_coverage.py`, `run_duplexlab_coverage.py --root . --model /path/to/pinned/snapshot`, `verify_duplexlab_coverage.py`, and `report_duplexlab_coverage.py --root .`. [Reproduction prerequisites and commands](reports/2026-10-20-coverage/REPRODUCE.md). Private test/dev playback is under `runs/duplexlab/coverage-v1/replay/`; human listening remains pending.
