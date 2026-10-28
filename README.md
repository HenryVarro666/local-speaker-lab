# Local Speaker Lab

**English** · [简体中文](README.zh-CN.md) · [MIT](LICENSE)

**Auditable speech experiments and an interruptible appointment-agent prototype.**

![Existing two-speaker workflow architecture](docs/figures/overview.svg)

## Problem and intended products

Speech systems can lose domain terms, conversational fillers or the latest correction while still producing plausible text. This lab makes those failures traceable to an audio input, a model/configuration and a fixed evaluation rule.

The main interactive showcase is a simulated appointment assistant that listens during speech, accepts a corrected slot and reschedules the same booking after explicit confirmation. Offline data/ASR studies ask how to improve content retention without selecting a flattering test set.

## Workflows and methods

| Workflow | Method and intended use | Current state |
|---|---|---|
| [Native scheduling agent](experiments/scheduling/README.md) | PersonaPlex continuous audio, separate business recognition, versioned simulated tools and controlled native readback | Cached baseline exercised; faster date-readback candidates rejected |
| [Speech Data Curation Lab](experiments/duplexlab/README.md) | Audio/source hashes, review, matched-budget selection, Qwen3-ASR SFT, grouped errors and paired uncertainty | Multiple three-seed pilots complete; no reliable selection/weighting advantage established |
| [Code-switching glossary](experiments/glossary/README.md) | No/correct/unrelated glossary on identical ASCEND clips with frozen train-derived vocabulary | 200 clips × 3 conditions complete; original term misses not repaired |
| [Two-speaker notes](docs/TWO_SPEAKER_GUIDE.md) | Separation → speaker matching → ASR/alignment → Ollama notes | Existing Ubuntu deployment route; independent of scheduling and DuplexLab claims |

The poster describes the existing two-speaker workflow. It is not an architecture claim for the native booking model.

## Results that changed decisions

| Study | Evidence | Decision |
|---|---|---|
| [Cached native readback](experiments/scheduling/native-injection/LATENCY_RESULTS.md) | Ten tool updates: first voiced gateway PCM median 3.338 → 1.552 s (53.52% lower); 12 key waveform checks | Keep cached prototype; real microphone/echo/subjective quality pending |
| [Prefill candidate](experiments/scheduling/native-injection/PREFILL_RESULTS.md) | Median 0.898 s, but only 11/12 key waveform guards | Reject; speed does not compensate for a wrong date |
| [Date-clarity follow-up](experiments/scheduling/date-clarity-2026-10-21/RESULTS.md) | Median 0.868 s and 20 core fact checks, but actual entry retained a human-confirmed wrong/unclear date | Reject; decoder replays did not reproduce or fix the known failure |
| [Normalization/token weights](experiments/duplexlab/reports/2026-10-21-normalization/RESULTS.md) | Four arms × seeds42/43/44, 500 training clips and 63 steps each; test362/10 current-split speakers: MER 9.7315/9.8353/9.8353/10.1329% | No reliable overall selection or weighting gain; native/uniform learned weights, batch1 inputs and transcriptions match |
| [Deletion ablation](experiments/duplexlab/reports/2026-10-20-deletion/RESULTS.md) | Random/selected/weighted MER 10.7691/10.3624/10.9632% on test280; same-data weighting has more total errors in all seeds | Do not adopt this weighting objective; assess total edits and deletion together |
| [Glossary control](experiments/glossary/ASCEND_200_RESULTS.md) | 600 real CPU inferences, paired validation complete | Overall small error change does not establish repaired terminology |

Scheduling timing is tool-update → same-epoch first voiced gateway PCM, excluding recognition, tool wait, browser/hardware and controlled cold start. The calendar is simulated and fixed at 2026-10-20. This is not native function calling or a real calendar integration.

DuplexLab studies use different frozen cohorts and cannot be ranked across rows. Confidence intervals include zero for the latest overall contrasts. Speaker IDs are disjoint within the current split but have historical project exposure; human audio review remains pending. Batch1 normalization equivalence does not cover ragged batches.

## Experiment navigation

[Decision index](docs/EXPERIMENT_INDEX.md) gives hypotheses, cohorts and adoption decisions. Experiment folders retain plans, aggregate summaries, revisions, failed candidates and reproduction instructions. Raw recordings, full predictions, SQLite state and checkpoints remain local.

## Start and verify

Python ≥3.11 for the core package; GPU workflows use separate Python 3.12 model environments. Start with [GETTING_STARTED](docs/GETTING_STARTED.md), then choose a workflow. Windows installs `tzdata` for the named calendar zone. Native backend preparation targets the Linux GPU environment and needs POSIX `patch`; its three deployment-tool tests are excluded from Windows portable checks.

Lightweight development setup:

```bash
python -m pip install --upgrade --only-binary=:all: -r requirements/bootstrap.txt
python -m pip install --build-constraint requirements/bootstrap.txt -r requirements/dev.txt
python -m pip install --no-build-isolation --editable ".[dev]"
python -m pytest -q
```

This checks synthetic/mocked CPU behavior; it does not load GPU models. [Development](docs/development.md) · [Architecture](docs/architecture.md) · [Deployment](DEPLOYMENT.md) · [Contributing](CONTRIBUTING.md).

## Next work

1. Freeze an actual-entry regression and diagnose booking startup-word/date clarity before further speed work.
2. Fix/save the tokenizer padding contract; compare mixed-length batches before another weighting claim.
3. Review reference/acoustic disagreements using train/dev, preserving observed tests as observed.
4. Run real microphone/headphone/AEC and broader natural-language tasks separately from synthetic demos.

## Reuse and publication

Original orchestration/adapters use MIT; models, ASCEND and integrated libraries retain their own licenses. [Third-party notices](THIRD_PARTY_NOTICES.md) list origins. Public source contains no weights, credentials or raw/generated audio-video. [Publication scope](PUBLICATION.md) · [Workspace](https://github.com/HenryVarro666/voice-ai-workspace).
