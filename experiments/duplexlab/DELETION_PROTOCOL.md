# Deletion retention: frozen study protocol

2026-10-20 America/Chicago. Frozen before new selection, training and new-test model output. Prior disfluency-v1 test320 is observed diagnostic data; primary normalization and references remain unchanged. All 1920 old paired alignments are rechecked, including all minimum-edit-path position ambiguity. Stable D positions are not automatically acoustic omissions. Repeat-count loss, tail substitution/deletion, source UNK, acronym/transliteration and segmentation are reported separately; human listening remains pending. No test-token whitelist is used.

## Data and selection

Train remains the audited1015 from speaker8/9/11/15/19/21. Dev is frozen upfront as the previous primary40 plus previous challenge12, speaker5/6; all52 are now the common training eval_file. Report primary40 and challenge12 separately as diagnostics. These are known development examples. No new dev is selected by model errors.

New test: seven speaker IDs10/13/14/16/18/24/26, each zh20/mixed10/en10, total280. Use other two official train shards, audio1–12s, audit pass, hash order seed20261023. Exclude ALL historical branch manifests including latest disfluency test320 and auxdev12, old20, prior multiseed300/coverage300 and continuation roles; clip/file/PCM hashes must not overlap. Infeasible quotas fail before training. These are new recordings and current-role speaker-disjoint, not project-never-seen people. Prior glossary read all train texts; session provenance and base pretraining exposure unknown.

Random500 retains selection seed20261021 and previous IDs, mixed250/zh125/en125. New selector retains all previous frequency/repetition/switch-span/speaker/difficulty/language-duration marginals, adds per-language base-D count bins0/1/2+, any last-quarter-reference-token D, and source-UNK presence. Base predictions come ONLY from the fixed train pool. Minimize sum((S+I+0.5D)/N), ID tie1e-7, so a deleted-word training example is discounted less harshly than before while deletion-risk bins are retained. It is still a source-reference proxy. Audio within1% of Random. MILP120s/gap0, accept verified feasible nonworse solution or fall back to Random. No dev/test candidate sweep.

## Three arms and objective ablation

All start independently from the pinned Qwen3-ASR-0.6B base, seeds42/43/44, each500 clips/epoch1/batch1/gradacc8/lr2e-5/BF16/63 optimizer steps. Nine fresh runs. Same52 dev, all actual training arguments verified. Save final63-step checkpoint, no best-seed/checkpoint choice.

- random: Random data, original pinned upstream CE and seed adapter.
- selected: new selected data, same original CE. Isolates selection versus random.
- filtered: EXACT selected IDs/audio/targets, new retention CE adapter. Isolates objective versus selected. The historical comparison key filtered denotes this joint candidate.

Retention CE gives weight2 to target subwords overlapping eligible reference-token spans either (a) deleted by train base across ALL optimal edit paths, or (b) in adjacent repeated1–4-token motifs. Exclude source-UNK spans, digits, standalone ASCII letters and all-uppercase ASCII words from boosting; their original CE supervision stays weight1. All other supervised tokens, language header and EOS stay1; upstream prompt/padding mask stays0. No target rewriting, new prompt or glossary. Spans computed only from train reference/base hypotheses and frozen in a hash-bound sidecar. Tokenizer offset mapping must exactly reproduce the processor supervised suffix for EVERY collator example; fail on discrepancy. Unicode cross-character composition requiring offset review fails closed. Repeat proxy includes lexical reduplication and is not a human hesitation annotation.

Loss is sum(weight*FP32 shifted causal CE)/sum(valid weights) per example; Trainer gradient accumulation then applies. Additional model computation is only this loss; numeric objectives intentionally differ. Dev uses unweighted CE in the retention adapter. Record actual boosted-subword counts and masks, sidecar/adapter/source hashes. Frozen upstream source unmodified. This matches examples/audio/steps/hyperparameters, not exact BPE tokens or FLOPs. The third arm is a small ablation, not a full factorial study.

## Evaluation and acceptance

Complete base plus all nine dev reloads before DEV_COMPLETE, then base plus nine new-test runs. All same auto language/BF16/SDPA/max256/no aligner. Primary strict MER unchanged, report S/D/I/N, seed/language/speaker/filler/repetition groups; D/N and stable deletion positions additionally reported. Main contrasts: joint candidate minus random; selection-only minus random; objective-only minus selected. 5000 paired crossed training-seed/whole-speaker bootstrap, seed20261020; intervals exploratory and three comparisons not multiplicity adjusted. Improvement requires MER and deletion evidence together; a D reduction that merely becomes substitutions is not a content-retention success. No outcome-driven reruns or threshold changes.

Use isolated deletion-v1 directory and physical H100 GPU1; wait external jobs, never signal other projects or access apex. Public numerical aggregates/code/figures only; raw audio/full transcripts/predictions/weights stay private. Offline batch timing load-to-final-transcript is not first-token/TTS/first-sound latency or strict cold start. Preserve failures; synchronize bilingual README, report, shared status/evolution/samples. No agents, commits, pushes or deployment.
