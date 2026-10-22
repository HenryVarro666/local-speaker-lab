# Normalization and retention weights: frozen follow-up

2026-10-21 America/Chicago. One fixed follow-up to deletion-v1. Previous test280 is observed and is not the new test. No selector, target, scorer, inference-language or learning-rate changes are permitted. No test-token whitelist or outcome-driven rerun.

## Controls and fixed budget

Reuse the EXACT frozen deletion-v1 train1015, dev52 (primary40/challenge12), Random500 and Selected500; hash-check the old policy/selection/sidecar. All selected arms use identical ordered audio/targets. Train speakers8/9/11/15/19/21, dev5/6. Four arms, independently initialized from pinned Qwen3-ASR-0.6B revision5eb144179a02acc5e5ba31e748d22b0cf3e303b0, seeds42/43/44: twelve fresh SFTs.

- random: Random500, original native CE with study_sft (eval batch1).
- selected: Selected500, the identical original native CE adapter.
- uniform: Selected500, the unchanged retention_sft adapter with a sidecar whose every span list is empty. All valid supervised weights are exactly1. Same per-example shifted FP32 CE and Trainer accumulation normalization as filtered.
- filtered: Selected500, unchanged retention_sft and EXACT previous 2x sidecar. The historical filtered key denotes weighted normalized CE.

500 examples, epoch1, batch1, gradacc8, LR2e-5, BF16, final63 steps, eval batch1, identical52-dev eval_file, seed adapter unchanged. Verify actual18 training arguments, final checkpoint hashes, all suffixes, and actual uniform0 versus weighted369 boosted training subwords. The adapter's declared boost rule remains2 in both runtime audits; for uniform it has no eligible spans and observed weights are all1. Never call it a weighted control. No new loss implementation. This matches examples/audio/steps/hyperparameters, not exact BPE/FLOPs.

Diagnose the pinned library before training: actual native Trainer flag, thinker loss-function argument forwarding, actual ASR forward all-ones CE equality, and a controlled actual-Trainer scalar backward accumulation comparison. Report pre-clipping gradient scale, not an asserted effective Adam learning-rate multiplier. Native versus uniform also changes native versus externally computed CE code paths, checked for forward equality. Do not claim a complete normalization-by-weighting factorial: there is no native-accumulation weighted arm.

## New recordings, fixed before outcomes

Other two cached ASCEND official train shards, same pinned dataset revision737e9800ae31be9932ba8464c80366559bd28424. Test speaker IDs1/2/4/10/13/14/16/18/24/26. Per ID zh20/mixed10/en10 except ID13 zh10, IDs4/10/26 en3, ID24 mixed3: total362, zh190/mixed93/en79. These quotas are chosen from metadata-only remaining availability, before audits/model output. Hash order seed20261024,1–12s, audit-pass only; fail if infeasible. Exclude all prior branch manifests including deletion-v1 test280 and old20 by clip/file/PCM, and the entire1130-recording official validation split by file/PCM (including the925 protected recordings). No protected inference. Current-role speakers are disjoint, but all IDs have historical project exposure, historical glossary read all official train texts, and base pretraining/session independence is unknown.

## Ordering, contrasts and interpretation

Freeze protocol/source/manifests/sidecars/selection before any new dev/test output. Base plus all twelve dev ASR reloads finish before DEV_COMPLETE. No adaptation to dev outcomes. Then base plus twelve new-test runs, automatic language, BF16/SDPA, max256, no aligner. No best-seed/checkpoint choice; preserve failures.

Primary contrasts on strict MER (unchanged original references/NFKC scorer): normalization = uniform minus selected; weights = filtered minus uniform. Secondary: selection = selected minus random; joint = filtered minus random. Report all four rather than cherry-pick. 5000 paired crossed training-seed/whole-speaker bootstraps, seed20261020; exploratory unadjusted intervals with ten clusters and three seeds, not population proof. Also report S/D/I/N, language, speaker, fillers/frequency/repetition/switch/span, D/N intervals and all-three-seed stable deletion diagnostics. A deletion becoming a substitution is not recovered content. Human listening remains pending; source/segmentation ambiguity remains distinct from confirmed acoustic omissions.

Physical H100 GPU1 only, wait other work; no signals to unrelated jobs, no apex, agents, commits, pushes or deployment. Public aggregates/code/figures only; full references/hypotheses/audio/checkpoints/replay remain private. Offline timing is subprocess load-to-final transcript, not streaming first text/TTS/sound or strict cold start. Synchronize bilingual entrypoints, results, shared status/evolution/samples after completion.

## Preparation-only amendments, before training or new ASR

Attempt0 stopped at an incorrect925 count assertion for the complete validation split, which actually has1130 source rows. Protect all1130. Attempt1 passed source/audio audits but failed the initially proposed speaker4 English10 quota: only3 remained after exact audio/PCM exclusion. Reduce only that quota10→3 (369→362 total). Both failures and input directories are preserved; no training, development or new-test ASR occurred. Final policy is frozen only after this metadata/audio-only feasibility correction.
