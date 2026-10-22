# Separating normalization from retention weighting

The prior deletion study changed two factors together. Its negative weighted result cannot identify whether accumulation normalization or token weighting caused the change. This follow-up retains the previous data selector and adds the missing uniform normalized control.

| Arm | Training records | Per-token weights | Accumulation path |
|---|---|---|---|
| Random native | Same Random500 | Native CE | Original upstream |
| Selected native | Same Selected500 | Native CE | Original upstream |
| Selected normalized | Same Selected500 | All valid labels1 | Retention adapter, empty spans |
| Selected weighted | Same Selected500 | Original train-only eligible spans2, others1 | Same retention adapter |

The primary comparisons are normalized minus native on selected data, and weighted minus uniform with normalization fixed. Selected minus Random and weighted minus Random are reported as secondary comparisons. This is a four-arm chain, not a full2x2 factorial: native-accumulation weighted CE is absent.

All12 models restart from the same base, use seeds42/43/44,500 examples,63 optimizer updates and the same52 known dev recordings. Inference language stays automatic. Newtest362 recordings from10 current-role-separated speaker IDs are frozen before outcomes; history and session/pretraining exposure limit independence claims. Quotas were corrected only during input feasibility checks, before new ASR. See the [frozen protocol](../../NORMALIZATION_PROTOCOL.md).

The pinned Trainer diagnosis checks the actual native flag and forwarding behavior, forward all-ones CE equivalence, and a controlled scalar backward experiment. The observed8x full-window pre-clipping gradient scale does not establish an8x effective Adam learning rate. The last partial window has4 examples. Maximum-gradient clipping and optimizer dynamics can reduce scale effects.

All four outcomes, strict S/D/I/N, language/filler/repetition groups and paired whole-speaker intervals are reported. Whole-clip source references are unchanged; alignment-position ambiguity, segmentation and source errors remain separate from acoustic omissions. Human listening is pending. This offline ASR experiment supplies training/data/evaluation evidence, not streaming speech or Avatar acceptance.

Before this follow-up's model outputs, all189 optimizer steps per prior arm were found to exceed max_grad_norm1 (native selected median226, weighted normalized29.875). This strengthens the reason to test normalization rather than infer an effective step-size difference. Identical trained hashes across controls are permitted: fresh execution and reload provenance are verified separately from numerical weight uniqueness.

A supplemental16-case FP32/BF16 length/mask comparison against the pinned native `ForCausalLMLoss` found maximum forward difference2.3842e-7 and identical gradients (max0). Real-Trainer scalar checks confirmed both4x partial-window and8x full-window accumulation ratios before clipping. These CPU numerical checks complement the actual GPU ASR forward check; none is an ASR quality result.

Native Random500 also passes500 exact processor-supervision suffix checks (7305 supervised BPE pieces); selected500 has7383. With the same52 dev checks,1052 role-level suffix checks pass. Thus Random/Selected have the same examples/steps and near-matched audio duration, not an exactly equal subword budget. The two primary selected-data contrasts have identical subword targets.

Native and normalized logged training losses have different accumulation scales. A smaller logged normalized loss is not evidence of a better transcription model; quality decisions use the frozen reference-based ASR evaluation.

Preparation provenance clarification (before new-test outcomes): speaker4 English has44 metadata-eligible candidates;41 are automated clipping `needs_review`,3 pass. The frozen addendum's phrase about audio/PCM exclusion was imprecise for this quota bottleneck. It is a quality-audit constraint, not evidence that41 files were duplicates. The protocol artifact is preserved verbatim; no quota, reference, threshold, policy or model setting is changed by this wording correction.

Checkpoint artifact hashes are not conflated with learned-weight hashes. Native and uniform save different transient backend padding state in `tokenizer.json` (BatchLongest/Left versus null); their other tokenizer JSON content is compared structurally. The pinned wrapper explicitly processes one audio at a time with padding=True. A CPU check reconstructs its exact audio normalization/chunking/chat-prompt/processor calls and compares every input tensor on all52dev+362test audio inputs for each seed, without model weights, reference text or ASR output. Equivalence is scoped to the actual batch1 path; it does not certify arbitrary ragged batched inference. Final full-string predictions are checked separately.

Reloading those saved tokenizers yields padding_side left (native) versus right (uniform/weighted). This is a real serialization side effect, not a claim of byte-identical processor artifacts. Exact tensor equivalence is established only for this study's max_inference_batch_size1. Standardizing saved padding semantics and verifying mixed-length batched inputs belongs in a separate version; frozen checkpoints and inference settings are preserved here.
