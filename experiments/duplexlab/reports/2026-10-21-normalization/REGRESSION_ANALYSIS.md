# Deletion diagnosis on the new 362 recordings

Strict primary references/tokenization are unchanged. All24 SFT evaluation summaries are independently aligned and checked against SQLite-backed S/D/I/N. Counts below sum the three model repeats; source recordings remain362.

| Arm | All D | Filler-token D | Single ASCII letter D | Other D | Stable D positions in all 3 seeds |
|---|---:|---:|---:|---:|---:|
| Random native CE | 221 | 12 | 13 | 196 | 49 |
| Selected native CE | 237 | 15 | 16 | 206 | 52 |
| Selected normalized CE | 237 | 15 | 16 | 206 | 52 |
| Selected weighted CE | 271 | 13 | 15 | 243 | 50 |

| Contrast | D/N mean delta pp | Paired95% interval pp |
|---|---:|---|
| joint | +0.3461 | [-0.0518, +0.8547] |
| selection | +0.1107 | [-0.1938, +0.4078] |
| normalization | +0.0000 | [+0.0000, +0.0000] |
| weights | +0.2353 | [-0.1028, +0.8292] |

Weighted minus uniform normalized control has 7 additional stable aligned positions across 6 recordings. A stable reference position is not proof of an audible omission, and an extra stable position may coexist with equal/fewer total deletions because alignment moves elsewhere. Forced-all-optimal-path flags address positional ambiguity only; source references and acoustic truth still require listening.

| Clip ID | Reference positions/tokens (zero-based) | Weighted forced D in all paths | Uniform deleted seeds / matched all paths | Per-seed D delta | Per-seed total-edit delta |
|---|---|---|---|---|---|
| ascend-train-09118 | 4:你, 12:有 | True, True | [43, 44] / False, [] / False | [2, 0, 15] | [-4, -1, 11] |
| ascend-train-05162 | 2:是 | True | [43] / False | [0, 0, 0] | [-1, 0, -1] |
| ascend-train-03312 | 1:ll | True | [42, 43] / False | [7, 0, 10] | [4, 0, 4] |
| ascend-train-06485 | 11:这 | False | [43, 44] / False | [1, 0, 0] | [1, 0, 0] |
| ascend-train-07160 | 4:为 | False | [42, 44] / False | [0, 1, 0] | [0, 5, 0] |
| ascend-train-09444 | 28:inspirations | False | [42] / False | [0, 1, 0] | [0, 0, 1] |

## Consistent clip regressions versus stable word positions

These are separate criteria. More total errors in every seed does not require the same omitted token; an additional stable aligned token need not mean more total errors in each seed. Counts cover all 362 recordings, not only the stable-position table.

| Contrast | More total edits in all 3 seeds: clips | More D in all 3 seeds: clips |
|---|---:|---:|
| joint | 8 | 2 |
| selection | 6 | 2 |
| normalization | 0 | 0 |
| weights | 2 | 0 |

The paired uniform/weighted comparison isolates span weighting with the same adapter, normalization, data, labels and hyperparameters. Native/normalized also changes the CE computation route; numerical equivalence is checked, while small floating-point differences remain possible. It is not a complete normalization-by-weight factorial.

Source UNK, repeated copies, Chinese particles, transliteration and English segmentation can change an optimal alignment without establishing missing audio content. No new-test reference is rewritten, sensitivity normalization substituted, or case token added to training. All new-test cases are now observed diagnostics and cannot be reused as an untouched test after tuning.

Private playback contains full source audio and all 4 arms/3 seeds. Audio metadata and browser filters can be validated automatically; human listening is pending. [Strict results](RESULTS.md) · [Raw aggregate groups](DELETION_GROUPS.json).

## Text-only adjudication of the flagged cases

Of7 extra stable positions,4 are forced D in every optimal path and3 have positional ambiguity. **None has an unambiguously matched uniform counterpart in all 3 seeds.** No new acoustic omission is confirmed from these alignments alone. Both clips with higher total errors in every seed have D delta0/0/0. Source labels and strict primary scores remain unchanged.

| Clip | Text-level interpretation; acoustic review pending |
|---|---|
| ascend-train-09118 | The stable positions are 你 and 有. Uniform already deletes the first in43/44 and never unambiguously matches the second across all seeds. Weighted44 drops a much larger opening span;42/43 total errors improve. Not a new three-seed omission. |
| ascend-train-05162 | The 是 position competes with 叫/other wording. D delta is0/0/0 and total errors improve or tie. A stable D position is not a stable regression. |
| ascend-train-03312 | ll is a fragment of the reference contraction you'll, not an independent word. Uniform already deletes/substitutes it; weighted42/44 lose a larger bilingual span while43 matches uniform. Do not normalize it away or label it a newly omitted word across all seeds. |
| ascend-train-06485 | 这 competes with the surrounding wording; the weighted deletion is not forced in all optimal paths. Only42 has an extra total edit. |
| ascend-train-07160 | 为人类 versus 未来 changes nearby substitutions and deletions. Uniform42/44 already share the weighted output; only43 regresses. |
| ascend-train-09444 | inspirations versus inspiration plus so/suoyi involves number and English-Chinese boundary spelling. Alignment is nonunique;42 outputs are identical. No three-seed new acoustic omission is established. |
| ascend-train-07269 | Total-edit deltas+1/+1/+1, D0/0/0. English intelligent/intelligence wording and the initial Chinese particle differ. The source may contain nonstandard English; it is not silently corrected and acoustic/reference adjudication is pending. |
| ascend-train-05729 | Total-edit deltas+2/+1/+1, D0/0/0. Opening-word substitutions and repeated-copy counts drive the strict error increase; no deletion increase occurs on this clip. |

The contraction fragment `ll`, English plural/code-switch boundary and deletion-versus-substitution cases are not converted into a word-omission tally. Full source/hypotheses/audio remain in private replay. This post-test review changes no training policy, reference or primary normalization.
