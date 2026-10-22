# Prior deletion increase: reference, boundaries and word retention

All320 recordings × two arms × three seeds (1920 strict alignments) from disfluency-v1 were recomputed. The original scorer tie order and normalization were retained: diagonal before deletion before insertion. The three-repeat totals were Random S889/D296/I198 versus Disfluency S859/D318/I178. Overall edits fell28 while deletions rose22. Lexical filler deletions26→25, single ASCII-letter deletions27→27 and remaining deletions243→266. A single ASCII letter can be an ordinary English word, not necessarily an acronym.

Position accounting explains the net+22D:63 newly deleted positions and41 resolved positions across the three repeats. Per-seed added/resolved counts are26/12,14/16,23/13 (net+14/−2/+10). Of63 additions,35 are forced deleted across every minimum-edit path and28 have alternate optimal alignments. Additions split into filler5/single ASCII1/other57; resolutions filler6/single ASCII1/other34. This additional accounting uses the same old outputs after final training, without changing any frozen selector, loss or new-test condition. It does not establish35 acoustic word omissions.

Random has75 reference positions deleted in all three seeds; the candidate has82. Eleven candidate-stable positions across11 recordings are not random-stable; four earlier stable positions disappear. Only five of the11 are forced deleted under **every** minimum-edit path in all candidate seeds. None of those five is an exact match across every optimal path in all three random seeds. One repeated-character count clearly changes, but which repeated copy is deleted is ambiguous. These are transcript comparisons, not acoustic ground truth.

| Recording suffix | Stable position / token | Text-level diagnosis |
|---|---|---|
|03611|head 嗯|Random also deletes in43/44; only one additional seed loses it.|
|05996|但|Large cross-language mistranscription; optimal alignment position is ambiguous.|
|06064|tail 吧|Random also deletes in44. Candidate loses tail in all seeds; listening pending.|
|05988|莱|Source contains UNK and a Leica transliteration/segmentation mismatch; ambiguous alignment.|
|08446|tail 了|Random substitutes the last particle in all seeds. Candidate has D+1 each but **one fewer total edit each**; deletion increase is not overall regression.|
|04255|as|Repeated short English phrase with alternative optimal edit paths.|
|06633|说|Source G math / hypothesis GMAT plus nearby token alignment; Random also deletes in42/43.|
|08903|很|Candidate repeats once fewer in all seeds; Random preserves both copies. D+1/error+1 each; the identity of the dropped copy is ambiguous. Acoustic count still needs listening.|
|04662|是|Substitution changes deletion placement; no increase in total D in any seed.|
|04706|哼|Interjection orthography and tail presence; Random also deletes in43/44.|
|07082|alphago|alpha go word boundary plus Chinese/English spelling, not verified swallowed speech.|

For07082 only, a separately labeled boundary sensitivity splits source alphago into alpha go. It does **not** remove the residual difference: Random D0/0/0, candidate D2/2/2 on the sensitivity reference. Other script/transliteration differences remain. The sensitivity never replaces primary references, metrics, training labels or new-test scoring. Full per-seed strict/sensitivity counts and the all-path checks are in [PRIOR_ANALYSIS.json](PRIOR_ANALYSIS.json); complete source/hypothesis text stays in private run artifacts.

No old-test word list enters the new selector or loss. New rules generalize train-only base deletion risk, tail coverage and adjacent repetitions; source-UNK presence is matched, uncertain tokens excluded from extra loss weight. Uppercase acronym and numeric representations retain ordinary supervision. Human listening and source-reference correction remain pending; they must be recorded separately before making claims about actual swallowed words.

A separate engineering finding during the new study reproduced upstream batched dev label leakage. It affects interpretation of logged batched eval_loss, not the previously recomputed offline ASR transcripts. The [execution addendum](../../DELETION_EVAL_ADDENDUM.md) keeps the failure and applies batch1 evaluation to every final arm before any new-test inference.
