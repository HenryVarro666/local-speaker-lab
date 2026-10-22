# Reproduction and evidence audit

2026-10-20. Continuation workspace, not a clean-machine installer. Requires continuation-v1 source data and complete pool-base predictions under runs/duplexlab/h100-continuation-v1; prior coverage-v1/multiseed-v1 manifests for exclusion; pinned upstream SFT and seed adapter; and original H100 train-{random,filtered}.json configurations. Preserve revisions in the frozen protocol and replace machine paths only in a separate reproduction workspace. Outputs are create-only; never overwrite completed studies.

1. Run prior analysis without ASR. Check all1800 alignments against the original strict counts and original WAV hashes. Keep references/scores and human-listening status unchanged.
2. Read DISFLUENCY_PROTOCOL before selecting the candidate. Prepare current train1015/dev40 and new test320/4people. Exclude all previous branch clips, including coverage-v1's300, using ID/file/PCM. Verify all weighted and categorical marginals and≤1% audio budget; preserve actual preparation sources and hashes.
3. Transfer1375 original WAVs and verify every hash before launch. Use physicalGPU1 only; wait for external work without signals. Six models restart from the same base. Complete all dev reloads and the frozen marker before test.
4. [Dev addendum](../../DISFLUENCY_DEV_ADDENDUM.md): after seeing that dev40 lacks repeated-copy≥2, freeze12 new dev clips by textual coverage before test-base exists. Record both precheck and completed-transfer freeze with audio/manifest/policy hashes. Supplementary diagnostics run only after main END_HEALTH; they never affect training, model or seed choice. Primary dev40 remains the training evaluation file.
5. Mirror complete records/SQLite/protocols/sources/trainer states/exports to local runs; weights stay on H100. Independently re-score all14 batches/2520 inferences, reproduce both5000-replicate comparisons, then aggregate disfluency deletions and render reports. Do not repeat inference to verify statistics.

```bash
PYTHONPATH=src python scripts/analyze_duplexlab_disfluency.py
PYTHONPATH=src python scripts/prepare_duplexlab_disfluency.py --root . --shards /path/to/pinned/ASCEND/main
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_disfluency.py --root . --model /path/to/pinned/Qwen/snapshot
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_auxdev.py --root . --model /path/to/pinned/Qwen/snapshot
.venv-dev/bin/python scripts/verify_duplexlab_disfluency.py
.venv-dev/bin/python scripts/verify_duplexlab_auxdev.py
.venv-dev/bin/python scripts/summarize_duplexlab_disfluency.py
PYTHONPATH=src uv run --no-project --with matplotlib --python 3.12 python scripts/report_duplexlab_disfluency.py --root .
PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH .venv-dev/bin/python -m pytest -q -rfs
```

The observed H100 Python3.12.11/Torch2.8.0/Transformers4.57.6/qwen-asr0.0.6/SciPy1.16.3 runtime uses the existing lock. Verification binds exports, seeds/budgets, original trainer states, and actual dev/test weight hashes, not just filenames. Current role isolation is not project-wide never-seen-speaker or verified session independence. Newtest is observed after this study; a later selector needs another frozen test. Source packages explicitly include only reviewed public artifacts, not private audio/full text/weights/databases.

Auxdev preparation is task-specific: `prepare_duplexlab_auxdev.py` uses this workspace’s existing pinned local HF cache path; run it before any new test output and preserve the original executed preparation source. Do not run it after inspecting new outcomes and call that a pre-test freeze. The recorded12 have313 audit-pass candidates, speaker counts7/5, language counts4zh/2en/6mixed and exactly2 examples meeting each of the four hard-coverage minima. Seven extra batches add84 inferences; total21 batches/2604,320 new test clips and12 new dev clips. Metadata/browser checks are not human listening.
