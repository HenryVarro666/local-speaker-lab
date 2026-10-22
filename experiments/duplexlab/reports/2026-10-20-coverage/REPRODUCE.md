# Reproduction and evidence audit

This is a continuation in the existing DuplexLab workspace, not a clean-machine installer. It requires the frozen continuation-v1 data, complete base pool predictions mirrored locally under runs/duplexlab/h100-continuation-v1/pool-base, the earlier train-{random,filtered}.json pinned trainer configuration on H100, and the existing seed adapter. Those configurations are local evidence with machine-specific paths; preserve the model/trainer revisions and replace paths only in a separate reproduction workspace. Outputs are create-only; do not overwrite completed studies.

Pinned ASCEND737e9800ae31be9932ba8464c80366559bd28424; Qwen3-ASR-0.6B5eb144179a02acc5e5ba31e748d22b0cf3e303b0; upstream SFT7c6daf77a2421100f5fb066495372c00129d39ff. Existing H100 lock supplies SciPy1.16.3/HiGHS and the speech runtime. No new GPU-package installation was needed.

1. Read COVERAGE_PROTOCOL. Prepare the new split and fixed arms without looking at new test predictions. Only current train base predictions are used for the difficulty proxy.
2. Preserve actual preparation source/solver records before any cosmetic source formatting. Freeze manifests, policy, selected IDs and hashes. Inspect all four groups of exact marginals; no outcome-based rule search.
3. Transfer original WAVs, manifests, protocol, selection and executable sources to the isolated H100 project. Verify every WAV hash after transfer. The initial premature launch in this study failed at the missing-WAV guard before export/model work; its log is retained.
4. Run six fresh SFTs on physical GPU1. Every model is evaluated on dev40; DEV_COMPLETE must precede new test evaluation. Same batch1/grad_acc8/lr2e-5/epoch1/BF16,500 clips/63 steps per model. All three seeds use the explicit adapter, including42.
5. Mirror completed run records, SQLite, protocols, actual arguments, checkpoint trainer states, source snapshots and export JSONL/metadata back to local runs/duplexlab/coverage-v1. Model weights/optimizer files remain on H100. Do not copy partial SQLite as final evidence.
6. Independently re-score, verify source and model bindings, recompute both5000-replicate comparisons, then render the public report and private playback.

```bash
PYTHONPATH=src python scripts/prepare_duplexlab_coverage.py --root . --shards /path/to/pinned/ASCEND/main
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_coverage.py --root . --model /path/to/pinned/Qwen/snapshot
.venv-dev/bin/python scripts/verify_duplexlab_coverage.py
PYTHONPATH=src uv run --no-project --with matplotlib --python 3.12 python scripts/report_duplexlab_coverage.py --root .
PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH .venv-dev/bin/python -m pytest -q -rfs
```

The verifier loads no ASR model. It checks14 complete ASR batches/2380 inferences,1355 source audio files, current speaker isolation, prior-clip exclusion, exact coverage marginals, export roles/text prefixes/file hashes, actual seeds/budgets, finite original trainer logs and the weight hashes actually reloaded on dev/test. It reproduces server statistics using the same frozen references. Original source references remain unchanged; browser checks do not count as human listening.

Source text exposure and prior speaker use are disclosed in DESIGN and the protocol. This split supports a current-experiment speaker-disjoint comparison with new audio; it is not an external-corpus, never-seen-speaker or verified conversation-independent claim. A single constrained candidate is compared against Random; there is no same-split unconstrained-filter ablation. Do not directly compare its test MER with the earlier400-clip experiment.
