# Reproducing and auditing this continuation

This run extends the completed continuation-v1 experiment. Keep its frozen random500 / filtered500 exports, independent dev40, original seed42 checkpoints and source snapshots. Do not regenerate selections with a different root or silently substitute an updated trainer. Model revision: `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; ASCEND revision: `737e9800ae31be9932ba8464c80366559bd28424`; upstream SFT commit: `7c6daf77a2421100f5fb066495372c00129d39ff`.

The experiment protocol and audit-only amendment are fixed before new outcomes. All complete transcripts/audio/checkpoints stay private. Commands assume the repository root and a compatible isolated runtime; use the checked-in H100 lock. The training/evaluation commands below launch real GPU work and refuse existing outputs.

```bash
PYTHONPATH=src python scripts/analyze_duplexlab_regressions.py
PYTHONPATH=src python scripts/prepare_duplexlab_multiseed.py --root . --shards /path/to/pinned/ASCEND/main
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_multiseed.py --root . --model /path/to/pinned/Qwen3-ASR-0.6B/snapshot --phase train
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_multiseed.py --root . --model /path/to/pinned/Qwen3-ASR-0.6B/snapshot --phase evaluate
```

The new phase trains only43/44; seed42 is reused from the prior run. Its adapter imports the unmodified pinned trainer, sets the global seed before loading and injects the explicit Trainer seed. `data_seed=None` keeps the upstream fallback. Capture actual `training_args.bin` settings, not just the launch configuration. Seed42's earlier upstream run lacked pre-load set_seed; its Trainer effective seed was42. The fixed fully pretrained initialization is the same, but this execution distinction is retained as a limitation.

After completion, mirror new run records/protocols/SQLite/predictions/source snapshots into `runs/duplexlab/multiseed-v1/`. The old remote continuation is mirrored locally under `runs/duplexlab/h100-continuation-v1/`. Local source audio is under `data/duplexlab/multiseed-v1/`; the original old test audio remains under continuation-v1. Preserve the pre-outcome `SENSITIVITY_FREEZE.json` (209 pass-only IDs and amendment hash), source manifests and private regression files.

```bash
.venv-dev/bin/python scripts/verify_duplexlab_multiseed.py
PYTHONPATH=src uv run --no-project --with matplotlib --python /opt/homebrew/bin/python3.12 python scripts/report_duplexlab_multiseed.py --source runs/duplexlab/multiseed-v1 --old runs/duplexlab/h100-continuation-v1 --data data/duplexlab/multiseed-v1 --output experiments/duplexlab/reports/2026-10-20-multiseed
PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH .venv-dev/bin/python -m pytest -q -rfs
```

Local verification re-scores all eleven new complete runs against original WAVs and frozen protocols; it never loads an ASR model. It also independently recomputes paired seed/speaker comparisons, checks six distinct checkpoint hashes and creates the predeclared pass-only sensitivity result. The report refuses incomplete training or checkpoint reloads and rejects private paths/full transcripts from public aggregate JSON. Source releases use an explicit report allowlist.

A failed initial base freeze used the wrong revision representation and was rejected before any model inference. It is preserved under `failed-base-revision-before-inference/`; both coordinator source versions are archived. The corrected evaluation uses the Hub snapshot basename for the base and a content digest for trained checkpoints. No training was repeated for this repair.
