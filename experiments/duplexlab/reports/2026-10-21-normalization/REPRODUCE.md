# Reproduction

Use the pinned dataset revision, model snapshot, upstream SFT source and122 third-party dependencies from `requirements/locks/duplexlab-h100-py312.txt`. Model revision5eb144179a02acc5e5ba31e748d22b0cf3e303b0; upstream commit7c6daf77a2421100f5fb066495372c00129d39ff. The source entrypoint is [NORMALIZATION_PROTOCOL.md](../../NORMALIZATION_PROTOCOL.md).

The experiment relies on the already frozen deletion-v1 inputs, selection and train-only sidecar. It never re-optimizes the selector. Local input preparation:

```bash
PYTHONPATH=src python scripts/prepare_duplexlab_normalization.py --root . --shards /path/to/pinned/ASCEND/main
PYTHONPATH=src python scripts/verify_duplexlab_normalization_inputs.py --shards /path/to/pinned/ASCEND/main
```

The final preparer implements the feasible362-clip quotas. Historical preparation-only failures remain private. Copy only original audio/manifests/sidecars, excluding candidate-derived scratch; wait for the transfer to finish. Verify hashes, processor supervision suffixes (train500/dev52), actual pinned Trainer scaling and real-model forward CE before launching. The exact diagnosis/preflight execution sources and JSON evidence are private under `runs/duplexlab/normalization-v1` and hash-bound in the freeze marker. These scripts contain local cache paths and are not an additional public model implementation.

On an available physical GPU1, in the isolated pinned environment:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 python scripts/run_duplexlab_normalization.py --root . --model /path/to/pinned/snapshot
```

The runner refuses changed frozen inputs/sources, existing outputs or missing preflight evidence; waits for other GPU1 jobs; trains all12 fresh models and reloads each on dev before opening the new test. Verify final63 steps, actual18 arguments and checkpoint hashes. Uniform uses the unchanged retention adapter with empty span lists, exactly0 boosted subwords; weighted uses the original sidecar and369 boosted subwords. Both have500 training calls and52 dev calls.

Mirror results after completion with `rsync --exclude='*.wav'`: per-run audit WAVs duplicate the already verified original audio. Keep those waveforms remotely; do not copy model weights or live SQLite files. From the project root:

```bash
PYTHONPATH=src python scripts/verify_duplexlab_normalization.py
PYTHONPATH=src python scripts/summarize_duplexlab_normalization.py
PYTHONPATH=src python scripts/report_duplexlab_normalization.py
```

The report command needs Matplotlib; CPU validation needs NumPy/SoundFile, input source verification also PyArrow. Compare source fingerprints, strict original SQLite-backed scores, all four MER and D/N crossed-seed/whole-speaker bootstraps and actual checkpoint reloads. Raw references/hypotheses, audio, SQLite, checkpoints, training logs and browser replay stay private. Public packages list individual aggregate reports explicitly; do not add an entire experiment or runs directory to the allowlist. No credentials, deployment, streaming/hardware timing or human acceptance is implied.

The inherited mixed-language targets use `language None<asr_text>` without selecting one language. The [pinned upstream fine-tuning guide](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning/README.md) permits an unknown-language prefix and explains that it does not supervise language identification. This study holds that choice fixed and evaluates transcription text, not language-ID quality. No training prefix or dependency was changed after this check.
