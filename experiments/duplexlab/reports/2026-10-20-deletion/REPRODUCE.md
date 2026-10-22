# Reproduction and evidence

Use Python3.12 and the fixed H100 lock [duplexlab-h100-py312.txt](../../../../requirements/locks/duplexlab-h100-py312.txt), physical GPU1 only. The checked experiment uses Qwen3-ASR-0.6B revision5eb144179a02acc5e5ba31e748d22b0cf3e303b0 and upstream Qwen3-ASR commit7c6daf77a2421100f5fb066495372c00129d39ff, with unchanged upstream training source. Hash-bound seed and retention adapters are recorded in the private runtime source snapshot and public aggregate.

The following commands require the documented private historical manifests/predictions and licensed CAiRE/ASCEND snapshot737e9800ae31be9932ba8464c80366559bd28424. They are an experiment reproduction route, not a clean public one-command installer. Data preparation and coordinator outputs are create-only; do not rerun into completed deletion-v1 directories. Keep failed artifacts and use a new reviewed study identity for another experiment.

```bash
PYTHONPATH=src python scripts/analyze_duplexlab_deletions.py
PYTHONPATH=src python scripts/prepare_duplexlab_deletion.py --root . --shards "$ASCEND_SNAPSHOT/main"
# Transfer only frozen manifests, originals and loss sidecar; exclude candidate-derived scratch.
# Before training: verify1347 WAV hashes, frozen protocol/selection/weights and500 processor suffixes.
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_duplexlab_deletion.py --root . --model "$PINNED_LOCAL_ASR"
# Copy runs back without checkpoints/optimizer weights; remap only runtime input paths for local score recheck.
PYTHONPATH=src python scripts/verify_duplexlab_deletion.py
PYTHONPATH=src python scripts/summarize_duplexlab_deletion.py
PYTHONPATH=src python scripts/report_duplexlab_deletion.py
```

Read the pre-test [batching repair addendum](../../DELETION_EVAL_ADDENDUM.md): all final arms set dev eval batch1, legacy batch8 fails seven of eight suffix checks, and all52 dev suffixes pass individually. Two initial completed runs and one partial step50 run are archived/excluded; final nine restart from base. The coordinator requires PREFLIGHT.json and rejects changed runtime source, protocol, selection or sidecar. The private preflight.py is archived with the run; it uses the original processor/collator without loading model weights and verifies shifted weighted CE gradients. A first incomplete-transfer preflight was correctly refused before model load; its log remains. Only this task's rsync was interrupted to exclude unnecessary candidate-derived audio. No external compute process was signaled.

Private evidence includes PREPARATION_SOURCE/source snapshots, PREFLIGHT/INPUT_VERIFICATION, FREEZE_BINDING, DEV_COMPLETE, actual18 training arguments, original63-step trainer states, checkpoint hashes bound to dev/test reloads, weighted-loss execution counters, raw SQLite/predictions, comparisons and runtime environment pins. Full audio, transcripts, predictions and model weights are not in the source package. Offline timing endpoints are specified in results; model inference batch time is not streaming first-token, TTS first-chunk, first-sound latency or strict cold-start latency.
