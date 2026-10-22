"""Independent waveform check with the existing local CPU ASR environment."""

import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--pattern", default="stage-*.wav")
args = p.parse_args()
waves = sorted(args.run.glob(args.pattern))
if not waves:
    p.error("No waveform files matched the requested pattern")

import torch
from qwen_asr import Qwen3ASRModel
from huggingface_hub import snapshot_download

torch.set_num_threads(4)
revision = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
snapshot = snapshot_download(
    "Qwen/Qwen3-ASR-0.6B", revision=revision, local_files_only=True
)
model = Qwen3ASRModel.from_pretrained(
    snapshot,
    dtype=torch.float32,
    device_map="cpu",
    max_inference_batch_size=1,
    max_new_tokens=256,
)
rows = []
for wave in waves:
    with torch.inference_mode():
        result = model.transcribe(
            audio=str(wave.resolve()), language="English", return_time_stamps=False
        )
    rows.append(
        {
            "file": wave.name,
            "relative_file": str(wave.relative_to(args.run)),
            "sha256": hashlib.sha256(wave.read_bytes()).hexdigest(),
            "text": result[0].text,
        }
    )
out = {
    "model": "Qwen/Qwen3-ASR-0.6B",
    "revision": revision,
    "device": "cpu",
    "dtype": "float32",
    "language": "English",
    "human_listening": False,
    "predictions": rows,
}
(args.run / "waveform-qwen-asr.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
