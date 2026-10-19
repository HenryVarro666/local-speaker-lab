"""Optional SpeechBrain comparison backend; explicitly resamples model bandwidth."""

import hashlib
import math
from pathlib import Path

from .common import absolute_path, dependency, read_audio, torch_device


class Separator:
    def run(self, payload: dict, options: dict) -> dict:
        torch = dependency("torch", "SepFormer")
        np = dependency("numpy", "SepFormer")
        sf = dependency("soundfile", "SepFormer")
        signal = dependency("scipy.signal", "SepFormer")
        separation = dependency("speechbrain.inference.separation", "SepFormer")
        device = torch_device(torch, options.get("device", "cuda:0"))
        model_name = options.get("model", "speechbrain/sepformer-wsj02mix")
        cache = Path(options.get("cache_dir", "~/.cache/speaker-lab/sepformer")).expanduser().resolve()
        # SpeechBrain reuses files already present in savedir before fetching from
        # `source`. A shared savedir can silently load the previous model's weights
        # when the user changes this replaceable backend's model option.
        model_cache = cache / hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:16]
        model_cache.mkdir(parents=True, exist_ok=True)
        model = separation.SepformerSeparation.from_hparams(
            source=model_name, savedir=str(model_cache), run_opts={"device": str(device)}
        )
        model_rate = int(model.hparams.sample_rate)
        divisor = math.gcd(16000, model_rate)
        results = []
        for job in payload["jobs"]:
            audio = read_audio(job["input_path"])
            if len(audio) / 16000 > 20.001:
                raise ValueError("Split separation jobs into clips of at most 20 seconds.")
            working = signal.resample_poly(audio, model_rate // divisor, 16000 // divisor)
            with torch.inference_mode():
                # separate_file normalizes every source independently; separate_batch does not.
                result = model.separate_batch(torch.from_numpy(working.astype("float32")).unsqueeze(0))
            sources = result.detach().cpu().numpy()
            if sources.ndim != 3 or sources.shape[0] != 1 or sources.shape[2] != 2:
                raise RuntimeError(f"SepFormer must produce exactly two sources; received {sources.shape}")
            output = absolute_path(job["output_dir"], exists=False)
            output.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(2):
                waveform = signal.resample_poly(sources[0, :, index], 16000 // divisor, model_rate // divisor)
                waveform = np.pad(waveform[:len(audio)], (0, max(0, len(audio) - len(waveform))))
                if not np.isfinite(waveform).all():
                    raise RuntimeError("SepFormer returned non-finite samples")
                path = output / f"source_{index}.wav"
                sf.write(str(path), waveform, 16000, subtype="FLOAT")
                paths.append(str(path))
            results.append({"id": job["id"], "sources": paths})
        return {"jobs": results, "model": model_name, "model_sample_rate": model_rate}
