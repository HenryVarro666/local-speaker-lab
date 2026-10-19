"""ClearVoice 0.1.2 public inference API, guarded against unsafe defaults."""

from contextlib import contextmanager
import os
from pathlib import Path

from .common import absolute_path, dependency, read_audio, torch_device


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def checkpoint_ready(directory: Path) -> bool:
    """Upstream can silently leave random weights if downloads fail. Fail closed."""
    pointer = directory / "last_best_checkpoint"
    if not pointer.is_file():
        return False
    names = pointer.read_text(encoding="utf-8").splitlines()
    if not names or not names[0].strip():
        return False
    weights = (directory / names[0].strip()).resolve()
    if not weights.is_relative_to(directory.resolve()):
        raise ValueError("ClearVoice checkpoint pointer escapes its checkpoint directory")
    return weights.is_file() and weights.stat().st_size > 0


def load_checkpoint_strict(torch, model, checkpoint_path, model_key=None):
    """Require every inference tensor instead of retaining random initial values.

    ClearVoice 0.1.2's loader silently skips missing or shape-mismatched tensors.
    Its official checkpoint is a plain tensor/optimizer mapping supported by
    PyTorch's restricted weights-only loader; no unrestricted pickle is needed.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("ClearVoice checkpoint must contain a state dictionary")
    pretrained = checkpoint.get(model_key, checkpoint)
    if not isinstance(pretrained, dict):
        raise ValueError("ClearVoice checkpoint has no model state dictionary")
    matched = {}
    failures = []
    for key, value in model.state_dict().items():
        candidates = (key, key.removeprefix("module."), "module." + key)
        source = next((pretrained[name] for name in candidates if name in pretrained), None)
        if source is None or not hasattr(source, "shape") or tuple(source.shape) != tuple(value.shape):
            failures.append(key)
        else:
            matched[key] = source
    if failures:
        raise ValueError(
            f"ClearVoice checkpoint is missing or has incompatible shapes for {len(failures)} "
            f"model tensors: {', '.join(failures[:5])}. Refusing partially random weights."
        )
    model.load_state_dict(matched, strict=True)


@contextmanager
def strict_checkpoint_loading(torch, networks):
    # Each backend executes in its own subprocess. Keep this override scoped to
    # construction so upstream code cannot silently accept an incompatible cache.
    previous = networks.SpeechModel._load_model

    def load(instance, model, checkpoint_path, model_key=None):
        load_checkpoint_strict(torch, model, checkpoint_path, model_key)

    networks.SpeechModel._load_model = load
    try:
        yield
    finally:
        networks.SpeechModel._load_model = previous


class Separator:
    def run(self, payload: dict, options: dict) -> dict:
        torch = dependency("torch", "ClearVoice")
        np = dependency("numpy", "ClearVoice")
        sf = dependency("soundfile", "ClearVoice")
        clearvoice = dependency("clearvoice", "ClearVoice")
        networks = dependency("clearvoice.networks", "ClearVoice")
        hub = dependency("huggingface_hub", "ClearVoice")
        device = torch_device(torch, options.get("device", "cuda:0"))
        model_name = options.get("model", "MossFormer2_SS_16K")
        if model_name != "MossFormer2_SS_16K":
            raise ValueError("The clearvoice adapter supports the two-speaker MossFormer2_SS_16K model.")
        cache = Path(options.get("cache_dir", "~/.cache/speaker-lab/clearvoice")).expanduser().resolve()
        directory = cache / "checkpoints" / model_name
        directory.mkdir(parents=True, exist_ok=True)
        if not checkpoint_ready(directory):
            hub.snapshot_download(repo_id=f"alibabasglab/{model_name}", local_dir=str(directory))
        if not checkpoint_ready(directory):
            raise RuntimeError(f"ClearVoice checkpoint is incomplete in {directory}; refusing untrained weights.")
        # ClearVoice config uses a relative checkpoint directory; isolate it from cwd.
        with working_directory(cache), strict_checkpoint_loading(torch, networks):
            model = clearvoice.ClearVoice(task="speech_separation", model_names=[model_name])
        if len(model.models) != 1:
            raise RuntimeError("Expected exactly one ClearVoice model")
        network = model.models[0]
        if Path(network.args.checkpoint_dir).is_absolute():
            loaded_directory = Path(network.args.checkpoint_dir).resolve()
        else:
            loaded_directory = (cache / network.args.checkpoint_dir).resolve()
        if loaded_directory != directory.resolve():
            raise RuntimeError("ClearVoice changed its checkpoint layout; update the adapter.")
        network.device = device
        network.args.use_cuda = int(device.type == "cuda")
        network.model.to(device)
        network.model.eval()
        # Upstream's 2-second internal chunking does not align speaker permutations.
        # We keep every external job within one inference pass and match it ourselves.
        network.args.one_time_decode_length = 20.0
        network.args.decode_window = min(float(network.args.decode_window), 2.0)
        results = []
        for job in payload["jobs"]:
            path = str(absolute_path(job["input_path"]))
            audio = read_audio(path)
            if len(audio) / 16000 > 20.0:
                raise ValueError("ClearVoice jobs must be at most 20 seconds to avoid internal speaker swaps.")
            if float(np.max(np.abs(audio))) < 1e-8:
                raise ValueError("Cannot separate a silent clip")
            output = absolute_path(job["output_dir"], exists=False)
            output.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                # The file API normalizes its input and fails to restore the
                # original gain for separation lists. The public NumPy API also
                # avoids its pydub/FFmpeg conversion of our floating-point WAV.
                separated = model(input_path=audio[np.newaxis, :])
            separated = np.asarray(separated, dtype="float32")
            if separated.shape != (2, 1, len(audio)):
                raise RuntimeError(
                    f"ClearVoice NumPy API must return (2, 1, {len(audio)}); received {separated.shape}"
                )
            sources = []
            for index, source in enumerate(separated):
                waveform = source[0]
                if not np.isfinite(waveform).all():
                    raise RuntimeError("ClearVoice returned non-finite samples")
                # The upstream PCM writer casts without saturation; write its public
                # returned arrays as floating WAV so peaks >1 do not wrap or clip.
                source_path = output / f"source_{index}.wav"
                sf.write(str(source_path), waveform, 16000, subtype="FLOAT")
                sources.append(source_path)
            results.append({"id": job["id"], "sources": [str(path) for path in sources]})
        return {
            "jobs": results, "model": model_name,
            "amplitude_scaling": (
                "NumPy input preserves original input scale; upstream ClearVoice normalizes "
                "each source RMS to the mixture RMS; original per-speaker loudness is not preserved"
            ),
        }
