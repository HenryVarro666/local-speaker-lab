"""Diarization and two-way voice identity matching with pyannote.audio 4.x."""

import math
import os

from .common import audio_dict, dependency, torch_device


class Diarizer:
    def run(self, payload: dict, options: dict) -> dict:
        torch = dependency("torch", "diarization")
        pyannote = dependency("pyannote.audio", "diarization")
        model_name = options.get("model", "pyannote/speaker-diarization-community-1")
        device = torch_device(torch, options.get("device", "cuda:0"))
        token = os.environ.get(options.get("token_env", "HF_TOKEN"))
        pipeline = pyannote.Pipeline.from_pretrained(model_name, token=token)
        if pipeline is None:
            raise RuntimeError("Cannot load pyannote pipeline; accept the model conditions and set HF_TOKEN.")
        pipeline.to(device)
        with torch.inference_mode():
            output = pipeline(audio_dict(payload["audio_path"], torch), num_speakers=2)
        # Exclusive diarization removes simultaneous turns and would lose overlap.
        annotation = getattr(output, "speaker_diarization", output)
        turns = [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda turn: (turn["start"], turn["end"], turn["speaker"]))
        return {"turns": turns, "model": model_name}


def assign_speakers(similarities, speakers: list[str], min_margin: float = 0.05) -> dict:
    """Choose the better of the only two bijective source→speaker assignments.

    Margin is the difference between *mean* cosine scores of the assignments.
    It is an ambiguity indicator, not a calibrated probability/confidence score.
    """
    if not math.isfinite(min_margin) or not 0 <= min_margin <= 2:
        raise ValueError("min_margin must be finite and between 0 and 2")
    direct = float(similarities[0][0] + similarities[1][1]) / 2
    swapped = float(similarities[0][1] + similarities[1][0]) / 2
    order = speakers if direct >= swapped else list(reversed(speakers))
    margin = abs(direct - swapped)
    return {
        "speakers": list(order),
        "margin": margin,
        "uncertain": bool(margin < min_margin),
        "similarities": [[float(value) for value in row] for row in similarities],
    }


class SpeakerMatcher:
    def run(self, payload: dict, options: dict) -> dict:
        min_margin = float(options.get("min_margin", 0.05))
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 2:
            raise ValueError("min_margin must be finite and between 0 and 2")
        torch = dependency("torch", "diarization")
        np = dependency("numpy", "diarization")
        pyannote = dependency("pyannote.audio", "diarization")
        model_name = options.get("model", "pyannote/wespeaker-voxceleb-resnet34-LM")
        device = torch_device(torch, options.get("device", "cuda:0"))
        token = os.environ.get(options.get("token_env", "HF_TOKEN"))
        model = pyannote.Model.from_pretrained(model_name, token=token)
        if model is None:
            raise RuntimeError("Cannot load speaker embedding model; check model access and HF_TOKEN.")
        inference = pyannote.Inference(model, window="whole")
        inference.to(device)

        def embedding(path):
            with torch.inference_mode():
                vector = np.asarray(inference(audio_dict(path, torch)), dtype="float64").reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or norm < 1e-10:
                raise ValueError(f"Invalid voice embedding for {path}; use a longer, audible reference.")
            return vector / norm

        references = payload["references"]
        speakers = sorted(references)
        if len(speakers) != 2:
            raise ValueError("Speaker matching requires exactly two clean reference clips.")
        reference_vectors = np.stack([embedding(references[speaker]) for speaker in speakers])
        results = []
        for job in payload["jobs"]:
            if len(job["sources"]) != 2:
                raise ValueError("Speaker matching requires exactly two separated sources.")
            vectors = np.stack([embedding(path) for path in job["sources"]])
            assignment = assign_speakers(vectors @ reference_vectors.T, speakers, min_margin)
            results.append({"id": job["id"], **assignment})
        return {"jobs": results, "model": model_name}
