"""Lazy model adapters. Importing this package never imports a GPU framework."""

from importlib import import_module
from typing import Any, Protocol


class Backend(Protocol):
    def run(self, payload: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]: ...


BUILTINS = {
    "diarize": {"pyannote": "speaker_lab.backends.pyannote:Diarizer"},
    "separate": {
        "clearvoice": "speaker_lab.backends.clearvoice:Separator",
        "sepformer": "speaker_lab.backends.sepformer:Separator",
    },
    "match": {"pyannote_embedding": "speaker_lab.backends.pyannote:SpeakerMatcher"},
    "transcribe": {"qwen": "speaker_lab.backends.qwen:Transcriber"},
}


def load_backend(stage: str, name: str) -> Backend:
    """Load a built-in alias or trusted, user-configured ``module:Class`` plugin."""
    if not isinstance(stage, str) or stage not in BUILTINS:
        raise ValueError(f"Unknown stage: {stage!r}; expected one of {list(BUILTINS)}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Backend name must be a non-empty alias or module:Class string.")
    target = BUILTINS[stage].get(name, name)
    if ":" not in target:
        raise ValueError(f"Unknown {stage} backend {name!r}; use an alias or module:Class")
    module, class_name = target.split(":", 1)
    if not module or not all(part.isidentifier() for part in module.split(".")) or not class_name.isidentifier():
        raise ValueError(f"Invalid backend {target!r}; expected module:Class")
    backend = getattr(import_module(module), class_name)()
    if not callable(getattr(backend, "run", None)):
        raise TypeError(f"Backend {target!r} must implement run(payload, options)")
    return backend
