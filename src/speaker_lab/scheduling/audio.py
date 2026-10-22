"""PCM validation and optional CPU business observer; native audio stays 24 kHz."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import numpy as np


def validate_pcm(payload: bytes) -> np.ndarray:
    if not payload or len(payload) % 4 or len(payload) > 48000 * 4:
        raise ValueError("Expected at most two seconds of little-endian float32 PCM")
    pcm = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1.01:
        raise ValueError("Invalid PCM samples")
    return pcm


class UtteranceBuffer:
    """Bounded in-memory acoustic context for explicit model reconnects."""

    def __init__(self, max_samples=240000):
        self.max_samples = max_samples
        self.pre = deque(maxlen=6)
        self.current = deque()
        self.samples = 0
        self.active = False

    def feed(self, pcm):
        if not self.active and float(np.sqrt(np.mean(pcm**2))) > 0.008:
            self.active = True
            self.current.extend(self.pre)
            self.samples = sum(len(p) for p in self.current)
        if self.active:
            self.current.append(pcm.copy())
            self.samples += len(pcm)
            while self.samples > self.max_samples and self.current:
                self.samples -= len(self.current.popleft())
        self.pre.append(pcm.copy())

    def finish(self):
        result = (
            np.concatenate(list(self.current))
            if self.current
            else np.empty(0, dtype=np.float32)
        )
        self.current.clear()
        self.samples = 0
        self.active = False
        self.pre.clear()
        return result


class VoskObserver:
    """Streaming ASR observer for the tool controller, not PersonaPlex's transcript."""

    def __init__(self, model):
        from vosk import KaldiRecognizer

        self.decoder = KaldiRecognizer(model, 16000)
        self.speech = False
        self.quiet_samples = 0
        self.last_partial = ""

    @staticmethod
    def load(path: Path):
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        return Model(str(path))

    def feed(self, pcm: np.ndarray) -> list[dict]:
        from scipy.signal import resample_poly

        output = []
        # 24 kHz → 16 kHz is only for this observer, never the native speech stream.
        wave = resample_poly(pcm, 2, 3)
        encoded = (np.clip(wave, -1, 1) * 32767).astype("<i2").tobytes()
        loud = float(np.sqrt(np.mean(pcm**2))) > 0.008
        if loud:
            self.speech = True
            self.quiet_samples = 0
        elif self.speech:
            self.quiet_samples += len(pcm)
        final = self.decoder.AcceptWaveform(encoded)
        if final:
            result = json.loads(self.decoder.Result()).get("text", "")
        elif self.speech and self.quiet_samples >= 14400:
            result = json.loads(self.decoder.FinalResult()).get("text", "")
            self.speech = False
            self.quiet_samples = 0
        else:
            result = None
        if result is not None:
            self.last_partial = ""
            if result:
                output.append(
                    {"kind": "observer_final", "text": result, "source": "vosk"}
                )
        else:
            partial = json.loads(self.decoder.PartialResult()).get("partial", "")
            if partial and partial != self.last_partial:
                self.last_partial = partial
                output.append(
                    {"kind": "observer_partial", "text": partial, "source": "vosk"}
                )
        return output
