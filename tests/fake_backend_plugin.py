"""Synthetic subprocess fixtures, never registered as production backends."""
from pathlib import Path

import numpy as np
import soundfile as sf


class Diarizer:
    def run(self, payload, options):
        return {"turns": [{"start": 0, "end": 4, "speaker": "first"},
                           {"start": 2, "end": 6, "speaker": "second"}], "model": "test-only"}


class Separator:
    def run(self, payload, options):
        rows = []
        for job in payload["jobs"]:
            values = np.arange(job["input_start"], job["input_end"])
            time = values / 16000
            first = (0.1 * np.sin(2 * np.pi * 330 * time) * (time < 4)).astype("float32")
            second = (0.1 * np.sin(2 * np.pi * 660 * time) * (time >= 2)).astype("float32")
            directory = Path(job["output_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            paths = []
            for index, source in enumerate((second, first)):
                path = directory / f"source_{index}.wav"
                sf.write(path, source, 16000, subtype="FLOAT")
                paths.append(str(path))
            rows.append({"id": job["id"], "sources": paths})
        return {"jobs": rows, "model": "test-only"}


class Matcher:
    def run(self, payload, options):
        return {"jobs": [{"id": job["id"], "speakers": ["SPEAKER_01", "SPEAKER_00"],
                          "margin": 1.0, "uncertain": False} for job in payload["jobs"]], "model": "test-only"}


class Transcriber:
    def run(self, payload, options):
        return {"segments": [{"job_id": job["id"], "speaker": job["speaker"], "start": job["start"],
                              "end": job["end"], "text": "测试文字 " + job["speaker"], "words": []}
                             for job in payload["jobs"]], "model": "test-only"}
