"""Adapter contract tests without downloading weights or requiring a GPU."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from speaker_lab.backends import load_backend
from speaker_lab.backends.clearvoice import (
    checkpoint_ready, load_checkpoint_strict, strict_checkpoint_loading, Separator as ClearVoiceSeparator,
)
from speaker_lab.backends.pyannote import assign_speakers, Diarizer
from speaker_lab.backends.qwen import aligned_words, Transcriber
from speaker_lab.backends.sepformer import Separator as SepFormerSeparator
from speaker_lab.worker import main as worker_main


class AdapterTests(unittest.TestCase):
    def test_builtin_registry_does_not_import_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import speaker_lab.backends; import speaker_lab.worker; "
             "import sys; assert 'torch' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRaisesRegex(ValueError, "Unknown", load_backend, "diarize", "missing")

    def test_assignment_can_swap_sources_and_flag_ambiguity(self):
        assigned = assign_speakers([[0.1, 0.95], [0.90, 0.1]], ["A", "B"])
        self.assertEqual(assigned["speakers"], ["B", "A"])
        self.assertFalse(assigned["uncertain"])
        self.assertAlmostEqual(assigned["margin"], 0.825)
        tied = assign_speakers([[0.9, 0.9], [0.9, 0.9]], ["A", "B"])
        self.assertTrue(tied["uncertain"])
        self.assertEqual(tied["margin"], 0)
        for invalid in (float("nan"), float("inf"), -1, 2.1):
            self.assertRaisesRegex(ValueError, "finite", assign_speakers, [[1, 0], [0, 1]], ["A", "B"], invalid)

    def test_alignment_keeps_real_times_and_applies_original_offset(self):
        result = SimpleNamespace(time_stamps=SimpleNamespace(items=[
            SimpleNamespace(text="你好", start_time=0.2, end_time=0.8),
            SimpleNamespace(text="。", start_time=0.8, end_time=1.001),
        ]))
        words = aligned_words(result, offset=30, duration=1)
        self.assertEqual(words, [
            {"text": "你好", "start": 30.2, "end": 30.8},
            {"text": "。", "start": 30.8, "end": 31},
        ])
        self.assertEqual(aligned_words(SimpleNamespace(time_stamps=None), 30, 1), [])

    def test_pyannote_uses_overlap_annotation_and_exact_two_speakers(self):
        calls = {}
        segment = lambda a, b: SimpleNamespace(start=a, end=b)

        class Annotation:
            def itertracks(self, yield_label):
                assert yield_label
                return iter([(segment(0, 3), 0, "A"), (segment(1, 4), 1, "B")])

        class Pipeline:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls["init"] = kwargs
                return cls()

            def to(self, device):
                pass

            def __call__(self, audio, **kwargs):
                calls["run"] = kwargs
                return SimpleNamespace(speaker_diarization=Annotation(), exclusive_speaker_diarization=None)

        fake_torch = SimpleNamespace(inference_mode=nullcontext)
        modules = {"torch": fake_torch, "pyannote.audio": SimpleNamespace(Pipeline=Pipeline)}
        with patch("speaker_lab.backends.pyannote.dependency", side_effect=lambda name, _: modules[name]), \
             patch("speaker_lab.backends.pyannote.torch_device", return_value="cpu"), \
             patch("speaker_lab.backends.pyannote.audio_dict", return_value={}), \
             patch.dict(os.environ, {"TEST_HF_TOKEN": "test-token"}):
            result = Diarizer().run({"audio_path": "/audio.wav"}, {"token_env": "TEST_HF_TOKEN"})
        self.assertEqual(calls["init"], {"token": "test-token"})
        self.assertEqual(calls["run"], {"num_speakers": 2})
        self.assertEqual(result["turns"][0]["end"], 3)
        self.assertEqual(result["turns"][1]["start"], 1)

    def test_qwen_calls_documented_api_with_aligner_and_offsets(self):
        calls = {}

        class Qwen:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls["init"] = kwargs
                return cls()

            def transcribe(self, **kwargs):
                calls["transcribe"] = kwargs
                return [SimpleNamespace(text="你好", time_stamps=SimpleNamespace(items=[
                    SimpleNamespace(text="你好", start_time=0.25, end_time=0.8)
                ]))]

        fake_torch = SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16")
        modules = {"torch": fake_torch, "qwen_asr": SimpleNamespace(Qwen3ASRModel=Qwen)}
        with patch("speaker_lab.backends.qwen.dependency", side_effect=lambda name, _: modules[name]), \
             patch("speaker_lab.backends.qwen.torch_device", return_value="cpu"), \
             patch("speaker_lab.backends.qwen.absolute_path", return_value=Path("/clip.wav")), \
             patch("speaker_lab.backends.qwen.read_audio", return_value=[0] * 16000):
            result = Transcriber().run({"jobs": [
                {"id": "clip-1", "audio_path": "/clip.wav", "speaker": "A", "start": 20}
            ]}, {})
        self.assertEqual(calls["init"]["forced_aligner"], "Qwen/Qwen3-ForcedAligner-0.6B")
        self.assertTrue(calls["transcribe"]["return_time_stamps"])
        self.assertEqual(result["segments"][0]["start"], 20.25)
        self.assertEqual(result["segments"][0]["end"], 20.8)
        self.assertEqual(result["segments"][0]["job_id"], "clip-1")

    def test_incomplete_clearvoice_checkpoint_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertFalse(checkpoint_ready(directory))
            (directory / "last_best_checkpoint").write_text("missing.pt\n", encoding="utf-8")
            self.assertFalse(checkpoint_ready(directory))
            (directory / "missing.pt").write_bytes(b"weights")
            self.assertTrue(checkpoint_ready(directory))
            (directory / "last_best_checkpoint").write_text("../outside.pt", encoding="utf-8")
            self.assertRaisesRegex(ValueError, "escapes", checkpoint_ready, directory)

    def test_clearvoice_prefetches_checkpoint_and_bypasses_internal_chunking(self):
        import numpy as np

        calls = {}
        network = SimpleNamespace(
            args=SimpleNamespace(checkpoint_dir="checkpoints/MossFormer2_SS_16K", decode_window=2),
            model=SimpleNamespace(to=lambda device: None, eval=lambda: None),
        )

        class ClearVoice:
            def __init__(self, **kwargs):
                calls["init"] = kwargs
                self.models = [network]

            def __call__(self, **kwargs):
                calls["inference"] = kwargs
                return np.stack([np.ones((1, 16000)) * 1.2, np.ones((1, 16000)) * 0.2])

        def write(path, samples, rate, **kwargs):
            calls.setdefault("write", []).append({"path": path, "samples": samples, "rate": rate, **kwargs})

        def snapshot_download(repo_id, local_dir):
            calls["download"] = repo_id
            directory = Path(local_dir)
            (directory / "last_best_checkpoint").write_text("weights.pt\n", encoding="utf-8")
            (directory / "weights.pt").write_bytes(b"checkpoint")

        modules = {
            "torch": SimpleNamespace(inference_mode=nullcontext), "numpy": np,
            "soundfile": SimpleNamespace(write=write),
            "clearvoice": SimpleNamespace(ClearVoice=ClearVoice),
            "clearvoice.networks": SimpleNamespace(SpeechModel=type("SpeechModel", (), {"_load_model": lambda self: None})),
            "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
        }
        with TemporaryDirectory() as temporary, \
             patch("speaker_lab.backends.clearvoice.dependency", side_effect=lambda name, _: modules[name]), \
             patch("speaker_lab.backends.clearvoice.torch_device", return_value=SimpleNamespace(type="cpu")), \
             patch("speaker_lab.backends.clearvoice.absolute_path", side_effect=lambda value, **kwargs: Path(value)), \
             patch("speaker_lab.backends.clearvoice.read_audio", return_value=np.ones(16000) * 0.1):
            result = ClearVoiceSeparator().run({"jobs": [{
                "id": "overlap", "input_path": str(Path(temporary) / "input.wav"),
                "output_dir": str(Path(temporary) / "output"),
            }]}, {"cache_dir": str(Path(temporary) / "cache")})
        self.assertEqual(calls["download"], "alibabasglab/MossFormer2_SS_16K")
        np.testing.assert_allclose(calls["inference"]["input_path"], np.ones((1, 16000)) * 0.1)
        self.assertNotIn("online_write", calls["inference"])
        self.assertEqual(network.args.one_time_decode_length, 20)
        self.assertTrue(calls["write"][0]["path"].endswith("source_0.wav"))
        self.assertEqual(calls["write"][0]["subtype"], "FLOAT")
        self.assertGreater(float(calls["write"][0]["samples"].max()), 1)
        self.assertEqual(len(result["jobs"][0]["sources"]), 2)

    def test_clearvoice_rejects_missing_or_wrong_shape_tensors(self):
        import numpy as np

        expected = {"encoder.weight": np.zeros((4, 1, 16)), "decoder.weight": np.zeros((4, 1, 16))}
        calls = {}
        model = SimpleNamespace(
            state_dict=lambda: expected,
            load_state_dict=lambda state, **kwargs: calls.update(state=state, **kwargs),
        )
        torch = SimpleNamespace(load=lambda path, **kwargs: {"model": {"encoder.weight": expected["encoder.weight"]}})
        self.assertRaisesRegex(ValueError, "partially random", load_checkpoint_strict, torch, model, "weights.pt", "model")
        torch.load = lambda path, **kwargs: {"model": {**expected, "decoder.weight": np.zeros((2, 1, 16))}}
        self.assertRaisesRegex(ValueError, "incompatible shapes", load_checkpoint_strict, torch, model, "weights.pt", "model")
        self.assertFalse(calls)

    def test_clearvoice_loads_weights_only_and_accepts_data_parallel_prefix(self):
        import numpy as np

        calls = {}
        expected = {"encoder.weight": np.zeros((4, 1, 16))}
        pretrained = {"module.encoder.weight": np.ones((4, 1, 16))}

        def load(path, **kwargs):
            calls["torch_load"] = kwargs
            return {"model": pretrained, "optimizer": {}}

        model = SimpleNamespace(
            state_dict=lambda: expected,
            load_state_dict=lambda state, **kwargs: calls.update(state=state, **kwargs),
        )
        load_checkpoint_strict(SimpleNamespace(load=load), model, "weights.pt", "model")
        self.assertEqual(calls["torch_load"], {"map_location": "cpu", "weights_only": True})
        self.assertTrue(calls["strict"])
        self.assertIs(calls["state"]["encoder.weight"], pretrained["module.encoder.weight"])

    def test_clearvoice_scoped_loader_restores_upstream_after_failure(self):
        class SpeechModel:
            def _load_model(self):
                pass

        original = SpeechModel._load_model
        networks = SimpleNamespace(SpeechModel=SpeechModel)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with strict_checkpoint_loading(SimpleNamespace(), networks):
                self.assertIsNot(SpeechModel._load_model, original)
                raise RuntimeError("failed")
        self.assertIs(SpeechModel._load_model, original)

    def test_worker_serializes_failure_to_response(self):
        with TemporaryDirectory() as temporary:
            request, response = Path(temporary) / "request.json", Path(temporary) / "response.json"
            request.write_text(json.dumps({"stage": "diarize", "backend": "missing"}), encoding="utf-8")
            self.assertEqual(worker_main(["--request", str(request), "--response", str(response)]), 1)
            data = json.loads(response.read_text(encoding="utf-8"))
            self.assertFalse(data["ok"])
            self.assertIn("Unknown diarize backend", data["error"])
            self.assertIn("ValueError", data["traceback"])

    def test_sepformer_uses_separate_persistent_cache_for_each_model(self):
        calls = []

        class Sepformer:
            @classmethod
            def from_hparams(cls, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(hparams=SimpleNamespace(sample_rate=8000))

        modules = {
            "torch": SimpleNamespace(), "numpy": SimpleNamespace(),
            "soundfile": SimpleNamespace(), "scipy.signal": SimpleNamespace(),
            "speechbrain.inference.separation": SimpleNamespace(SepformerSeparation=Sepformer),
        }
        with TemporaryDirectory() as temporary, \
             patch("speaker_lab.backends.sepformer.dependency", side_effect=lambda name, _: modules[name]), \
             patch("speaker_lab.backends.sepformer.torch_device", return_value="cpu"):
            for model_name in ("speechbrain/sepformer-wsj02mix", "speechbrain/sepformer-libri2mix", "speechbrain/sepformer-wsj02mix"):
                SepFormerSeparator().run({"jobs": []}, {"model": model_name, "cache_dir": temporary})
            self.assertNotEqual(calls[0]["savedir"], calls[1]["savedir"])
            self.assertEqual(calls[0]["savedir"], calls[2]["savedir"])
            for call in calls:
                self.assertEqual(Path(call["savedir"]).parent, Path(temporary).resolve())
                self.assertTrue(Path(call["savedir"]).is_dir())


if __name__ == "__main__":
    unittest.main()
