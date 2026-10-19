"""Native-stack diagnostics must fail before falsely reporting GPU readiness."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location("speaker_lab_doctor", Path(__file__).resolve().parents[1] / "scripts/doctor.py")
doctor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(doctor)


class ArrayTensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return ArrayTensor(self.value.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_audio_probe_uses_actual_float_wav_format():
    # Executes installed libsndfile; no model packages or downloads required.
    doctor.probe_audio_decode()


@pytest.mark.parametrize("bad_shape", [False, True])
def test_torchcodec_probe_decodes_and_checks_real_encoded_audio(monkeypatch, bad_shape):
    import io
    import soundfile as sf

    class Decoder:
        def __init__(self, encoded, sample_rate, num_channels):
            assert sample_rate == 16000 and num_channels == 1
            with sf.SoundFile(io.BytesIO(encoded)) as source:
                assert source.subtype == "FLOAT"
                self.samples = source.read(dtype="float32")[None, :]

        def get_all_samples(self):
            samples = self.samples[:, :-1] if bad_shape else self.samples
            return SimpleNamespace(sample_rate=16000, data=ArrayTensor(samples))

    monkeypatch.setitem(sys.modules, "torchcodec.decoders", SimpleNamespace(AudioDecoder=Decoder))
    if bad_shape:
        with pytest.raises(RuntimeError, match="shape or sample rate"):
            doctor.probe_audio_decode(torchcodec=True)
    else:
        doctor.probe_audio_decode(torchcodec=True)


def test_numpy_bridge_checks_array_contents():
    doctor.probe_numpy_bridge(SimpleNamespace(from_numpy=lambda array: ArrayTensor(array)))
    with pytest.raises(AssertionError):
        doctor.probe_numpy_bridge(SimpleNamespace(from_numpy=lambda array: ArrayTensor(np.zeros_like(array))))


def test_successful_imports_do_not_hide_native_decode_failure(monkeypatch, capsys):
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "test")

    def fail_decode(**kwargs):
        raise RuntimeError("cannot load libtorchcodec: missing libavutil")

    monkeypatch.setattr(doctor, "probe_audio_decode", fail_decode)
    assert doctor.probe("diarize") == 1
    assert "missing libavutil" in capsys.readouterr().err


def test_numpy_abi_failure_is_a_failed_probe_before_cuda(monkeypatch, capsys):
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(doctor.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(doctor, "probe_audio_decode", lambda **kwargs: None)

    def broken_bridge(array):
        raise RuntimeError("Numpy is not available")

    torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda="12.8"), from_numpy=broken_bridge)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(__version__="test"))
    assert doctor.probe("separate") == 1
    assert "Numpy is not available" in capsys.readouterr().err
