#!/usr/bin/env python3
"""Exercise installed ClearVoice model code with random weights, without downloads.

This validates imports, a real MossFormer2 forward pass, and the decoder's output
contract. It does not measure speech separation quality or validate pretrained
weights. Use the separator environment, for example:

    .venvs/separate/bin/python scripts/smoke_models.py --device cuda:0
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--checkpoint-shapes", type=Path,
        help="Optional audit JSON mapping official checkpoint keys to shapes.",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from clearvoice import ClearVoice
    from clearvoice.networks import SpeechModel
    from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
    from clearvoice.utils.decode import decode_one_audio_mossformer2_ss_16k

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run scripts/doctor.py first.")
    torch.manual_seed(0)
    settings = SimpleNamespace(
        encoder_embedding_dim=32, mossformer_sequence_dim=32,
        num_mossformer_layer=1, encoder_kernel_size=16, num_spks=2,
        sampling_rate=16000, decode_window=0.1, one_time_decode_length=20.0,
        network="MossFormer2_SS_16K",
    )
    model = MossFormer2_SS_16K(settings).model.to(device).eval()
    # Attach the real network without its pretrained checkpoint constructor.
    network = SpeechModel.__new__(SpeechModel)
    network.model, network.device, network.args = model, device, settings
    interface = ClearVoice.__new__(ClearVoice)
    interface.models = [network]
    audio = np.random.default_rng(0).normal(0, 0.05, (1, 1600)).astype("float32")
    with torch.inference_mode():
        raw = model(torch.from_numpy(audio).to(device))
        decoded = decode_one_audio_mossformer2_ss_16k(model, device, audio, settings)
        returned = interface(input_path=audio)
    if len(raw) != 2 or any(tuple(source.shape) != (1, 1600) for source in raw):
        raise RuntimeError(f"Unexpected MossFormer2 output shapes: {[tuple(x.shape) for x in raw]}")
    if len(decoded) != 2 or any(source.ndim != 1 or len(source) < 1600 for source in decoded):
        raise RuntimeError("Unexpected ClearVoice decoder output shape")
    if not all(np.isfinite(source).all() for source in decoded):
        raise RuntimeError("ClearVoice decoder returned non-finite samples")
    if returned.shape != (2, 1, 1600) or not np.isfinite(returned).all():
        raise RuntimeError("ClearVoice public NumPy API changed its output shape")
    np.testing.assert_allclose(
        np.sqrt(np.mean(returned ** 2, axis=(1, 2))),
        np.sqrt(np.mean(audio ** 2)), rtol=1e-4,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"PASS: actual ClearVoice MossFormer2 forward + NumPy API + decoder on {device} (random weights)")

    if args.checkpoint_shapes:
        expected = json.loads(args.checkpoint_shapes.read_text(encoding="utf-8"))
        settings.encoder_embedding_dim = settings.mossformer_sequence_dim = 512
        settings.num_mossformer_layer = 24
        # Meta tensors verify the full architecture without allocating its weights.
        with torch.device("meta"):
            full = MossFormer2_SS_16K(settings).model
        actual = {key: list(value.shape) for key, value in full.state_dict().items()}
        differences = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        if differences:
            raise RuntimeError(f"Official checkpoint architecture mismatch: {differences[:10]}")
        print(f"PASS: full model matches {len(expected)} official checkpoint tensor shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
