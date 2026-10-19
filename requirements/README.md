# Dependency maintenance

The installer targets Ubuntu 24.04 x86_64 and CPython 3.12. Each worker has its
own environment because ClearVoice / Qwen need NumPy 1.x while pyannote 4 uses
NumPy 2.x. Model inference uses the official PyTorch 2.8.0 / CUDA 12.8 wheels.
Do not upgrade TorchAudio independently: SpeechBrain 1.0.3 still calls the legacy
`list_audio_backends` API, which was removed in TorchAudio 2.9.

`bootstrap.txt` pins pip, setuptools and wheel separately from runtime packages.
The Ubuntu installer applies it on every run, then installs the local editable
package with `--no-build-isolation --no-deps` using that pinned backend.
`pyproject.toml` also pins the same setuptools version, so a normal isolated
`pip install .` uses the identical build backend. Update these pins together.
The installer also passes `--build-constraint requirements/bootstrap.txt` to
third-party runtime installs, so source-only packages use the same setuptools
and wheel pins in their isolated build environments. This requires pip 25.3,
which bootstrap installs before runtime dependencies.
The runtime snapshots pin versions, not artifact hashes; they reproduce the
dependency graph rather than provide a fully hash-verified offline wheelhouse.

TorchCodec 0.7 is paired with torch 2.8. Its **CPU** wheel is intentional: only
audio decoding is used, and this avoids CUDA video decoding runtime dependencies.
`diarize.txt` uses arbitrary equality (`===0.7.0`) so a CUDA local version such as
`0.7.0+cu128` cannot satisfy the requirement. PyPI's Linux cp312 wheel has the same
SHA-256 as the official PyTorch CPU index wheel:
`3cd23c3296c9b071d56bb2c534a6a98275d65c1a6a7213cdb72a26ec9f9d2fd8`.

To regenerate snapshots, install `uv==0.13.1` in a disposable tooling environment,
then run from the project root:

```bash
uv pip compile requirements/app.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
  --output-file requirements/locks/app-linux-py312.txt

for stage in diarize separate asr separate-sb; do
  uv pip compile "requirements/$stage.txt" \
    --constraint requirements/constraints-torch.txt \
    --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    --output-file "requirements/locks/$stage-linux-py312.txt"
done
```

This command retains existing pins unless the inputs force a change; use
`--upgrade-package PACKAGE` for intentional upgrades. The extra index is only the
official PyTorch wheel service. Review the diff for unexpected transitive
upgrades, especially NumPy, torch, torchaudio, torchvision and TorchCodec.
The installed runtime uses the normal PyPI index for model dependencies after
preinstalling the GPU torch wheels. Do not add the CUDA index to every pip command.

Resolution is a packaging check. After any dependency change, reinstall on the
Ubuntu machine, run `scripts/doctor.py` (native WAV decode, NumPy/PyTorch ABI,
CUDA kernels), and run a real recording through every enabled model backend.
The source-only `python-speech-features` and `sox` packages
may be built into pure Python wheels during installation; no CUDA compilation
is required for the default model backends.

Primary compatibility references:

- [PyTorch historical wheels](https://pytorch.org/get-started/previous-versions/)
- [TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)
- [TorchCodec 0.7 PyPI artifacts](https://pypi.org/project/torchcodec/0.7.0/#files)
- [TorchAudio API removals](https://docs.pytorch.org/audio/stable/index.html)
- [pip build constraints](https://pip.pypa.io/en/stable/user_guide/#build-constraints)
