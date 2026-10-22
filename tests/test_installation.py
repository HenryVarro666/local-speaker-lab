"""Dependency invariants that a successful resolver alone cannot enforce."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]


def requirements(path: Path) -> dict[str, Requirement]:
    return {
        canonicalize_name(requirement.name): requirement
        for line in path.read_text(encoding="utf-8").splitlines()
        if (clean := line.strip()) and not clean.startswith("#")
        for requirement in [Requirement(clean)]
    }


def test_diarize_requires_cpu_codec_even_when_cuda_codec_is_installed():
    # ==0.7.0 also accepts +cu128; === is needed when repairing an old env.
    requirement = requirements(ROOT / "requirements/diarize.txt")["torchcodec"]
    assert "0.7.0" in requirement.specifier
    assert "0.7.0+cu128" not in requirement.specifier
    locked = requirements(ROOT / "requirements/locks/diarize-linux-py312.txt")["torchcodec"]
    assert "0.7.0" in locked.specifier
    assert "+cu128" not in str(locked.specifier)


def test_cpu_codec_keeps_gpu_torch_and_legacy_torchaudio():
    pinned = requirements(ROOT / "requirements/locks/diarize-linux-py312.txt")
    assert "2.8.0+cu128" in pinned["torch"].specifier
    assert "2.8.0+cu128" in pinned["torchaudio"].specifier
    # A CUDA runtime package must not be silently replaced by a CPU torch build.
    assert "2.8.0+cpu" not in pinned["torch"].specifier


def pinned_version(requirement: Requirement) -> Version:
    pins = list(requirement.specifier)
    assert len(pins) == 1 and pins[0].operator in {"==", "==="}
    assert "*" not in pins[0].version
    return Version(pins[0].version)


def test_direct_runtime_inputs_are_represented_by_each_lock():
    """An edited top-level input must never silently disagree with its snapshot."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = [Requirement(value) for value in metadata["project"]["dependencies"]]
    constraints = requirements(ROOT / "requirements/constraints-torch.txt")
    for stage in ["app", "diarize", "separate", "asr", "separate-sb"]:
        direct = requirements(ROOT / f"requirements/{stage}.txt")
        locked = requirements(ROOT / f"requirements/locks/{stage}-linux-py312.txt")
        for name, requirement in direct.items():
            assert name in locked, (stage, name)
            assert pinned_version(locked[name]) in requirement.specifier, (stage, name)
        for requirement in core:
            # These snapshots target Linux even when checked on a Windows host.
            if requirement.marker and not requirement.marker.evaluate({"sys_platform": "linux"}):
                continue
            name = canonicalize_name(requirement.name)
            assert pinned_version(locked[name]) in requirement.specifier, (stage, name)
        for name, constraint in constraints.items():
            if name in locked:
                assert pinned_version(locked[name]) in constraint.specifier, (stage, name)


def test_build_tools_have_one_exact_backend_version():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    backend = Requirement(metadata["build-system"]["requires"][0])
    bootstrap = requirements(ROOT / "requirements/bootstrap.txt")
    assert pinned_version(backend) == pinned_version(bootstrap["setuptools"])
    for stage in ["diarize", "separate", "asr", "separate-sb"]:
        locked = requirements(ROOT / f"requirements/locks/{stage}-linux-py312.txt")
        assert pinned_version(locked["setuptools"]) == pinned_version(backend)
    for requirement in bootstrap.values():
        pinned_version(requirement)


def test_host_and_model_numpy_abi_boundaries_stay_separate():
    major_versions = {}
    for stage in ["app", "diarize", "separate", "asr", "separate-sb"]:
        locked = requirements(ROOT / f"requirements/locks/{stage}-linux-py312.txt")
        major_versions[stage] = pinned_version(locked["numpy"]).major
        if stage == "app":
            assert not ({"torch", "torchaudio", "clearvoice", "speechbrain", "qwen-asr"} & locked.keys())
    assert major_versions == {"app": 2, "diarize": 2, "separate": 1, "asr": 1, "separate-sb": 1}
