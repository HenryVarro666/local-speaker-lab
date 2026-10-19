"""Exercise the delivered source archive and installed distribution off checkout."""

import importlib.util
from email import message_from_bytes
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_source_archive_builds_and_installs_without_checkout_imports(tmp_path):
    """No build isolation, dependency fetch, GPU package, or model is needed."""
    spec = importlib.util.spec_from_file_location("distribution_package", ROOT / "scripts/package.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    archive = package.build_archive(tmp_path / "source.zip")
    source_parent = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(source_parent)
    source = source_parent / "local-speaker-lab"
    output = tmp_path / "distributions"
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PIP_NO_INDEX="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1")

    def run(*command):
        completed = subprocess.run(command, cwd=tmp_path, env=environment, text=True,
                                   encoding="utf-8", capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
        return completed.stdout

    # Build's default path creates an sdist and then a wheel from that sdist.
    # This catches files available in checkout but missing from either delivery.
    run(sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(source))
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1
    assert len(list(output.glob("*.tar.gz"))) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        expected = {path.relative_to(source / "src").as_posix()
                    for path in (source / "src/speaker_lab").rglob("*.py")}
        assert expected <= set(wheel.namelist())
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        assert message_from_bytes(wheel.read(metadata_path))["License-Expression"] == "MIT"
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            packaged = next(name for name in wheel.namelist()
                            if name.endswith(".dist-info/licenses/" + filename))
            assert wheel.read(packaged) == (source / filename).read_bytes()

    # A new environment has no editable install and no inherited site packages.
    # Installing just the local wheel proves entry points and lazy commands work
    # independently of scientific libraries and optional model dependencies.
    installed = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True).create(installed)
    binary = installed / ("Scripts" if os.name == "nt" else "bin")
    python = binary / ("python.exe" if os.name == "nt" else "python")
    run(str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0]))
    expected_version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    report = json.loads(run(str(python), "-I", "-c", """
import importlib.metadata
import json
from pathlib import Path
import sys
import speaker_lab
import speaker_lab.backends
import speaker_lab.worker
print(json.dumps({
    'version': importlib.metadata.version('local-speaker-lab'),
    'module': str(Path(speaker_lab.__file__).resolve()),
    'heavy_imports': sorted(set(sys.modules) & {'torch', 'transformers', 'gradio', 'numpy', 'scipy'}),
}))
"""))
    assert report["version"] == expected_version
    assert Path(report["module"]).is_relative_to(installed)
    assert report["heavy_imports"] == []
    for name in ("speaker-lab", "speaker-lab-ui"):
        executable = binary / (name + ".exe" if os.name == "nt" else name)
        assert "usage:" in run(str(executable), "--help")
    assert "--request" in run(str(python), "-I", "-m", "speaker_lab.worker", "--help")
