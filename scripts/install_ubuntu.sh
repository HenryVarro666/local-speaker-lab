#!/usr/bin/env bash
# Run explicitly with: bash scripts/install_ubuntu.sh
set -Eeuo pipefail
trap 'printf "Installation stopped at line %s. Existing environments and caches were preserved.\n" "$LINENO" >&2' ERR

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
APP_ONLY=0
WITH_SPEECHBRAIN=0
while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="${2:?--python requires an executable}"; shift 2 ;;
    --app-only) APP_ONLY=1; shift ;;
    --with-speechbrain) WITH_SPEECHBRAIN=1; shift ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/install_ubuntu.sh [--python python3.12] [--app-only] [--with-speechbrain]' \
        'Creates isolated local venvs. Does not install drivers, run sudo, pull model weights or start Ollama.'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == Linux ]] || { printf 'This installer targets Ubuntu Linux.\n' >&2; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { printf 'The locked environments target Ubuntu x86_64.\n' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf 'Missing %s. See README prerequisites.\n' "$PYTHON_BIN" >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the pinned worker dependencies"'
if (( ! APP_ONLY )); then
  command -v ffmpeg >/dev/null || { printf 'Missing ffmpeg. Install the README system packages first.\n' >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is unavailable. Install a Blackwell-capable NVIDIA driver first.\n' >&2; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

make_env() {
  local name="$1"
  local environment=".venvs/$name" environment_python=".venvs/$name/bin/python"
  # venv writes pyvenv.cfg before setting up Python and ensurepip. An interrupted
  # first run can therefore leave that file without a usable interpreter/pip.
  # Check an existing interpreter before repairing anything: a working 3.11 (or
  # another unsupported version) must never be overwritten with 3.12.
  if [[ -x "$environment_python" ]]; then
    "$environment_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing venv must use Python 3.12; move it aside to recreate"'
  elif [[ -e "$environment_python" || -L "$environment_python" ]]; then
    printf 'Existing venv Python is unusable: %s. Move this environment aside to recreate it.\n' "$environment_python" >&2
    return 1
  fi
  if [[ ! -f "$environment/pyvenv.cfg" || ! -x "$environment_python" ]]; then
    # Reusing this directory completes partial creation and preserves its files.
    "$PYTHON_BIN" -m venv "$environment"
  fi
  "$environment_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing venv must use Python 3.12; move it aside to recreate"'
  if ! "$environment_python" -m pip --version >/dev/null 2>&1; then
    "$environment_python" -m ensurepip --upgrade
  fi
  "$environment_python" -m pip install --upgrade --only-binary=:all: --requirement requirements/bootstrap.txt
}
project_install() {
  # Only this local package skips dependency resolution. All third-party model
  # packages above/below install their complete dependencies with pip.
  # Use the same pinned build backend installed above. The default isolated
  # build would perform a second, independent backend installation.
  ".venvs/$1/bin/python" -m pip install --no-build-isolation --no-deps --editable .
  ".venvs/$1/bin/python" -m pip check
  ".venvs/$1/bin/python" -m pip freeze > ".venvs/$1/installed-versions.txt"
}
speech_env() {
  local name="$1" requirements_file="$2" vision="${3:-0}"
  make_env "$name"
  local torch_packages=('torch==2.8.0' 'torchaudio==2.8.0')
  if (( vision )); then torch_packages+=('torchvision==0.23.0'); fi
  ".venvs/$name/bin/python" -m pip install "${torch_packages[@]}" \
    --index-url https://download.pytorch.org/whl/cu128
  ".venvs/$name/bin/python" -m pip install \
    --build-constraint requirements/bootstrap.txt \
    --constraint requirements/constraints-torch.txt --requirement "$requirements_file" \
    --requirement "requirements/$name.txt"
  project_install "$name"
}

make_env app
".venvs/app/bin/python" -m pip install --build-constraint requirements/bootstrap.txt \
  --requirement requirements/locks/app-linux-py312.txt
project_install app
if (( ! APP_ONLY )); then
  speech_env diarize requirements/locks/diarize-linux-py312.txt
  speech_env separate requirements/locks/separate-linux-py312.txt 1
  speech_env asr requirements/locks/asr-linux-py312.txt
  if (( WITH_SPEECHBRAIN )); then speech_env separate-sb requirements/locks/separate-sb-linux-py312.txt; fi
fi
printf '\n%s\n' 'Installation finished. Next: configure HF_TOKEN, install/pull Ollama, then run:' \
  '.venvs/app/bin/python scripts/doctor.py' 'bash scripts/run.sh'
