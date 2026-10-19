#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
[[ -x .venvs/app/bin/python ]] || { printf 'Run bash scripts/install_ubuntu.sh first.\n' >&2; exit 1; }
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYANNOTE_METRICS_ENABLED=0
export GRADIO_ANALYTICS_ENABLED=False
exec .venvs/app/bin/python -m speaker_lab.app --config configs/default.toml "$@"
