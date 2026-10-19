#!/usr/bin/env bash
# Standalone bootstrap. Run as your normal user with sudo access, not with sudo bash.
set -Eeuo pipefail
umask 077

usage() {
  cat <<'HELP'
Usage:
  bash deploy_ubuntu.sh --zip /path/local-speaker-lab.zip --audio /path/two-speakers.wav [options]
  bash deploy_ubuntu.sh --upgrade --zip /path/local-speaker-lab.zip --audio /path/two-speakers.wav
  bash deploy_ubuntu.sh --audio /path/two-speakers.wav [options]

Options:
  --zip FILE                Verify FILE.sha256 and safely extract/reuse its sources.
  --audio FILE              Recording containing two speakers, solo speech, and overlap.
  --destination DIR         Project directory (default: $HOME/speaker-lab/local-speaker-lab).
  --upgrade                 Choose old 0.1.3 and new 0.1.4 directories; preserve the old version.
  --old-directory DIR       Old project directory for --upgrade (used as the prompt default).
  --install-driver          Let the deployment runner install the Ubuntu GPU driver if needed.
  --skip-system-packages    Skip apt; required system packages must already be installed.
  -h, --help                Show this help.

Targets native Ubuntu 24.04 x86_64 with Python 3.12. Without --zip, run this script
from an already extracted project; --destination is then unnecessary. Existing
source files must match the ZIP byte for byte; user changes are never overwritten.
Virtual environments, downloaded models, recordings, and previous logs are retained.
Upgrade mode requires --zip and reads directory selections from /dev/tty. For a
noninteractive repeat run, explicitly supply both --old-directory and --destination.
HF_TOKEN is requested privately by the Python runner, never as a command-line flag.
Accept the model's conditions once at:
https://huggingface.co/pyannote/speaker-diarization-community-1
HELP
}

ZIP_FILE=''
AUDIO_FILE=''
DESTINATION="${HOME}/speaker-lab/local-speaker-lab"
DESTINATION_SET=0
UPGRADE=0
OLD_DIRECTORY=''
OLD_DIRECTORY_SET=0
INSTALL_DRIVER=0
SKIP_SYSTEM_PACKAGES=0
PARSE_ERROR=''
while (($#)); do
  case "$1" in
    --zip|--audio|--destination|--old-directory)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        PARSE_ERROR="A value is required for $1."
        break
      fi
      case "$1" in
        --zip) ZIP_FILE="$2" ;;
        --audio) AUDIO_FILE="$2" ;;
        --destination) DESTINATION="$2"; DESTINATION_SET=1 ;;
        --old-directory) OLD_DIRECTORY="$2"; OLD_DIRECTORY_SET=1 ;;
      esac
      shift 2 ;;
    --upgrade) UPGRADE=1; shift ;;
    --install-driver) INSTALL_DRIVER=1; shift ;;
    --skip-system-packages) SKIP_SYSTEM_PACKAGES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) PARSE_ERROR='Unknown argument. Use --help for supported options.'; break ;;
  esac
done

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ((UPGRADE)) && ((!DESTINATION_SET)); then
  DESTINATION="${HOME}/speaker-lab/local-speaker-lab-0.1.4"
elif [[ -z "$ZIP_FILE" && "$DESTINATION_SET" == 0 ]]; then
  DESTINATION="$SCRIPT_DIRECTORY"
fi
BOOTSTRAP_STAGE='arguments'
BOOTSTRAP_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BOOTSTRAP_LINE=0
BOOTSTRAP_ERROR=''
RUNNER_PID=''
redact_into() {
  local output_name="$1" redacted_text="$2" token="${HF_TOKEN:-}"
  if [[ -n "$token" ]]; then redacted_text="${redacted_text//"$token"/[REDACTED]}"; fi
  while [[ "$redacted_text" =~ hf_[A-Za-z0-9]{8,} ]]; do
    redacted_text="${redacted_text//"${BASH_REMATCH[0]}"/[REDACTED]}"
  done
  printf -v "$output_name" '%s' "$redacted_text"
}
redact() {
  local value
  redact_into value "$1"
  printf '%s' "$value"
}
redact_stream() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    redact "$line"
    printf '\n'
  done
}
json_string() {
  local value char code index LC_ALL=C
  redact_into value "$1"
  printf '"'
  for ((index=0; index<${#value}; index++)); do
    char="${value:index:1}"
    case "$char" in
      '"') printf '\\"' ;;
      '\') printf '\\\\' ;;
      *)
        printf -v code '%d' "'$char"
        if ((code < 32)); then printf '\\u%04x' "$code"; else printf '%s' "$char"; fi ;;
    esac
  done
  printf '"'
}

# Invalid destinations and unwritable parents still get a report. Do not try to
# create /deployment_logs before rejecting --destination /.
if [[ -z "$DESTINATION" ]]; then
  PARSE_ERROR="${PARSE_ERROR:-The project destination must not be empty.}"
elif ! DESTINATION="$(realpath -m -- "$DESTINATION" 2>/dev/null)"; then
  PARSE_ERROR="${PARSE_ERROR:-Cannot resolve the project destination.}"
elif [[ "$DESTINATION" == / ]]; then
  PARSE_ERROR="${PARSE_ERROR:-The filesystem root cannot be a project destination.}"
fi
LOG_DIRECTORY=''
LOG_FALLBACK=0
if [[ -n "$DESTINATION" && "$DESTINATION" != / ]]; then
  LOG_PARENT="$(dirname -- "$DESTINATION")/deployment_logs"
  # Upgrade selections must be validated before writing anything into either
  # version. In particular, a mistaken nested destination must not alter old.
  if ((UPGRADE)); then LOG_PARENT="${HOME}/speaker-lab/deployment_logs"; fi
  if mkdir -p -- "$LOG_PARENT" 2>/dev/null; then
    LOG_DIRECTORY="$(mktemp -d "$LOG_PARENT/deploy-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX" 2>/dev/null)" || LOG_DIRECTORY=''
  fi
fi
if [[ -z "$LOG_DIRECTORY" ]]; then
  LOG_FALLBACK=1
  LOG_PARENT="${HOME:-/tmp}/speaker-lab/deployment_logs"
  if mkdir -p -- "$LOG_PARENT" 2>/dev/null; then
    LOG_DIRECTORY="$(mktemp -d "$LOG_PARENT/deploy-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX" 2>/dev/null)" || LOG_DIRECTORY=''
  fi
fi
if [[ -z "$LOG_DIRECTORY" ]]; then
  LOG_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/speaker-lab-deploy.XXXXXXXX" 2>/dev/null)" || LOG_DIRECTORY=''
fi
if [[ -z "$LOG_DIRECTORY" ]]; then
  printf 'ERROR: No writable directory is available for deployment logs or a failure report.\n' >&2
  exit 1
fi
LOG_DIRECTORY="$(realpath -- "$LOG_DIRECTORY")"
LOG_FILE="$LOG_DIRECTORY/bootstrap.log"
exec 3>&2
exec > >(redact_stream | tee -a "$LOG_FILE") 2>&1

finish_bootstrap() {
  local exit_code="$?" status='FAILED'
  trap - EXIT ERR INT TERM
  set +e
  if ((exit_code == 0)); then status='COMPLETED'; fi
  if ((exit_code == 20)); then status='REBOOT_REQUIRED'; fi
  {
    printf '{\n  "status": "%s",\n  "exit_code": %s,\n  "stage": ' "$status" "$exit_code"
    json_string "$BOOTSTRAP_STAGE"
    printf ',\n  "started_at": '; json_string "$BOOTSTRAP_START"
    printf ',\n  "finished_at": '; json_string "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf ',\n  "failed_line": %s,\n  "project_directory": ' "$BOOTSTRAP_LINE"
    json_string "$DESTINATION"
    printf ',\n  "error": '; json_string "$BOOTSTRAP_ERROR"
    printf ',\n  "log_directory": '; json_string "$LOG_DIRECTORY"
    printf '\n}\n'
  } > "$LOG_DIRECTORY/bootstrap-report.json"
  {
    printf '# Deployment bootstrap: %s\n\n' "$status"
    printf -- '- Stage: `%s`\n- Exit code: `%s`\n' "$BOOTSTRAP_STAGE" "$exit_code"
    printf -- '- Project: `%s`\n- Logs: `%s`\n' "$(redact "$DESTINATION")" "$(redact "$LOG_DIRECTORY")"
    if [[ -n "$BOOTSTRAP_ERROR" ]]; then printf '\nFailure: %s\n' "$(redact "$BOOTSTRAP_ERROR")"; fi
    printf '\nSee `bootstrap.log` for bootstrap diagnostics and `deployment-report.md` for the deployment and acceptance report, when generated.\n'
    if ((exit_code == 20)); then
      printf '\nThe deployment requires a system reboot. Reboot Ubuntu, then repeat the same command to continue.\n'
    elif ((exit_code != 0)); then
      printf '\nThe bootstrap stopped at the first failure. Fix the reported cause and repeat the same command; existing environments and downloads are retained.\n'
    fi
  } > "$LOG_DIRECTORY/bootstrap-report.md"
  if [[ ! -f "$LOG_DIRECTORY/deployment-report.md" ]]; then
    printf '\nAcceptance has not started or did not complete. Bootstrap report:\n'
    cat -- "$LOG_DIRECTORY/bootstrap-report.md"
    if [[ -f "$LOG_DIRECTORY/upgrade-report.json" ]]; then
      printf '\nUpgrade/cache report:\n'
      cat -- "$LOG_DIRECTORY/upgrade-report.json"
    fi
  fi
  printf '\nBootstrap %s (exit %s). Logs: %s\n' "$status" "$exit_code" "$LOG_DIRECTORY"
  printf 'Bootstrap report: %s\nBootstrap log: %s\n' "$LOG_DIRECTORY/bootstrap-report.md" "$LOG_FILE"
  if [[ -f "$LOG_DIRECTORY/upgrade-report.json" ]]; then
    printf 'Upgrade report: %s\n' "$LOG_DIRECTORY/upgrade-report.json"
  fi
  exit "$exit_code"
}
bootstrap_error() {
  local exit_code="$1"
  BOOTSTRAP_LINE="$2"
  BOOTSTRAP_ERROR="${BOOTSTRAP_ERROR:-Command failed at line $BOOTSTRAP_LINE (exit $exit_code). See bootstrap.log.}"
  return "$exit_code"
}
trap 'bootstrap_error "$?" "$LINENO"' ERR
trap finish_bootstrap EXIT
stop_bootstrap() {
  local exit_code="$1"
  BOOTSTRAP_ERROR="$2"
  trap '' INT TERM
  if [[ -n "$RUNNER_PID" ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
    # Background jobs may inherit SIGINT ignored from noninteractive Bash.
    # The Python runner explicitly handles TERM, cancels its subprocess group,
    # and writes its failure report before this shell writes the bootstrap report.
    kill -TERM "$RUNNER_PID" 2>/dev/null || true
    wait "$RUNNER_PID" || true
  fi
  exit "$exit_code"
}
trap 'stop_bootstrap 130 "Interrupted by user; remaining steps were not executed."' INT
trap 'stop_bootstrap 143 "Terminated; remaining steps were not executed."' TERM

die() { BOOTSTRAP_ERROR="$1"; printf 'ERROR: %s\n' "$(redact "$1")" >&2; exit "${2:-1}"; }
printf 'Deployment logs: %s\n' "$LOG_DIRECTORY"
if ((LOG_FALLBACK)); then printf 'The preferred log directory was unavailable or invalid; using the fallback above.\n'; fi
[[ -z "$PARSE_ERROR" ]] || die "$PARSE_ERROR" 2
select_upgrade_directories() {
  local selection=''
  OLD_DIRECTORY="${OLD_DIRECTORY:-${HOME}/speaker-lab/local-speaker-lab}"
  # Directory selections use the controlling terminal. A recording piped into
  # stdin or a tee logging pipeline must never supply prompt answers.
  if { exec 4<>/dev/tty; } 2>/dev/null; then
    printf 'Upgrade 0.1.3 -> 0.1.4; the old project will be retained.\n' >&4
    printf 'Old 0.1.3 directory [%s]: ' "$(redact "$OLD_DIRECTORY")" >&4
    IFS= read -r selection <&4 || die 'Directory selection was cancelled.' 2
    if [[ -n "$selection" ]]; then OLD_DIRECTORY="$selection"; fi
    printf 'New 0.1.4 directory [%s]: ' "$(redact "$DESTINATION")" >&4
    IFS= read -r selection <&4 || die 'Directory selection was cancelled.' 2
    if [[ -n "$selection" ]]; then DESTINATION="$selection"; fi
    exec 4>&-
  elif ((!OLD_DIRECTORY_SET || !DESTINATION_SET)); then
    die 'Upgrade directory selection requires a terminal; otherwise supply both --old-directory and --destination.' 2
  fi
  [[ "$OLD_DIRECTORY" != '~/'* ]] || OLD_DIRECTORY="${HOME}/${OLD_DIRECTORY:2}"
  [[ "$DESTINATION" != '~/'* ]] || DESTINATION="${HOME}/${DESTINATION:2}"
  [[ -n "$OLD_DIRECTORY" && -n "$DESTINATION" ]] || die 'Old and new project directories must not be empty.' 2
  OLD_DIRECTORY="$(realpath -m -- "$OLD_DIRECTORY")" || die 'Cannot resolve the old project directory.' 2
  DESTINATION="$(realpath -m -- "$DESTINATION")" || die 'Cannot resolve the new project directory.' 2
  [[ "$OLD_DIRECTORY" != / && "$DESTINATION" != / ]] || die 'The filesystem root cannot be a project directory.' 2
  [[ "$OLD_DIRECTORY" != "$DESTINATION" && "$DESTINATION/" != "$OLD_DIRECTORY/"* && "$OLD_DIRECTORY/" != "$DESTINATION/"* ]] \
    || die 'Old and new project directories must be separate and must not contain one another.' 2
  [[ -d "$OLD_DIRECTORY" && -f "$OLD_DIRECTORY/pyproject.toml" && -f "$OLD_DIRECTORY/scripts/deploy_and_accept.py" ]] \
    || die 'The old directory must be an extracted local-speaker-lab 0.1.3 project.' 2
  printf 'Upgrade old directory: %s\nUpgrade new directory: %s\n' "$OLD_DIRECTORY" "$DESTINATION"
}
if ((UPGRADE)); then
  BOOTSTRAP_STAGE='upgrade-directories'
  [[ -n "$ZIP_FILE" ]] || die '--upgrade requires --zip pointing to the new 0.1.4 archive.' 2
  select_upgrade_directories
elif ((OLD_DIRECTORY_SET)); then
  die '--old-directory is only valid with --upgrade.' 2
fi
[[ "$DESTINATION" != / ]] || die 'The filesystem root cannot be a project destination.' 2
[[ -n "$AUDIO_FILE" ]] || die '--audio is required; use --help for an example.' 2
[[ -f "$AUDIO_FILE" && -s "$AUDIO_FILE" ]] || die 'The supplied recording must be a readable, nonempty regular file.' 2
AUDIO_FILE="$(realpath -e -- "$AUDIO_FILE")"

BOOTSTRAP_STAGE='platform'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'This script requires native Ubuntu 24.04 x86_64.'
[[ -r /etc/os-release ]] || die '/etc/os-release is missing.'
# /etc/os-release is the operating system's root-owned metadata.
. /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] || die 'The locked deployment currently supports Ubuntu 24.04 only.'
if [[ -r /proc/sys/kernel/osrelease ]] && grep -qi microsoft /proc/sys/kernel/osrelease; then
  die 'Use native Ubuntu; this deployment does not install a WSL host driver.'
fi
((EUID != 0)) || die 'Run as a normal user with sudo access, not with sudo bash.'

BOOTSTRAP_STAGE='system-packages'
if ((!SKIP_SYSTEM_PACKAGES)); then
  SYSTEM_PACKAGES=(python3.12 python3.12-venv ffmpeg libsndfile1 libgl1
    libglib2.0-0t64 sox libsox-fmt-all git curl ca-certificates unzip jq zstd
    ubuntu-drivers-common)
  MISSING_PACKAGES=()
  for package in "${SYSTEM_PACKAGES[@]}"; do
    package_status="$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)"
    if [[ "$package_status" != installed ]]; then MISSING_PACKAGES+=("$package"); fi
  done
  if ((${#MISSING_PACKAGES[@]})); then
    command -v sudo >/dev/null || die 'sudo is required to install missing system packages.'
    printf 'Installing missing system packages: %s\n' "${MISSING_PACKAGES[*]}"
    env -u HF_TOKEN sudo apt-get update
    env -u HF_TOKEN sudo apt-get install -y --no-install-recommends "${MISSING_PACKAGES[@]}"
  else
    printf 'System packages already installed; apt was skipped.\n'
  fi
else
  printf 'System package installation skipped by request.\n'
fi
command -v python3.12 >/dev/null || die 'python3.12 is required; install python3.12 and python3.12-venv.'
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'

if ((UPGRADE)); then
  BOOTSTRAP_STAGE='upgrade-version-validation'
  python3.12 - "$OLD_DIRECTORY" "$DESTINATION" <<'PYTHON_UPGRADE_VALIDATE'
from pathlib import Path
import sys
import tomllib

old, new = (Path(value).resolve() for value in sys.argv[1:])
if old == new or old in new.parents or new in old.parents:
    raise RuntimeError("Old and new project directories must be separate and not nested")
for directory, expected in ((old, "0.1.3"), (new, "0.1.4")):
    metadata = directory / "pyproject.toml"
    if directory == new and not metadata.exists():
        if directory.exists() and (not directory.is_dir() or any(item.name != ".cache" for item in directory.iterdir())):
            raise RuntimeError("The new directory must be empty, cache-only, or a 0.1.4 project")
        continue
    project = tomllib.loads(metadata.read_text(encoding="utf-8"))["project"]
    if project.get("name") != "local-speaker-lab" or project.get("version") != expected:
        raise RuntimeError(f"Expected local-speaker-lab {expected}: {directory}")
print("Upgrade versions verified; old source files will be preserved.")
PYTHON_UPGRADE_VALIDATE
fi

BOOTSTRAP_STAGE='source-extraction'
if [[ -n "$ZIP_FILE" ]]; then
  [[ -f "$ZIP_FILE" ]] || die 'The project ZIP does not exist.'
  ZIP_FILE="$(realpath -e -- "$ZIP_FILE")"
  [[ -f "$ZIP_FILE.sha256" ]] || die 'The ZIP checksum sidecar is missing. Place the matching .zip.sha256 beside the ZIP.'
  expected_checksum=''
  read -r expected_checksum checksum_filename < "$ZIP_FILE.sha256" || [[ -n "$expected_checksum" ]]
  [[ "$expected_checksum" =~ ^[a-fA-F0-9]{64}$ ]] || die 'The checksum sidecar must start with a SHA-256 digest.'
  actual_checksum="$(sha256sum < "$ZIP_FILE")"
  actual_checksum="${actual_checksum%% *}"
  [[ "${actual_checksum,,}" == "${expected_checksum,,}" ]] || die 'ZIP SHA-256 verification failed; download the ZIP and checksum again.'
  printf 'ZIP SHA-256 verified: %s\n' "$actual_checksum"
  printf '%s  %s\n' "$actual_checksum" "$(redact "$(basename -- "$ZIP_FILE")")" > "$LOG_DIRECTORY/source-sha256.txt"
  python3.12 - "$ZIP_FILE" "$DESTINATION" "$UPGRADE" <<'PYTHON_SAFE_EXTRACT'
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import tomllib
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def require_plain_path(root, relative):
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError("Refusing an existing symbolic link: " + str(current))
        if current.exists() and current != root / relative and not current.is_dir():
            raise RuntimeError("Expected a directory: " + str(current))


archive_path = Path(sys.argv[1])
destination = Path(sys.argv[2])
upgrade = len(sys.argv) > 3 and sys.argv[3] == "1"
required = {"pyproject.toml", "scripts/deploy_and_accept.py", "deploy_ubuntu.sh"}
if upgrade:
    required.add("scripts/upgrade_cache.py")
if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
    raise RuntimeError("The project destination must be an ordinary directory")
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path) as archive:
    files = {}
    seen = set()
    total_bytes = 0
    if len(archive.infolist()) > 20000:
        raise RuntimeError("Too many files in source ZIP")
    for info in archive.infolist():
        name = info.orig_filename
        parts = name.rstrip("/").split("/")
        if ("\\" in name or "\x00" in name or any(p in {"", ".", ".."} for p in parts)
                or not parts or parts[0] != "local-speaker-lab"
                or (len(parts) == 1 and not info.is_dir())):
            raise RuntimeError("Unsafe ZIP member path: " + repr(name))
        if name in seen:
            raise RuntimeError("Duplicate ZIP member: " + repr(name))
        seen.add(name)
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or info.flag_bits & 1:
            raise RuntimeError("Links, special files, and encrypted ZIP members are unsupported")
        if info.is_dir():
            if kind not in {0, stat.S_IFDIR}:
                raise RuntimeError("Invalid ZIP directory")
            continue
        if kind == stat.S_IFDIR:
            raise RuntimeError("Invalid ZIP file")
        relative = PurePosixPath(*parts[1:])
        total_bytes += info.file_size
        if total_bytes > 256 * 1024 * 1024:
            raise RuntimeError("Source ZIP exceeds the 256 MiB safety limit")
        files[relative] = info
    if not required.issubset({str(path) for path in files}):
        raise RuntimeError("ZIP is missing the project bootstrap, runner, or metadata")
    with tempfile.TemporaryDirectory(prefix=".speaker-lab-unpack-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "local-speaker-lab"
        staged.mkdir()
        # Fully decode every member and check its CRC before touching project files.
        for relative, info in files.items():
            output = staged / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open("xb") as target:
                shutil.copyfileobj(source, target)
        if upgrade:
            project = tomllib.loads((staged / "pyproject.toml").read_text(encoding="utf-8"))["project"]
            if project.get("name") != "local-speaker-lab" or project.get("version") != "0.1.4":
                raise RuntimeError("Upgrade requires the local-speaker-lab 0.1.4 ZIP")
        if not destination.exists():
            os.replace(staged, destination)
            print("Extracted project:", destination)
        else:
            for relative in files:
                require_plain_path(destination, relative)
                existing = destination / relative
                if existing.exists() and (not existing.is_file() or digest(existing) != digest(staged / relative)):
                    raise RuntimeError("Existing source differs from ZIP; preserve your changes and use a new --destination: " + str(existing))
            created = 0
            # Record the validated version first. Each file appears atomically,
            # so interruption cannot leave a truncated file that resembles a
            # user edit on the next run. Staging is on this project's volume.
            for relative in sorted(files, key=lambda path: (str(path) != "pyproject.toml", str(path))):
                existing = destination / relative
                if existing.exists():
                    continue
                existing.parent.mkdir(parents=True, exist_ok=True)
                os.link(staged / relative, existing)
                created += 1
            print("Reused matching project sources; added", created, "missing files:", destination)
PYTHON_SAFE_EXTRACT
else
  [[ -f "$DESTINATION/scripts/deploy_and_accept.py" && -f "$DESTINATION/pyproject.toml" ]] \
    || die 'Without --zip, this script must be inside an extracted project (or use --destination to select one).'
  printf 'Using extracted project: %s\n' "$DESTINATION"
fi

if ((UPGRADE)); then
  BOOTSTRAP_STAGE='upgrade-cache'
  [[ -f "$DESTINATION/scripts/upgrade_cache.py" ]] || die 'The supplied archive does not contain the 0.1.4 upgrade cache helper.'
  python3.12 "$DESTINATION/scripts/upgrade_cache.py" --old "$OLD_DIRECTORY" --new "$DESTINATION" \
    --report "$LOG_DIRECTORY/upgrade-report.json"
fi

BOOTSTRAP_STAGE='deployment-and-acceptance'
cd -- "$DESTINATION"
# Keep sudo's interactive authorization on the original terminal; do not capture
# a password prompt in a subprocess pipe or include credentials in any log.
command -v sudo >/dev/null || die 'sudo is required for Ollama service and optional driver setup.'
printf 'Checking sudo access for system service setup.\n'
env -u HF_TOKEN sudo -v >&3 2>&3
RUNNER_ARGUMENTS=(--audio "$AUDIO_FILE" --log-dir "$LOG_DIRECTORY")
if ((UPGRADE)); then RUNNER_ARGUMENTS+=(--upgrade-report "$LOG_DIRECTORY/upgrade-report.json"); fi
if ((INSTALL_DRIVER)); then RUNNER_ARGUMENTS+=(--install-driver); fi
python3.12 -u scripts/deploy_and_accept.py "${RUNNER_ARGUMENTS[@]}" <&0 &
RUNNER_PID=$!
wait "$RUNNER_PID"
RUNNER_PID=''
BOOTSTRAP_STAGE='complete'
