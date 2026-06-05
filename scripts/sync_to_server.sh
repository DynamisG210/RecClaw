#!/usr/bin/env bash
set -euo pipefail

# Push the local RecClaw code tree to the GPU server while keeping experiment
# artifacts on the server. This is intentionally one-way: local code is source
# of truth; remote results/logs/checkpoints are preserved.
#
# Usage:
#   bash scripts/sync_to_server.sh --dry-run
#   bash scripts/sync_to_server.sh
#
# Optional env overrides:
#   REMOTE_HOST=gpu2
#   REMOTE_DIR=/path/to/remote/RecClaw
#   RSYNC_SSH=/path/to/ssh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-gpu2}"
REMOTE_DIR="${REMOTE_DIR:-RecClaw}"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: bash scripts/sync_to_server.sh [--dry-run] [--host HOST] [--dir REMOTE_DIR]

Defaults:
  --host ${REMOTE_HOST}
  --dir  ${REMOTE_DIR}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n)
      DRY_RUN=1
      shift
      ;;
    --host)
      REMOTE_HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --dir)
      REMOTE_DIR="${2:?missing value for --dir}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required but was not found." >&2
  exit 1
fi

make_ssh_wrapper() {
  local wrapper
  wrapper="$(mktemp "${TMPDIR:-/tmp}/recclaw-rsync-ssh.XXXXXX")"

  local ssh_bin="${RSYNC_SSH:-}"
  if [[ -z "${ssh_bin}" ]]; then
    if [[ -x /mnt/c/Windows/System32/OpenSSH/ssh.exe ]]; then
      ssh_bin="/mnt/c/Windows/System32/OpenSSH/ssh.exe"
    else
      ssh_bin="$(command -v ssh)"
    fi
  fi

  cat > "${wrapper}" <<EOF
#!/usr/bin/env bash
exec "${ssh_bin}" -o BatchMode=yes -o ConnectTimeout=20 "\$@"
EOF
  chmod 700 "${wrapper}"
  printf '%s\n' "${wrapper}"
}

SSH_WRAPPER="$(make_ssh_wrapper)"
trap 'rm -f "${SSH_WRAPPER}"' EXIT

EXCLUDES=(
  ".git/"
  ".codex/"
  ".venv/"
  ".env"
  "__pycache__/"
  "*.pyc"
  ".pytest_cache/"
  ".mypy_cache/"
  ".ruff_cache/"
  ".vscode/"
  "RecClaw_LabLog/"
  "results/"
  "log/"
  "logs/"
  "wandb/"
  "*.log"
  "*.out"
)

printf 'Sync source: %s/\n' "${PROJECT_ROOT}"
printf 'Sync target: %s:%s/\n' "${REMOTE_HOST}" "${REMOTE_DIR}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Mode: dry run\n'
fi

REMOTE_HAS_RSYNC=0
if command -v rsync >/dev/null 2>&1 \
  && "${SSH_WRAPPER}" "${REMOTE_HOST}" "command -v rsync >/dev/null 2>&1"; then
  REMOTE_HAS_RSYNC=1
fi

if [[ "${REMOTE_HAS_RSYNC}" -eq 1 ]]; then
  "${SSH_WRAPPER}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"
  RSYNC_ARGS=(
    -az
    --itemize-changes
    --delete
    -e "${SSH_WRAPPER}"
  )
  for pattern in "${EXCLUDES[@]}"; do
    RSYNC_ARGS+=(--exclude "${pattern}")
  done
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    RSYNC_ARGS=(-n "${RSYNC_ARGS[@]}")
  fi
  rsync "${RSYNC_ARGS[@]}" "${PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
else
  printf 'Remote rsync not found; using tar-over-ssh fallback.\n'
  TAR_ARGS=(-C "${PROJECT_ROOT}")
  TAR_EXCLUDES=("${EXCLUDES[@]}")
  TAR_EXCLUDES+=(
    ".git"
    ".codex"
    ".venv"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
    ".vscode"
    "RecClaw_LabLog"
    "RecClaw_LabLog/*"
    "results"
    "results/*"
    "log"
    "log/*"
    "logs"
    "logs/*"
    "wandb"
    "wandb/*"
  )
  for pattern in "${TAR_EXCLUDES[@]}"; do
    TAR_ARGS+=(--exclude="${pattern}")
  done

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'Files that would be included in tar fallback (first 200):\n'
    tar "${TAR_ARGS[@]}" -cf - . | tar -tf - | sed -n '1,200p'
    exit 0
  fi

  # Remote cleanup is limited to code/documentation paths. Server-side
  # results, logs, checkpoints, and LabLog directories are preserved.
  REMOTE_PREP=$(
    cat <<EOF
set -e
mkdir -p '${REMOTE_DIR}'
cd '${REMOTE_DIR}'
rm -rf configs notes recclaw_ext scripts tests
find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' -o -name 'requirements*.txt' -o -name '.gitignore' \) -delete
tar -xzf -
mkdir -p results log
EOF
  )
  tar "${TAR_ARGS[@]}" -czf - . | "${SSH_WRAPPER}" "${REMOTE_HOST}" "${REMOTE_PREP}"
fi
