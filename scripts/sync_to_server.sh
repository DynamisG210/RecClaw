#!/usr/bin/env bash
set -euo pipefail

# Template only.
# Fill in REMOTE_USER, REMOTE_HOST, and REMOTE_DIR before using this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_USER="your_user"
REMOTE_HOST="your.server.example.com"
REMOTE_DIR="/path/to/remote/RecClaw"

if [[ "${REMOTE_HOST}" == "your.server.example.com" || "${REMOTE_DIR}" == "/path/to/remote/RecClaw" ]]; then
  echo "Please edit REMOTE_HOST and REMOTE_DIR in scripts/sync_to_server.sh before running rsync." >&2
  exit 1
fi

rsync -avz \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".vscode/" \
  --exclude ".env" \
  --exclude "results/baseline/*.log" \
  --exclude "results/baseline/*.out" \
  --exclude "results/candidates/*.log" \
  --exclude "results/candidates/*.out" \
  "${PROJECT_ROOT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"